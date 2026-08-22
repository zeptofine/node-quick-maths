import ast
import traceback
from collections.abc import Generator
from dataclasses import dataclass
from typing import TypeAlias

import bpy
from bpy.types import Node, NodeGroup, NodeSocket, NodeTree

from .node_creation import NodeCreator
from .constants import (
    SHADER_NODE_BASIC_OPS,
    VALID_MATH_FUNCTIONS,
    VARIABLE_NAME,
)
from .rustlike_result import Err, Ok, Result

ShaderMathNodeInput: TypeAlias = "int | float | str | Operation"


@dataclass(frozen=True)
class Operation:
    name: str
    """ The name of the operation, as described in the bpy API. """

    inputs: tuple[ShaderMathNodeInput, ...]
    """ Inputs that will be connected during composing"""

    def variables(self, sort_mode="NONE") -> list[VARIABLE_NAME]:
        if sort_mode == "INSERTION":
            v: set[VARIABLE_NAME] = set()
            for input in self.inputs:
                if isinstance(input, Operation):
                    v.update(var for var in input.variables(sort_mode))
                elif isinstance(input, VARIABLE_NAME):
                    v.add(input)
            return list(v)

        vars: set[VARIABLE_NAME] = set()
        for input in self.inputs:
            if isinstance(input, Operation):
                vars.update(input.variables(sort_mode))
            elif isinstance(input, VARIABLE_NAME):
                vars.add(input)
        if sort_mode == "ALPHABET":
            return sorted(vars)
        return list(vars)

    @classmethod
    def parse(cls, e: ast.expr) -> ShaderMathNodeInput:
        def parse(e) -> ShaderMathNodeInput:
            """Parses the expression, carrying over settings."""
            return cls.parse(e)

        match e:
            case ast.Constant(value=v) | ast.Name(id=v) if isinstance(v, (str, int, float)):
                return v
            case ast.Expr(value=v):
                return parse(v)

            # check for Multiply Add
            case (
                ast.BinOp(op=ast.Add(), left=ast.BinOp(op=ast.Mult(), left=a, right=b), right=c)
                | ast.BinOp(op=ast.Add(), right=ast.BinOp(op=ast.Mult(), left=a, right=b), left=c)
            ):
                return cls(name="MULTIPLY_ADD", inputs=(parse(a), parse(b), parse(c)))

            # check for inverse sqrt
            case ast.BinOp(
                op=ast.Div(),
                left=ast.Constant(value=1),
                right=ast.Call(func=ast.Name(id="sqrt"), args=[first_argument, *_]),
            ):
                return cls(name="INVERSE_SQRT", inputs=(parse(first_argument),))

            # check for exponent
            case ast.BinOp(left=ast.Name(id="e"), op=ast.Pow(), right=right):
                return cls("EXPONENT", inputs=(parse(right),))

            case ast.BinOp(left=left, right=right, op=ast.FloorDiv()):
                return cls(
                    name="FLOOR",
                    inputs=(
                        cls(
                            name="DIVIDE",
                            inputs=(parse(left), parse(right)),
                        ),
                    ),
                )

            case ast.BinOp(op=op, left=left, right=right) if (t := type(op)) in SHADER_NODE_BASIC_OPS:
                return cls(
                    name=SHADER_NODE_BASIC_OPS[t],
                    inputs=(parse(left), parse(right)),
                )

            case ast.BinOp(op=op, left=left, right=right):
                msg = f"Unhandled operation {op}"
                raise NotImplementedError(msg)

            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value)) if isinstance(value, (float, int)):
                return -value

            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return cls(name="MULTIPLY", inputs=(parse(operand), -1))

            case ast.Compare(
                ops=[ast.Lt() | ast.LtE(), *_],
                left=left,
                comparators=comparators,
            ):
                return cls("LESS_THAN", inputs=(parse(left), parse(comparators[0])))
            case ast.Compare(
                ops=[ast.Gt() | ast.GtE(), *_],
                left=left,
                comparators=comparators,
            ):
                return cls("GREATER_THAN", inputs=(parse(left), parse(comparators[0])))
            case ast.Compare(
                ops=[ast.Eq(), *_],
                left=left,
                comparators=comparators,
            ):
                return cls("COMPARE", inputs=(parse(left), parse(comparators[0]), 0.5))

            case ast.Call(args=args, func=ast.Name(id=identifier)):
                inputs = tuple(parse(arg) for arg in args)
                return cls(
                    name=VALID_MATH_FUNCTIONS[identifier].enum_value,
                    inputs=inputs,
                )

        msg = f"Unhandled expression {ast.dump(e, indent=4)}"
        raise NotImplementedError(msg)


@dataclass(frozen=True)
class DepthedNode:
    depth: int
    node: Node


@dataclass(frozen=True)
class TaggedInput:
    var: str
    socket: NodeSocket


LayerList: TypeAlias = list[list[Node]]


@dataclass
class Tree:
    original_expression: str

    variables: list[VARIABLE_NAME]
    root: Operation

    creator: type[NodeCreator]

    @classmethod
    def parse(cls, s: str, e: ast.expr, creator: type[NodeCreator], sort_mode: str) -> Result["Tree", str]:
        try:
            parsed = Operation.parse(e)
        except Exception as e:
            traceback.print_exc()
            return Err(str(e))
        if not isinstance(parsed, Operation):
            return Err("Parsed expression is not an Operation")

        return Ok(Tree(s, parsed.variables(sort_mode), parsed, creator))

    def layers_and_connections(self, nt: bpy.types.NodeTree) -> tuple[LayerList, list[TaggedInput]]:
        noded: dict[int, list[bpy.types.Node]] = {}
        inputs: list[TaggedInput] = []

        for out in self.__nodes_and_var_connections(self.root, nt):
            if isinstance(out, DepthedNode):
                layer = noded.setdefault(out.depth, [])
                layer.append(out.node)
            else:
                inputs.append(out)

        nodes: list[list[bpy.types.Node]] = []
        for _, layer in sorted(noded.items()):
            nodes.append(layer)

        return nodes, inputs

    def __nodes_and_var_connections(
        self,
        operation: Operation,
        nt: NodeTree,
        depth: int = 0,
        parent_socket: NodeSocket | None = None,
    ) -> Generator[DepthedNode | TaggedInput, None, None]:
        node = self.creator.math_node(nt, operation.name)
        yield DepthedNode(depth, node)

        if parent_socket is not None:
            nt.links.new(node.outputs[0], parent_socket)

        for idx, inp in enumerate(operation.inputs):
            if isinstance(inp, str):
                yield TaggedInput(inp, node.inputs[idx])
                continue
            elif isinstance(inp, Operation):
                yield from self.__nodes_and_var_connections(inp, nt, depth + 1, node.inputs[idx])

    def new_group_tree(self) -> NodeTree:
        return self.creator.node_group(self.original_expression)

    @property
    def group_type(self) -> str:
        return self.creator.group_type
