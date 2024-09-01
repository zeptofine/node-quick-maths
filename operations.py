import ast
import difflib
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Union

import bpy

from .constants import (
    BAD_MATH_AST_NODES,
    SHADER_NODE_BASIC_OPS,
    VALID_MATH_FUNCTIONS,
    VARIABLE_NAME,
    Function,
)
from .rustlike_result import Err, Ok, Result

_InpT = TypeVar("_InpT")


@dataclass(frozen=True)
class Operation(Generic[_InpT]):
    name: str
    """ The name of the operation, as described in the bpy API. """

    inputs: tuple[_InpT, ...]
    """ Inputs that will be connected during composing"""

    @classmethod
    @abstractmethod
    def validate(
        cls,
        e: ast.Module,
        bad_nodes: tuple[type[ast.expr], ...] = BAD_MATH_AST_NODES,
        functions: dict[str, Function] = VALID_MATH_FUNCTIONS,
    ) -> Result[tuple, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse(cls, e: ast.expr) -> _InpT:
        raise NotImplementedError

    def to_tree(self, sort_mode="NONE"):
        return Tree(variables=self.variables(sort_mode), root=self)

    def variables(self, sort_mode="NONE") -> list[VARIABLE_NAME]:
        if sort_mode == "INSERTION":
            v: list[VARIABLE_NAME] = []
            for input in self.inputs:
                if isinstance(input, Operation):
                    v.extend(var for var in input.variables(sort_mode) if var not in v)
                elif isinstance(input, VARIABLE_NAME):
                    v.append(input)
            return v

        vars: set[VARIABLE_NAME] = set()
        for input in self.inputs:
            if isinstance(input, Operation):
                vars.update(input.variables(sort_mode))
            elif isinstance(input, VARIABLE_NAME):
                vars.add(input)
        if sort_mode == "ALPHABET":
            return sorted(vars)
        return list(vars)

    @abstractmethod
    def create_node(self, nt: bpy.types.NodeTree) -> bpy.types.Node:
        raise NotImplementedError

    def generate(
        self, nt: bpy.types.NodeTree
    ) -> tuple[
        bpy.types.Node,
        list[list[bpy.types.Node]],
        list[tuple[str, bpy.types.NodeInputs]],
    ]:
        """
        Creates nodes. returns a list of tuples of
        (
            toplevel node,
            subtrees
            input socket
        )
        """
        parent = self.create_node(nt)

        oinputs: list[tuple[str, bpy.types.NodeInputs]] = []
        children: list[bpy.types.Node] = []

        # [
        #     [                    child1,                                         child2                    ],
        #     [       child1.1,                child1.2,              child2.1,               child2.2       ],
        #     [child1.1.1, child1.1.2, child1.2.1, child1.2.2, child2.1.1, child2.1.2, child2.2.1, child2.2.2],
        # ]
        layers: list[list[bpy.types.Node]] = []

        processors = self.input_type_processors()

        for idx, child in enumerate(self.inputs):
            chosen_processor = None
            for t, processor in processors:
                if isinstance(child, t):
                    chosen_processor = processor
                    break

            if chosen_processor is not None:
                processor(nt, parent, oinputs, layers, children, idx, child)
            else:
                parent.inputs[idx].default_value = child

        if children:
            layers.insert(0, children)

        return parent, layers, oinputs

    def input_type_processors(
        self,
    ) -> list[
        tuple[
            type,
            Callable[
                [
                    bpy.types.NodeTree,
                    bpy.types.Node,
                    list[tuple[str, bpy.types.NodeInputs]],
                    list[list[bpy.types.Node]],
                    list[bpy.types.Node],
                    int,
                    Any,
                ],
                None,
            ],
        ]
    ]:
        return [
            (Operation, self._process_child_op),
            (str, self._process_variable),
        ]

    def _process_child_op(
        self,
        nt: bpy.types.NodeTree,
        parent: bpy.types.Node,
        oinputs: list[tuple[str, bpy.types.NodeInputs]],
        layers: list[list[bpy.types.Node]],
        children: list[bpy.types.Node],
        index: int,
        child: "Operation",
    ):
        node, sublayers, inputs = child.generate(nt)
        oinputs.extend(inputs)
        nt.links.new(node.outputs[0], parent.inputs[index])

        # add each layer to the tree
        if sublayers:
            for idx, layer in enumerate(sublayers):
                if len(layers) <= idx:
                    layers.append([])

                layers[idx].extend(layer)

        # add the node to children
        children.append(node)

    def _process_variable(
        self,
        nt: bpy.types.NodeTree,
        parent: bpy.types.Node,
        oinputs: list[tuple[str, bpy.types.NodeInputs]],
        layers: list[list[bpy.types.Node]],
        children: list[bpy.types.Node],
        index: int,
        child: "Operation",
    ):  # add it to the oinputs
        oinputs.append((child, parent.inputs[index]))


ShaderMathNodeInput = int | float | str | Operation["ShaderMathNodeInput"]


class ShaderMathOperation(Operation[ShaderMathNodeInput]):
    def create_node(self, nt: bpy.types.NodeTree) -> bpy.types.Node:
        node = nt.nodes.new("ShaderNodeMath")
        node.operation = self.name
        return node

    @classmethod
    def _check_bad_type(cls, node: ast.Constant) -> Result[tuple, str]:
        if not isinstance(node.value, int | float):
            return Err(f"Constants cannot be anything other than ints or floats.\n{node.value} is disallowed")
        return Ok(())

    @classmethod
    def validate_node(
        cls,
        node: ast.Expr,
        bad_nodes: tuple[type[ast.expr], ...],
        functions: dict[str, Function],
    ) -> Result[tuple, str]:
        # check if node is bad
        if any(isinstance(node, bad_node) for bad_node in bad_nodes):
            return Err(f"Do not use node of type: {type(node)} ")

        # check if node is a constant and it is a disallowed type
        if isinstance(node, ast.Constant):
            r = cls._check_bad_type(node)
            if r.is_err():
                return r

        # check if node is a call and it has an allowed function name
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return Err("Functions may only be called by name")
            name = node.func
            function = functions.get(name.id)
            if function is None:
                errmsg = f"Unrecognized function name: '{name.id}'"

                if matches := difflib.get_close_matches(name.id, list(functions)):
                    return Err(f"{errmsg}\nDid you mean one of these?\n{', '.join(matches)}")
                return Err(errmsg)

            # check if the number of arguments align with the number of arguments in the GOOD_CALLS
            elif all(len(node.args) != x for x in function.input_nums):
                return Err(
                    f"Function {name.id} is allowed, but\nthe number of arguments is wrong\n({len(node.args)} is not in {function.input_nums})"
                )

        return Ok(())

    @classmethod
    def validate(
        cls,
        e: ast.Module,
        bad_nodes: tuple[type[ast.expr], ...] = BAD_MATH_AST_NODES,
        functions: dict[str, Function] = VALID_MATH_FUNCTIONS,
    ) -> Result[tuple, str]:
        # check that the node in the ast body is just an Expr
        expr: ast.stmt

        if not e.body:
            return Err("Expression is empty")

        if not isinstance((expr := e.body[0]), ast.Expr):  # Unsure how this can show up
            return Err("Invalid expression type. Only create math expressions!")

        for node in ast.walk(expr):
            r = cls.validate_node(node, bad_nodes, functions)
            if r.is_err():
                return r

        return Ok(())

    @classmethod
    def parse(cls, e: ast.expr) -> ShaderMathNodeInput:
        def parse(e) -> ShaderMathNodeInput:
            """Parses the expression, carrying over settings."""
            return cls.parse(e)

        match e:
            case ast.Constant(value=v) | ast.Name(id=v):
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

            case ast.BinOp(op=op, left=left, right=right):
                if (t := type(op)) in SHADER_NODE_BASIC_OPS:
                    return cls(
                        name=SHADER_NODE_BASIC_OPS[t],
                        inputs=(parse(left), parse(right)),
                    )

                msg = f"Unhandled operation {op}"
                raise NotImplementedError(msg)

            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value)):
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


class CompositorSimpleMathOperation(ShaderMathOperation):
    def create_node(self, nt: bpy.types.NodeTree) -> bpy.types.Node:
        node: bpy.types.CompositorNodeMath = nt.nodes.new("CompositorNodeMath")
        node.operation = self.name
        return node


class TextureSimpleMathOperation(ShaderMathOperation):
    def create_node(self, nt: bpy.types.NodeTree) -> bpy.types.Node:
        node: bpy.types.TextureNodeMath = nt.nodes.new("TextureNodeMath")
        node.operation = self.name
        return node


ShaderVectorMathNodeInput = Union[
    int,
    float,
    str,
    "tuple[ShaderVectorMathNodeInput, ShaderVectorMathNodeInput, ShaderVectorMathNodeInput]",
    "Operation[ShaderVectorMathNodeInput]",
]


@dataclass
class Tree(Generic[_InpT]):
    variables: list[VARIABLE_NAME]
    root: Operation[_InpT]
