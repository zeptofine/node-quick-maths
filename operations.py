import ast
from abc import abstractmethod
from dataclasses import dataclass
import difflib
from typing import Self

import bpy
from .constants import (
    BAD_MATH_AST_NODES,
    SHADER_MATH_CALLS,
    SHADER_NODE_BASIC_OPS,
    VARIABLE_NAME,
)

from .rustlike_result import Err, Ok, Result


@dataclass(frozen=True)
class Operation:
    name: str
    """ The name of the operation, as described in the bpy API. """

    inputs: "tuple[_Input, ...]"
    """ Inputs that will be connected during composing"""

    @classmethod
    @abstractmethod
    def validate(
        cls,
        e: ast.Module,
        bad_nodes: tuple[type[ast.expr], ...] = BAD_MATH_AST_NODES,
        functions: dict[str, tuple[tuple[int, ...], str]] = SHADER_MATH_CALLS,
    ) -> Result[tuple, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse(cls, e: ast.expr) -> int | float | str | Self:
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

        oinputs = []
        children: list[bpy.types.Node] = []

        # [
        #     [                    child1,                                         child2                    ],
        #     [       child1.1,                child1.2,              child2.1,               child2.2       ],
        #     [child1.1.1, child1.1.2, child1.2.1, child1.2.2, child2.1.1, child2.1.2, child2.2.1, child2.2.2],
        # ]
        layers: list[list[bpy.types.Node]] = []

        for idx, child in enumerate(self.inputs):
            if isinstance(child, Operation):
                node, sublayers, inputs = child.generate(nt)
                oinputs.extend(inputs)
                nt.links.new(node.outputs[0], parent.inputs[idx])

                # add each layer to the tree
                if sublayers:
                    for idx, layer in enumerate(sublayers):
                        if len(layers) <= idx:
                            layers.append([])

                        layers[idx].extend(layer)

                # add the node to children
                children.append(node)

            elif isinstance(child, str):  # add it to the oinputs
                oinputs.append((child, parent.inputs[idx]))
            else:
                parent.inputs[idx].default_value = child

        if children:
            layers.insert(0, children)

        return parent, layers, oinputs


_Input = int | float | str | Operation


class ShaderMathOperation(Operation):
    def create_node(self, nt: bpy.types.NodeTree) -> bpy.types.Node:
        node = nt.nodes.new("ShaderNodeMath")
        node.operation = self.name
        return node

    @classmethod
    def _check_bad_type(cls, node: ast.Constant) -> Result[tuple, str]:
        if not isinstance(node.value, (int, float)):
            return Err(
                f"Constants cannot be anything other than ints or floats.\n{node.value} is disallowed"
            )
        return Ok(())

    @classmethod
    def validate_node(
        cls,
        node: ast.Expr,
        bad_nodes: tuple[type[ast.expr], ...],
        functions: dict[str, tuple[tuple[int, ...], str]],
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
            allowed_nums = functions.get(name.id)
            if allowed_nums is None:
                errmsg = f"Unrecognized function name: '{name.id}'"

                if matches := difflib.get_close_matches(name.id, list(functions)):
                    return Err(
                        f"{errmsg}\nDid you mean one of these?\n{', '.join(matches)}"
                    )
                return Err(errmsg)

            # check if the number of arguments align with the number of arguments in the GOOD_CALLS
            elif all(len(node.args) != x for x in allowed_nums[0]):
                return Err(
                    f"Function {name.id} is allowed, but\nthe number of arguments is wrong\n({len(node.args)} is not in {allowed_nums[0]})"
                )

        return Ok(())

    @classmethod
    def validate(
        cls,
        e: ast.Module,
        bad_nodes: tuple[type[ast.expr], ...] = BAD_MATH_AST_NODES,
        functions: dict[str, tuple[tuple[int, ...], str]] = SHADER_MATH_CALLS,
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
    def parse(cls, e: ast.expr) -> int | float | str | Self:
        def parse(e):
            """Parses the expression, carrying over settings."""
            return cls.parse(e)

        if isinstance(e, ast.BinOp):
            op = e.op

            if type(op) in SHADER_NODE_BASIC_OPS:
                return cls(
                    name=SHADER_NODE_BASIC_OPS[type(op)],
                    inputs=(
                        parse(e.left),
                        parse(e.right),
                    ),
                )

            if isinstance(op, ast.Add):
                # check for Multiply Add
                if isinstance(e.left, ast.BinOp) and isinstance(e.left.op, ast.Mult):
                    return cls(
                        name="MULTIPLY_ADD",
                        inputs=(
                            parse(e.left.left),
                            parse(e.left.right),
                            parse(e.right),
                        ),
                    )
                if isinstance(e.right, ast.BinOp) and isinstance(e.right.op, ast.Mult):
                    return cls(
                        name="MULTIPLY_ADD",
                        inputs=(
                            parse(e.right.left),
                            parse(e.right.right),
                            parse(e.left),
                        ),
                    )

                return cls(name="ADD", inputs=(parse(e.left), parse(e.right)))

            if isinstance(op, ast.Mult):
                return cls(name="MULTIPLY", inputs=(parse(e.left), parse(e.right)))

            if isinstance(op, ast.Div):
                # check for Inverse Square Root
                if (
                    isinstance(e.left, ast.Constant)
                    and e.left.value == 1
                    and isinstance(e.right, ast.Call)
                    and e.right.func.id == "sqrt"
                ):
                    return cls(
                        name="INVERSE_SQRT",
                        inputs=(parse(e.right.args[0]),),
                    )
                return cls(
                    name="DIVIDE",
                    inputs=(parse(e.left), parse(e.right)),
                )

            # check for Exponent(
            if isinstance(op, ast.Pow):
                if (
                    isinstance(op, ast.Pow)
                    and isinstance(e.left, ast.Name)
                    and e.left.id == "e"
                ):
                    return cls(
                        name="EXPONENT",
                        inputs=(parse(e.right),),
                    )
                return cls(
                    name="POWER",
                    inputs=(parse(e.left), parse(e.right)),
                )

            if isinstance(op, ast.FloorDiv):
                return cls(
                    name="FLOOR",
                    inputs=(
                        cls(
                            name="DIVIDE",
                            inputs=(parse(e.left), parse(e.right)),
                        ),
                    ),
                )

            raise NotImplementedError(f"Unhandled operation {op}")

        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            if isinstance(e.operand, ast.Constant):
                return -e.operand.value
            else:
                return cls(
                    name="MULTIPLY",
                    inputs=(parse(e.operand), -1),
                )

        if isinstance(e, ast.Compare):
            if isinstance(e.ops[0], (ast.Gt, ast.GtE)):
                inputs = (
                    parse(e.left),
                    parse(e.comparators[0]),
                )
                name = "LESS_THAN"

                return cls(
                    name=name,
                    inputs=inputs,
                )

            elif isinstance(e.ops[0], (ast.Lt, ast.LtE)):
                inputs = (
                    parse(e.left),
                    parse(e.comparators[0]),
                )
                name = "GREATER_THAN"

                return cls(
                    name=name,
                    inputs=inputs,
                )

            elif isinstance(e.ops[0], ast.Eq):  # Opinion
                return cls(
                    name="COMPARE",
                    inputs=(
                        parse(e.left),
                        parse(e.comparators[0]),
                        0.5,
                    ),
                )

        if isinstance(e, ast.Call):
            inputs = []
            for arg in e.args:
                inputs.append(parse(arg))

            return cls(
                name=SHADER_MATH_CALLS[e.func.id][1],
                inputs=tuple(inputs),
            )

        if isinstance(e, ast.Constant):
            return e.value

        if isinstance(e, ast.Name):
            return e.id

        if isinstance(e, ast.Expr):
            return parse(e.value)

        raise NotImplementedError(f"Unhandled expression {ast.dump(e, indent=4)}")


class CompositorMathOperation(ShaderMathOperation):
    def create_node(self, nt: bpy.types.NodeTree) -> bpy.types.Node:
        node: bpy.types.CompositorNodeMath = nt.nodes.new("CompositorNodeMath")
        node.operation = self.name
        return node


class TextureMathOperation(ShaderMathOperation):
    def create_node(self, nt: bpy.types.NodeTree) -> bpy.types.Node:
        node: bpy.types.TextureNodeMath = nt.nodes.new("TextureNodeMath")
        node.operation = self.name
        return node


@dataclass
class Tree:
    variables: list[VARIABLE_NAME]
    root: Operation
