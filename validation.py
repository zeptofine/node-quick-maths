"""
Helper functions to validate string input that a tree can be generated from them
"""

import ast
import difflib

from .constants import (
    BAD_MATH_AST_NODES,
    VALID_MATH_FUNCTIONS,
    Function,
)
from .rustlike_result import Err, Ok, Result


def validate(
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
        r = validate_node(node, bad_nodes, functions)
        if r.is_err():
            return r

    return Ok(())


def validate_node(
    node: ast.Expr,
    bad_nodes: tuple[type[ast.expr], ...],
    functions: dict[str, Function],
) -> Result[tuple, str]:
    # check if node is bad
    if any(isinstance(node, bad_node) for bad_node in bad_nodes):
        return Err(f"Do not use node of type: {type(node)} ")

    # check if node is a constant and it is a disallowed type
    if isinstance(node, ast.Constant):
        r = _check_bad_type(node)
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


def _check_bad_type(node: ast.Constant) -> Result[tuple, str]:
    if not isinstance(node.value, int | float):
        return Err(f"Constants cannot be anything other than ints or floats.\n{node.value} is disallowed")
    return Ok(())
