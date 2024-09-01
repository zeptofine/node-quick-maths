import ast
import math
from dataclasses import dataclass
from functools import cache

BAD_MATH_AST_NODES = (
    ast.BoolOp,
    ast.NamedExpr,
    ast.Lambda,
    ast.IfExp,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.FormattedValue,
    ast.JoinedStr,
    ast.Attribute,
    ast.Subscript,
    ast.Starred,
    ast.List,
    ast.Tuple,
    ast.Slice,
)


# same as BAD_MATH_AST_NODES but without tuple
BAD_VECTOR_MATH_AST_NODES = (
    ast.BoolOp,
    ast.NamedExpr,
    ast.Lambda,
    ast.IfExp,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.FormattedValue,
    ast.JoinedStr,
    ast.Attribute,
    ast.Subscript,
    ast.Starred,
    ast.List,
    ast.Slice,
)

TYPICAL_VAR_NAMES = ("x", "y", "z", "a", "b", "c")


@dataclass(frozen=True)
class Function:
    input_nums: tuple[int, ...]
    enum_value: str
    printable_args: str | None = None
    input_types: tuple[type, ...] | None = None

    @cache
    def __str__(self):
        if self.printable_args is None:
            # generate args from the number of inputs
            s = ", ".join(
                var
                for var, _ in zip(
                    TYPICAL_VAR_NAMES,
                    range(self.input_nums[0]),
                )
            )
            return f"({s})"

        if "(" in self.printable_args:
            return self.printable_args
        else:
            return f"({self.printable_args})"


PrintRepresent = str


def F(
    nums: int | tuple[int, ...],
    enum_value: str,
    printable_args: str | None = None,
    input_types: tuple[type, ...] | None = None,
):
    if isinstance(nums, int):
        nums = (nums,)
    return Function(nums, enum_value, printable_args, input_types)


SHADER_MATH_CALLS: dict[str, dict[str, Function | PrintRepresent]] = {
    "Functions": {
        # ADD, SUBTRACT, MULTIPLY, DIVIDE have dedicated operators
        "+": "x + y",
        "-": "x - y",
        "*": "x * y",
        "/": "x / y",
        # MULTIPLY_ADD: Special case (a * b + c) or (a + b * c),
        # POWER has a dedicated operator
        "**": "x ** y",
        "log": F((1, 2), "LOGARITHM", "x[, base]"),
        "sqrt": F(1, "SQRT"),
        # INVERSE_SQRT: Special case ( 1 / sqrt(x) )
        "abs": F(1, "ABSOLUTE"),
        "exp": F(1, "EXPONENT", "(x) (or e ** x)"),
    },
    "Comparison": {
        "min": F(2, "MINIMUM"),
        "max": F(2, "MAXIMUM"),
        # LESS_THAN, GREATER_THAN have dedicated operators
        "<": "x < y",
        ">": "x > y",
        "sign": F(1, "SIGN"),
        "cmp": F(3, "COMPARE"),
        "smin": F(3, "SMOOTH_MIN"),
        "smax": F(3, "SMOOTH_MAX"),
    },
    "Rounding": {
        "round": F(1, "ROUND"),
        "floor": F(1, "FLOOR"),
        "ceil": F(1, "CEIL"),
        "trunc": F(1, "TRUNC"),
        "int": F(1, "TRUNC"),  # alias for truncate
        "frac": F(1, "FRACT"),
        # MODULO has a dedicated operator
        "%": "x % y",
        "fmod": F(2, "FLOORED_MODULO"),
        "wrap": F(3, "WRAP"),
        "snap": F(2, "SNAP"),
        "pingpong": F(2, "PINGPONG"),
    },
    "Trigonometric": {
        "sin": F(1, "SINE"),
        "cos": F(1, "COSINE"),
        "tan": F(1, "TANGENT"),
        "asin": F(1, "ARCSINE"),
        "acos": F(1, "ARCCOSINE"),
        "atan": F(1, "ARCTANGENT"),
        "atan2": F(2, "ARCTAN2"),
        "sinh": F(1, "SINH"),
        "cosh": F(1, "COSH"),
        "tanh": F(1, "TANH"),
    },
    "Conversion": {
        "rad": F(1, "RADIANS"),
        "deg": F(1, "DEGREES"),
    },
}

VALID_MATH_FUNCTIONS: dict[str, Function] = {
    k: v for _, group in SHADER_MATH_CALLS.items() for k, v in group.items() if isinstance(v, Function)
}


VECTOR_MATH_CALLS: dict[str, dict[str, Function | PrintRepresent]] = {
    "Functions": {
        # ADD, SUBTRACT, MULTIPLY, DIVIDE have dedicated operators
        "+": "x + y",
        "-": "x - y",
        "*": "x * y",
        "/": "x / y",
        # MULTIPLY_ADD: Special case (a * b + c) or (a + b * c),
    },
    "1": {
        "cross": F((2,), "CROSS_PRODUCT"),
        "project": F((2,), "PROJECT"),
        "reflect": F((2,), "REFLECT"),
        "refract": F((2, 3), "REFRACT", "x, y[, IOR]"),
        "ffw": F((2, 3), "FACE_FORWARD", "x, y[, z]"),
        "dot": F((2,), "DOT_PRODUCT"),
    },
    "Calculations": {
        "dist": F((2,), "DISTANCE"),
        "len": F((1,), "LENGTH"),
        "scale": F((2,), "SCALE", "x, scale", (tuple, float)),
        "normalize": F((1,), "NORMALIZE"),
    },
    "Wrappings": {
        "abs": F((1,), "ABSOLUTE"),
        "min": F((2,), "MINIMUM"),
        "max": F((2,), "MAXIMUM"),
        "floor": F((1,), "FLOOR"),
        "ceil": F((1,), "CEIL"),
        "frac": F((1,), "FRACTION"),
        "%": "x % y",
        "wrap": F((3,), "WRAP"),
        "snap": F((2,), "SNAP"),
    },
    "Trigonometric": {
        "sin": F((1,), "SINE"),
        "cos": F((1,), "COSINE"),
        "tan": F((1,), "TANGENT"),
    },
}


SHADER_NODE_BASIC_OPS: dict[type[ast.operator], str] = {
    ast.Add: "ADD",
    ast.Sub: "SUBTRACT",
    ast.Mult: "MULTIPLY",
    ast.Div: "DIVIDE",
    ast.Mod: "MODULO",
    ast.Pow: "POWER",
    # Floordiv is covered
    # MatMult, LShift, RShift, BitOr, BitXor, BitAnd will not be implemented
}

SHADER_VECTOR_NODE_BASIC_OPS: dict[type[ast.operator], str] = {
    # Add is covered
    ast.Sub: "SUBTRACT",
    # Mult is covered
    # Div is covered
    ast.Mod: "MODULO",
    # Floordiv is covered
    # Pow, MatMult, LShift, RShift, BitOr, BitXor, BitAnd will not be implemented
}


VARIABLE_NAME = str


ASSUMABLE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}
