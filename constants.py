import ast
import math


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

SHADER_MATH_CALLS = {  # Function inputs in python, names Blender
    # -------
    # Functions
    # ADD, SUBTRACT, MULTIPLY, DIVIDE have dedicated operators
    # MULTIPLY_ADD: Special case (a * b + c) or (a + b * c),
    # POWER has a dedicated operator
    "log": ((1, 2), "LOGARITHM"),  # (x[, base])
    "sqrt": ((1,), "SQRT"),  # (x)
    # INVERSE_SQRT: Special case ( 1 / sqrt(x) )
    "abs": ((1,), "ABSOLUTE"),  # (x)
    "exp": ((1,), "EXPONENT"),  # (x) or (e ** x)
    # -------
    # Comparison
    "min": ((2,), "MINIMUM"),  # (x, y)
    "max": ((2,), "MAXIMUM"),  # (x, y)
    # LESS_THAN, GREATER_THAN have dedicated operators
    "sign": ((1,), "SIGN"),  # (x), SIGN
    "cmp": ((3,), "COMPARE"),  # (x, y, z)
    "smin": ((3,), "SMOOTH_MIN"),  # (x, y, z)
    "smax": ((3,), "SMOOTH_MAX"),  # (x, y, z)
    # -------
    # Rounding
    "round": ((1,), "ROUND"),  # (x)
    "floor": ((1,), "FLOOR"),  # (x)
    "ceil": ((1,), "CEIL"),  # (x)
    "trunc": ((1,), "TRUNC"),  # (x)
    "int": ((1,), "TRUNC"),  # alias for truncate
    "frac": ((1,), "FRACT"),  # (x)
    # MODULO has a dedicated operator
    "fmod": ((2,), "FLOORED_MODULO"),  # (x, y)
    "wrap": ((3,), "WRAP"),  # (x, y, z)
    "snap": ((2,), "SNAP"),  # (x, y)
    "pingpong": ((2,), "PINGPONG"),  # (x, y)
    # -------
    # Trigonometric
    "sin": ((1,), "SINE"),  # (x)
    "cos": ((1,), "COSINE"),  # (x)
    "tan": ((1,), "TANGENT"),  # (x)
    "asin": ((1,), "ARCSINE"),  # (x)
    "acos": ((1,), "ARCCOSINE"),  # (x)
    "atan": ((1,), "ARCTANGENT"),  # (x)
    "atan2": ((2,), "ARCTAN2"),  # (x, y)
    "sinh": ((1,), "SINH"),  # (x)
    "cosh": ((1,), "COSH"),  # (x)
    "tanh": ((1,), "TANH"),  # (x)
    # -------
    # Conversion
    "rad": ((1,), "RADIANS"),  # (x)
    "deg": ((1,), "DEGREES"),  # (x)
}

SHADER_NODE_BASIC_OPS: dict[type[ast.operator], str] = {
    # Add is covered
    ast.Sub: "SUBTRACT",
    # Mult is covered
    # Div is covered
    ast.Mod: "MODULO",
    # Pow is covered
    # Floordiv is covered
    # MatMult, LShift, RShift, BitOr, BitXor, BitAnd will not be implemented
}

VARIABLE_NAME = str

PRINTABLE_SHADER_MATH_CALLS = {
    "Functions": (
        "+",
        "-",
        "*",
        "/",
        "**",
        "log(x[, base])",
        "sqrt(x)",
        "1 / sqrt(x)",
        "abs(x)",
        "exp(x) or e ** x",
    ),
    "Comparison": (
        "min(x, y)",
        "max(x, y)",
        "x > y",
        "x < y",
        "sign(x)",
        "cmp(x, y, z)",
        "smin(x, y, z)",
        "smax(x, y, z)",
    ),
    "Rounding": (
        "round(x)",
        "floor(x)",
        "ceil(x)",
        "trunc(x)",
        "frac(x)",
        "x % y",
        "fmod(x, y)",
        "wrap(x, y, z)",
        "snap(x, y)",
        "pingpong(x, y)",
    ),
    "Trigonometric": (
        "sin(x)",
        "cos(x)",
        "tan(x)",
        "asin(x)",
        "acos(x)",
        "atan(x)",
        "atan2(x,y)",
        "sinh(x)",
        "cosh(x)",
        "tanh(x)",
    ),
    "Conversion": ("rad(x)", "deg(x)"),
}


ASSUMABLE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}
