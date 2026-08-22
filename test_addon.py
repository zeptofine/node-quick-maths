import ast
import os
import sys
import unittest
from unittest.mock import MagicMock

# Stub Blender environment when running outside Blender
if "bpy" not in sys.modules or not hasattr(sys.modules.get("bpy"), "types"):
    bpy_mock = MagicMock()
    sys.modules["bpy"] = bpy_mock
    sys.modules["bpy.types"] = bpy_mock.types
    sys.modules["bpy.props"] = bpy_mock.props
    sys.modules["bpy.ops"] = bpy_mock.ops

# Ensure parent package can be loaded via relative imports
import importlib.util

pkg_dir = os.path.abspath(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location(
    "node_quick_maths",
    os.path.join(pkg_dir, "__init__.py"),
    submodule_search_locations=[pkg_dir],
)
addon = importlib.util.module_from_spec(spec)
sys.modules["node_quick_maths"] = addon
spec.loader.exec_module(addon)

from node_quick_maths.rustlike_result import Err, Ok
from node_quick_maths.constants import ASSUMABLE_CONSTANTS, Function, SHADER_NODE_BASIC_OPS, VALID_MATH_FUNCTIONS
from node_quick_maths.validation import validate
from node_quick_maths.operations import Operation, Tree
from node_quick_maths.node_creation import (
    CompNodeCreator,
    GeoNodeCreator,
    NodeCreator,
    ShaderNodeCreator,
    TextureNodeCreator,
)
from node_quick_maths.node_composers import ComposeNodes


class TestRustLikeResult(unittest.TestCase):
    def test_ok(self):
        res = Ok(42)
        self.assertFalse(res.is_err())
        self.assertEqual(res.unwrap(), 42)
        with self.assertRaises(Exception):
            res.unwrap_err()

    def test_err(self):
        res = Err("something went wrong")
        self.assertTrue(res.is_err())
        self.assertEqual(res.unwrap_err(), "something went wrong")
        with self.assertRaises(Exception):
            res.unwrap()


class TestValidation(unittest.TestCase):
    def _validate_str(self, s: str):
        mod = ast.parse(s, mode="exec")
        return validate(mod)

    def test_valid_basic_expressions(self):
        valid = ["x + y", "sin(x) * 2.5", "1 / sqrt(x)", "e ** x", "a * b + c", "x < 5", "atan2(y, x)"]
        for expr in valid:
            self.assertFalse(self._validate_str(expr).is_err(), f"Expected '{expr}' to be valid")

    def test_empty_expression(self):
        mod = ast.parse("", mode="exec")
        res = validate(mod)
        self.assertTrue(res.is_err())
        self.assertIn("empty", res.unwrap_err().lower())

    def test_disallowed_ast_nodes(self):
        disallowed = ["[x for x in y]", "{'a': 1}", "x if y else z", "(lambda x: x)(y)", "x and y"]
        for expr in disallowed:
            self.assertTrue(self._validate_str(expr).is_err(), f"Expected '{expr}' to be rejected")

    def test_disallowed_constant_types(self):
        for val_str in ["'hello'", "True", "False", "None", "b'bytes'"]:
            res = self._validate_str(val_str)
            self.assertTrue(res.is_err(), f"Expected literal {val_str} to be rejected")
            self.assertIn("Constants cannot be anything other than ints or floats", res.unwrap_err())

    def test_unrecognized_function(self):
        res = self._validate_str("foobar(x)")
        self.assertTrue(res.is_err())
        self.assertIn("Unrecognized function name", res.unwrap_err())

    def test_unrecognized_function_with_typo_suggestion(self):
        res = self._validate_str("siin(x)")
        self.assertTrue(res.is_err())
        self.assertIn("Did you mean", res.unwrap_err())

    def test_wrong_argument_count(self):
        res = self._validate_str("sin(x, y)")
        self.assertTrue(res.is_err())
        self.assertIn("number of arguments is wrong", res.unwrap_err())


class TestOperations(unittest.TestCase):
    def _parse(self, expr_str: str):
        expr = ast.parse(expr_str, mode="eval").body
        return Operation.parse(expr)

    def test_basic_arithmetic(self):
        op_add = self._parse("x + y")
        self.assertEqual(op_add, Operation(name="ADD", inputs=("x", "y")))

        op_sub = self._parse("x - y")
        self.assertEqual(op_sub, Operation(name="SUBTRACT", inputs=("x", "y")))

        op_mul = self._parse("x * y")
        self.assertEqual(op_mul, Operation(name="MULTIPLY", inputs=("x", "y")))

        op_div = self._parse("x / y")
        self.assertEqual(op_div, Operation(name="DIVIDE", inputs=("x", "y")))

        op_pow = self._parse("x ** y")
        self.assertEqual(op_pow, Operation(name="POWER", inputs=("x", "y")))

        op_mod = self._parse("x % y")
        self.assertEqual(op_mod, Operation(name="MODULO", inputs=("x", "y")))

    def test_floor_div(self):
        op = self._parse("x // y")
        self.assertEqual(op, Operation(name="FLOOR", inputs=(Operation(name="DIVIDE", inputs=("x", "y")),)))

    def test_multiply_add_patterns(self):
        op1 = self._parse("a * b + c")
        self.assertEqual(op1, Operation(name="MULTIPLY_ADD", inputs=("a", "b", "c")))

        op2 = self._parse("c + a * b")
        self.assertEqual(op2, Operation(name="MULTIPLY_ADD", inputs=("a", "b", "c")))

    def test_inverse_sqrt(self):
        op = self._parse("1 / sqrt(x)")
        self.assertEqual(op, Operation(name="INVERSE_SQRT", inputs=("x",)))

    def test_exponent(self):
        op = self._parse("e ** x")
        self.assertEqual(op, Operation(name="EXPONENT", inputs=("x",)))

    def test_unary_sub(self):
        self.assertEqual(self._parse("-5"), -5)
        self.assertEqual(self._parse("-3.14"), -3.14)
        self.assertEqual(self._parse("-x"), Operation(name="MULTIPLY", inputs=("x", -1)))

    def test_comparisons(self):
        self.assertEqual(self._parse("x < y"), Operation(name="LESS_THAN", inputs=("x", "y")))
        self.assertEqual(self._parse("x > y"), Operation(name="GREATER_THAN", inputs=("x", "y")))
        self.assertEqual(self._parse("x == y"), Operation(name="COMPARE", inputs=("x", "y", 0.5)))

    def test_function_calls(self):
        self.assertEqual(self._parse("sin(x)"), Operation(name="SINE", inputs=("x",)))
        self.assertEqual(self._parse("atan2(y, x)"), Operation(name="ARCTAN2", inputs=("y", "x")))

    def test_variable_extraction_and_sorting(self):
        op = self._parse("z + y * x")
        self.assertEqual(op.variables("ALPHABET"), ["x", "y", "z"])
        self.assertCountEqual(op.variables("INSERTION"), ["z", "y", "x"])


class TestComplexMathExpressions(unittest.TestCase):
    def _validate_and_parse(self, expr_str: str):
        mod = ast.parse(expr_str, mode="exec")
        val_res = validate(mod)
        self.assertFalse(val_res.is_err(), f"Validation failed for: {expr_str}: {val_res}")
        expr = ast.parse(expr_str, mode="eval").body
        return Operation.parse(expr)

    def test_quadratic_formula(self):
        # (-b + sqrt(b**2 - 4*a*c)) / (2*a)
        expr = "(-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)"
        op = self._validate_and_parse(expr)
        self.assertIsInstance(op, Operation)
        self.assertEqual(op.name, "DIVIDE")
        self.assertEqual(op.variables("ALPHABET"), ["a", "b", "c"])

    def test_3d_euclidean_distance(self):
        expr = "sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)"
        op = self._validate_and_parse(expr)
        self.assertEqual(op.name, "SQRT")
        self.assertEqual(sorted(op.variables("ALPHABET")), ["x1", "x2", "y1", "y2", "z1", "z2"])

    def test_fourier_series_synthesis(self):
        expr = "sin(2 * pi * f * t) + 0.5 * sin(4 * pi * f * t) + 0.25 * sin(6 * pi * f * t)"
        op = self._validate_and_parse(expr)
        self.assertEqual(op.name, "MULTIPLY_ADD")
        self.assertEqual(sorted(op.variables("ALPHABET")), ["f", "pi", "t"])

    def test_nested_signal_wrapping_and_smin(self):
        expr = "wrap(smin(sin(x) * cos(y), exp(z // 2), 0.1), -1.0, 1.0)"
        op = self._validate_and_parse(expr)
        self.assertEqual(op.name, "WRAP")
        self.assertEqual(op.variables("ALPHABET"), ["x", "y", "z"])

    def test_combined_special_patterns(self):
        # Combines INVERSE_SQRT, MULTIPLY_ADD (x*y+z and c+a*b), EXPONENT, and FLOOR DIV
        expr = "(1 / sqrt(x + 1)) * (a * b + c) + (d * e + f) + (e ** k) + (u // v)"
        op = self._validate_and_parse(expr)
        self.assertIsInstance(op, Operation)
        self.assertEqual(op.variables("ALPHABET"), ["a", "b", "c", "d", "e", "f", "k", "u", "v", "x"])

    def test_variadic_log_calls(self):
        op1 = self._validate_and_parse("log(x)")
        self.assertEqual(op1, Operation(name="LOGARITHM", inputs=("x",)))

        op2 = self._validate_and_parse("log(x, 10)")
        self.assertEqual(op2, Operation(name="LOGARITHM", inputs=("x", 10)))

    def test_all_valid_shader_functions(self):
        sample_calls = [
            ("min(a, b)", "MINIMUM"),
            ("max(a, b)", "MAXIMUM"),
            ("sign(x)", "SIGN"),
            ("cmp(a, b, 0.05)", "COMPARE"),
            ("smin(a, b, 0.1)", "SMOOTH_MIN"),
            ("smax(a, b, 0.1)", "SMOOTH_MAX"),
            ("round(x)", "ROUND"),
            ("floor(x)", "FLOOR"),
            ("ceil(x)", "CEIL"),
            ("trunc(x)", "TRUNC"),
            ("int(x)", "TRUNC"),
            ("frac(x)", "FRACT"),
            ("fmod(a, b)", "FLOORED_MODULO"),
            ("snap(a, 0.5)", "SNAP"),
            ("pingpong(a, 2.0)", "PINGPONG"),
            ("sin(x)", "SINE"),
            ("cos(x)", "COSINE"),
            ("tan(x)", "TANGENT"),
            ("asin(x)", "ARCSINE"),
            ("acos(x)", "ARCCOSINE"),
            ("atan(x)", "ARCTANGENT"),
            ("atan2(y, x)", "ARCTAN2"),
            ("sinh(x)", "SINH"),
            ("cosh(x)", "COSH"),
            ("tanh(x)", "TANH"),
            ("rad(deg_val)", "RADIANS"),
            ("deg(rad_val)", "DEGREES"),
        ]
        for expr, expected_enum in sample_calls:
            op = self._validate_and_parse(expr)
            self.assertEqual(op.name, expected_enum, f"Expected {expr} to resolve to {expected_enum}")

    def test_deep_unary_and_parens(self):
        expr = "-(-sin(-x) + -(y * -2.5))"
        op = self._validate_and_parse(expr)
        self.assertIsInstance(op, Operation)


class TestComplexInvalidExpressions(unittest.TestCase):
    def _assert_invalid(self, expr_str: str):
        try:
            mod = ast.parse(expr_str, mode="exec")
            res = validate(mod)
            self.assertTrue(res.is_err(), f"Expected expression '{expr_str}' to fail validation")
        except SyntaxError:
            pass  # Syntax errors are also rejected by the addon

    def test_bitwise_and_matrix_operations(self):
        invalid = [
            "x | y",
            "x & y",
            "x ^ y",
            "x << 2",
            "x >> 2",
            "~x",
            "x @ y",
        ]
        for expr in invalid:
            # Ast validation checks against BAD_MATH_AST_NODES or operations.parse unhandled
            mod = ast.parse(expr, mode="exec")
            val_res = validate(mod)
            if not val_res.is_err():
                # If ast.walk allows it, Operation.parse should raise NotImplementedError
                with self.assertRaises(NotImplementedError):
                    Operation.parse(ast.parse(expr, mode="eval").body)

    def test_subscripts_and_slices(self):
        invalid = [
            "x[0]",
            "x[1:5]",
            "sin(x)[0]",
            "x['key']",
        ]
        for expr in invalid:
            self._assert_invalid(expr)

    def test_attribute_and_method_calls(self):
        invalid = [
            "math.sin(x)",
            "x.sin()",
            "obj.val + 1",
        ]
        for expr in invalid:
            self._assert_invalid(expr)

    def test_indirect_and_non_name_calls(self):
        invalid = [
            "(sin or cos)(x)",
            "funcs[0](x)",
            "(lambda x: x)(5)",
        ]
        for expr in invalid:
            self._assert_invalid(expr)

    def test_data_structures_and_comprehensions(self):
        invalid = [
            "[x, y, z]",
            "{'a': 1, 'b': 2}",
            "{x, y, z}",
            "sin([x for x in y])",
            "sin((x, y))",
        ]
        for expr in invalid:
            self._assert_invalid(expr)

    def test_statements_and_assignments(self):
        invalid = [
            "x = 5",
            "x += 1",
            "for i in range(10): pass",
            "def foo(): pass",
            "import math",
            "x := 5",
        ]
        for expr in invalid:
            self._assert_invalid(expr)

    def test_invalid_function_arities(self):
        wrong_arities = [
            "sqrt(x, y)",
            "sqrt()",
            "sin()",
            "sin(x, y)",
            "atan2(x)",
            "atan2(x, y, z)",
            "abs(x, y)",
            "smin(x)",
            "smin(a, b)",
            "smin(a, b, c, d)",
            "wrap(a, b)",
            "wrap(a, b, c, d)",
            "log(a, b, c)",
            "log()",
        ]
        for expr in wrong_arities:
            self._assert_invalid(expr)


class TestTreeAndNodeCreation(unittest.TestCase):
    def test_tree_parse_valid(self):
        mod = ast.parse("x + y", mode="exec")
        res = Tree.parse("x + y", mod.body[0], ShaderNodeCreator, "ALPHABET")
        self.assertFalse(res.is_err())
        tree = res.unwrap()
        self.assertEqual(tree.variables, ["x", "y"])
        self.assertEqual(tree.group_type, "ShaderNodeGroup")

    def test_tree_layers_and_connections(self):
        mod = ast.parse("x * y + z", mode="exec")
        tree = Tree.parse("x * y + z", mod.body[0], ShaderNodeCreator, "INSERTION").unwrap()

        mock_nt = MagicMock()
        mock_node = MagicMock()
        mock_node.inputs = [MagicMock(), MagicMock(), MagicMock()]
        mock_node.outputs = [MagicMock()]
        mock_nt.nodes.new.return_value = mock_node

        layers, inputs = tree.layers_and_connections(mock_nt)
        self.assertEqual(len(layers), 1)
        self.assertEqual(len(inputs), 3)

    def test_node_creators(self):
        self.assertEqual(ShaderNodeCreator.tree_type, "ShaderNodeTree")
        self.assertEqual(GeoNodeCreator.tree_type, "GeometryNodeTree")
        self.assertEqual(CompNodeCreator.tree_type, "CompositorNodeTree")
        self.assertEqual(TextureNodeCreator.tree_type, "TextureNodeTree")


class TestNodeComposers(unittest.TestCase):
    def test_preview_recursion(self):
        op = Operation(name="ADD", inputs=(Operation(name="MULTIPLY", inputs=("x", "y")), "z"))
        mock_layout = MagicMock()
        composer = ComposeNodes()
        composer.preview(op, mock_layout)
        self.assertTrue(mock_layout.column.called)


if __name__ == "__main__":
    unittest.main()
