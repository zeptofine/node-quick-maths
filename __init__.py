from abc import abstractmethod
import ast
from dataclasses import dataclass
from typing import Self, Set

import bpy
from bpy.types import Context, Event


def convert_expression(s: str) -> ast.Module:
    return ast.parse(s, mode="exec")


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

# minus ast.Tuple
# plus ast.Compare
BAD_VECCTOR_AST_NODES = (
    ast.BoolOp,
    ast.Compare,
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


SHADER_MATH_CALLS = {  # Function inputs in python, names Blender
    # -------
    # Functions
    # ADD, SUBTRACT, MULTIPLY, DIVIDE, POWER have dedicated operators
    "log": ((1, 2), "LOGARITHM"),  # (x[, base])
    "sqrt": ((1,), "SQRT"),  # (x)
    "abs": ((1,), "ABSOLUTE"),  # (x)
    "exp": ((1,), "EXPONENT"),  # (x)
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

PRINTABLE_SHADER_MATH_CALLS = {
    "Functions": (
        "+",
        "-",
        "*",
        "/",
        "**",
        "log(x[, base])",
        "sqrt(x)",
        "abs(x)",
        "exp(x)",
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


SHADER_NODE_BASIC_OPS = {
    ast.Add: "ADD",
    ast.Sub: "SUBTRACT",
    ast.Mult: "MULTIPLY",
    ast.Div: "DIVIDE",
    ast.Mod: "MODULO",
    ast.Pow: "POWER",
}


VARIABLE_NAME = str


@dataclass
class Operation:
    name: str
    inputs: "list[int | float | str | Operation]"

    @classmethod
    @abstractmethod
    def validate(
        cls,
        e: ast.Module,
        bad_nodes: tuple[ast.Expr, ...],
        functions: dict[str, tuple[tuple[int, ...], str]],
    ) -> tuple[bool, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def parse(cls, e: ast.expr) -> int | float | str | Self:
        raise NotImplementedError

    def to_tree(self):
        return Tree(variables=self.variables(), toplevel_operation=self)

    def variables(self) -> set[VARIABLE_NAME]:
        vars: set[VARIABLE_NAME] = set()
        for input in self.inputs:
            if isinstance(input, Operation):
                vars.update(input.variables())
            elif isinstance(input, VARIABLE_NAME):
                vars.add(input)
        return vars

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
        #     [                    child1,                                        child2                     ],
        #     [        child1.1,               child1.2,              child2.1,               child2.2       ],
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


class ShaderMathOperation(Operation):
    def create_node(self, nt: bpy.types.NodeTree) -> bpy.types.Node:
        node = nt.nodes.new("ShaderNodeMath")
        node.operation = self.name
        return node

    @classmethod
    def _check_bad_type(cls, node: ast.Constant) -> tuple[bool, str]:
        if not isinstance(node.value, (int, float)):
            return (
                False,
                f"Constants cannot be anything other than ints or floats.\n{node.value} is disallowed",
            )
        return True, ""

    @classmethod
    def validate_node(
        cls,
        node: ast.Expr,
        bad_nodes: tuple[type[ast.expr], ...],
        functions: dict[str, tuple[tuple[int, ...], str]],
    ) -> tuple[bool, str]:
        # check if node is bad
        if any(isinstance(node, bad_node) for bad_node in bad_nodes):
            return (False, f"Do not use node of type: {type(node)} ")

        # check if node is a constant and it is a disallowed type
        if isinstance(node, ast.Constant):
            s, err = cls._check_bad_type(node)
            if not s:
                return s, err

        # check if node is a call and it has an allowed function name
        if isinstance(node, ast.Call):
            name: ast.Name = node.func
            allowed_nums = functions.get(name.id)
            if allowed_nums is None:
                return (
                    False,
                    f"Unrecognized function name: '{name.id}'",
                )

            # check if the number of arguments align with the number of arguments in the GOOD_CALLS
            if all(len(node.args) != x for x in allowed_nums[0]):
                return (
                    False,
                    f"Function {name.id} is allowed, but\nthe number of arguments is wrong\n({len(node.args)} is not in {allowed_nums[0]})",
                )

        return True, ""

    @classmethod
    def validate(
        cls,
        e: ast.Module,
        bad_nodes: tuple[type[ast.expr], ...] = BAD_MATH_AST_NODES,
        functions: dict[str, tuple[tuple[int, ...], str]] = SHADER_MATH_CALLS,
    ) -> tuple[bool, str]:
        # check that the node in the ast body is just an Expr
        expr: ast.Expr

        if not isinstance((expr := e.body[0]), ast.Expr):
            return (False, "Invalid expression type. Only create math expressions!")

        for node in ast.walk(expr):
            s, err = cls.validate_node(node, bad_nodes, functions)
            if not s:
                return s, err

        return (True, "")

    @classmethod
    def parse(cls, e: ast.expr) -> int | float | str | Self:
        if isinstance(e, ast.BinOp):
            op = e.op

            if type(op) in SHADER_NODE_BASIC_OPS:
                return cls(
                    name=SHADER_NODE_BASIC_OPS[type(op)],
                    inputs=[cls.parse(e.left), cls.parse(e.right)],
                )
            elif isinstance(op, ast.FloorDiv):
                return cls(
                    name="FLOOR",
                    inputs=[
                        cls(
                            name="DIVIDE",
                            inputs=[cls.parse(e.left), cls.parse(e.right)],
                        ),
                        0,
                    ],
                )
            else:
                raise NotImplementedError(f"Unhandled operation {op}")

        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            if isinstance(e.operand, ast.Constant):
                return -e.operand.value
            else:
                return cls(
                    name="MULTIPLY",
                    inputs=[cls.parse(e.operand), -1],
                )

        if isinstance(e, ast.Compare):
            if isinstance(e.ops[0], (ast.Gt, ast.GtE)):
                return cls(
                    name="GREATER_THAN",
                    inputs=[cls.parse(e.left), cls.parse(e.comparators[0])],
                )
            elif isinstance(e.ops[0], (ast.Lt, ast.LtE)):
                return cls(
                    name="LESS_THAN",
                    inputs=[cls.parse(e.left), cls.parse(e.comparators[0])],
                )
            elif isinstance(e.ops[0], ast.Eq):  # Opinion
                return cls(
                    name="COMPARE",
                    inputs=[
                        cls.parse(e.left),
                        cls.parse(e.comparators[0]),
                        0.5,
                    ],
                )

        if isinstance(e, ast.Call):
            inputs = []
            for arg in e.args:
                inputs.append(cls.parse(arg))

            return cls(name=SHADER_MATH_CALLS[e.func.id][1], inputs=inputs)

        if isinstance(e, ast.Constant):
            return e.value

        if isinstance(e, ast.Name):
            return e.id

        if isinstance(e, ast.Expr):
            return cls.parse(e.value)

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
    toplevel_operation: Operation

    def parse(e: ast.Expr, t: type[Operation]) -> Self:
        op = t.parse(e)
        assert isinstance(
            op, Operation
        ), f"{type(e)} is not a math operation:\n{ast.dump(e, indent=4)}"
        return Tree(variables=op.variables(), toplevel_operation=op)


def _new_reroute(nt: bpy.types.NodeTree, name: str):
    node = nt.nodes.new("NodeReroute")
    if name:
        node.label = name
    return node


@dataclass
class ComposeNodes:
    use_reroutes: bool = True
    center_nodes: bool = True
    hide_nodes: bool = False

    def run(self, tree: Tree, context: bpy.types.Context):
        bpy.ops.node.select_all(action="DESELECT")
        # Hmm...
        space_data: bpy.types.SpaceNodeEditor = context.space_data
        node_tree: bpy.types.NodeTree = space_data.edit_tree
        offset = (space_data.cursor_location[0], space_data.cursor_location[1])

        layers = self.execute(tree, nt=node_tree)

        # node.height is too small to represent the actual visible height of the nodes
        # for some reason, so these are a little bigger than the width margin
        if self.hide_nodes:
            height_margin = -20
        else:
            height_margin = 60
        width_margin = 20

        layer_heights: list[float] = []
        for layer in layers:
            if self.hide_nodes:
                for node in layer:
                    node.hide = True

            # calculate the total height of the layer
            height = 0
            for node in layer:
                height += node.height + height_margin
            layer_heights.append(height)

        max_height = max(layer_heights)

        # loop through every layer in the subtrees, move them to the correct location
        for l_idx, (layer, layer_heights) in enumerate(zip(layers, layer_heights)):
            # offset the layer by user-defined rules
            if self.center_nodes:
                layer_offset = (
                    offset[0],
                    offset[1] - ((max_height / 2) - (layer_heights / 2)),
                )
            else:
                layer_offset = offset

            # move the nodes
            for n_idx, node in enumerate(layer):
                position = (
                    (layer_offset[0] - (node.width + width_margin) * l_idx),
                    (layer_offset[1] - (node.height + height_margin) * n_idx),
                )
                node.location = position

        bpy.ops.node.translate_attach_remove_on_cancel("INVOKE_DEFAULT")

        return {"FINISHED"}

    @staticmethod
    @abstractmethod
    def _new_value(nt: bpy.types.NodeTree, name: str = ""):
        _new_reroute(nt, name)

    def execute(self, tree: Tree, nt: bpy.types.NodeTree) -> list[list[bpy.types.Node]]:
        sublayers: list[list[bpy.types.Node]]
        node, sublayers, inputs = tree.toplevel_operation.generate(
            nt=nt,
        )

        sublayers.insert(0, [node])

        # Create the new variables as Value nodes
        vars = {}
        for idx, variable in enumerate(tree.variables):  # create a Value
            if self.use_reroutes:
                vars[variable] = _new_reroute(nt, name=variable)
            else:
                vars[variable] = self._new_value(nt, name=variable)

        # Connect the Value nodes to the corresponding sockets
        for variable, input in inputs:
            nt.links.new(vars[variable].outputs[0], input)

        sublayers.append(list(vars.values()))

        return sublayers


class ComposeShaderMathNodes(ComposeNodes):
    @staticmethod
    def _new_value(nt: bpy.types.NodeTree, name: str = ""):
        node = nt.nodes.new("ShaderNodeValue")
        if name:
            node.label = name
        return node


class ComposeCompositorMathNodes(ComposeNodes):
    @staticmethod
    def _new_value(nt: bpy.types.NodeTree, name: str = ""):
        node = nt.nodes.new("CompositorNodeValue")
        if name:
            node.label = name
        return node


class ComposeTextureMathNodes(ComposeNodes): ...


class ComposeNodeTree(bpy.types.Operator):
    bl_idname = "node.compose_qm_nodes"
    bl_label = "Node Quick Maths: Compose nodes"

    show_available_functions: bpy.props.BoolProperty(
        name="Show available functions",
        default=False,
    )
    use_reroutes: bpy.props.BoolProperty(
        name="Use reroutes instead of Values",
        description="Required in Texture Node Editor.",
    )
    hide_nodes: bpy.props.BoolProperty(
        name="Hide nodes",
        description="Hides nodes to preserve grid space.",
    )
    center_nodes: bpy.props.BoolProperty(
        name="Center nodes",
        description="Centers the generated nodes. This is a personal preference.",
    )

    editor_type: bpy.props.StringProperty(name="Editor Type")
    expression: bpy.props.StringProperty(
        description="The source expression for the nodes."
    )

    def invoke(self, context: Context, event: Event) -> Set[str]:
        wm = context.window_manager
        ui_mode = context.area.ui_type
        self.editor_type = ui_mode
        if context.preferences.addons[__name__].preferences.debug_prints:
            print(f"NQM: Editor type: {self.editor_type}")

        self.use_reroutes = context.preferences.addons[
            __name__
        ].preferences.use_reroutes

        if self.editor_type == "TextureNodeTree":
            self.use_reroutes = True

        return wm.invoke_props_dialog(self, confirm_text="Create", width=600)

    def current_operation_type(
        self,
    ) -> tuple[type[Operation], type[ComposeNodes]] | None:
        if self.editor_type in {"ShaderNodeTree", "GeometryNodeTree"}:
            return (ShaderMathOperation, ComposeShaderMathNodes)
        elif self.editor_type == "CompositorNodeTree":
            return (CompositorMathOperation, ComposeCompositorMathNodes)
        elif self.editor_type == "TextureNodeTree":
            return (TextureMathOperation, ComposeTextureMathNodes)
        else:
            return None

    def generate_tree(self, expression: str) -> tuple[bool, str | Tree]:
        op_type = self.current_operation_type()
        if op_type is None:
            return (False, "No known operation type available")
        op, _ = op_type
        try:
            mod = convert_expression(expression)
            op.validate(mod)
            expr: ast.Expr = mod.body[0]
        except SyntaxError:
            return (False, "Could not parse expression")

        try:
            return True, op.parse(expr).to_tree()
        except Exception as e:
            return (False, f"{e}")

    def execute(self, context: Context):
        # Create nodes from tree

        bpy.ops.node.select_all(action="DESELECT")

        tree = self.generate_tree(self.expression)[1]
        if not isinstance(tree, Tree):
            return {"CANCELLED"}

        _, composer_class = self.current_operation_type()

        composer = composer_class(
            use_reroutes=self.use_reroutes,
            center_nodes=self.center_nodes,
            hide_nodes=self.hide_nodes,
        )

        return composer.run(tree, context)

    def draw(self, context: Context):
        layout = self.layout

        if self.current_operation_type() is None:
            layout.label(text="This node editor is currently not supported!")
            return

        layout.prop(self, "expression")

        options_box = layout.box()
        options = options_box.column(align=True)

        if self.editor_type != "TextureNodeTree":
            options.prop(self, "use_reroutes")
        options.prop(self, "hide_nodes")
        options.prop(
            self,
            "center_nodes",
        )
        options.prop(self, "show_available_functions")

        if self.show_available_functions:
            functions_box = layout.column()
            functions_box.label(text="Available functions:")
            functions_box.separator()
            b = functions_box.box()
            row = b.row()
            for type_, funcs in PRINTABLE_SHADER_MATH_CALLS.items():
                func_row = row.column(heading=type_)
                for func in funcs:
                    func_row.label(text=func, translate=False)

        txt = self.expression

        if txt:
            valid, err_msg = self.generate_tree(txt)
            if not valid:  # print the error and a picture
                if context.preferences.addons[__name__].preferences.debug_prints:
                    print(err_msg)

                b = layout.box()
                functions_box = b.column()
                functions_box.label(text="Error:")
                for line in err_msg.splitlines():
                    functions_box.label(text=line, translate=False)


class Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    debug_prints: bpy.props.BoolProperty(
        name="Debug prints",
        description="Enables debug prints in the terminal.",
        default=False,
    )

    use_reroutes: bpy.props.BoolProperty(
        name="Use reroutes instead of Values",
        description="Required in Texture Node Editor.",
    )

    def draw(self, context):
        layout = self.layout

        row = layout.column(align=True)
        row.label(
            text="Check the Keymaps settings to edit activation. Default is Ctrl + M"
        )
        row.prop(self, "debug_prints")
        row.prop(self, "use_reroutes")


addon_keymaps = []


def registerKeymaps():
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        km = wm.keyconfigs.addon.keymaps.get("Node Editor")
        if not km:
            km = wm.keyconfigs.addon.keymaps.new(
                name="Node Editor", space_type="NODE_EDITOR"
            )

        kmi = km.keymap_items.new(
            "node.compose_qm_nodes",
            "M",
            "PRESS",
            ctrl=True,
        )
        addon_keymaps.append((km, kmi))


def unregisterKeymaps():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()


classes = (
    ComposeNodeTree,
    Preferences,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)
    registerKeymaps()


def unregister():
    unregisterKeymaps()
    from bpy.utils import unregister_class

    for cls in reversed(classes):
        unregister_class(cls)

    registerKeymaps()


if __package__ == "__main__":
    register()
