import ast
from dataclasses import dataclass
import traceback
from typing import Self, Set

import bpy
from bpy.props import BoolProperty
from bpy.types import Context, Event


def convert_expression(s: str) -> ast.Module:
    return ast.parse(s, mode="exec")


BAD_NODES = (
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
    ast.Compare,
    ast.FormattedValue,
    ast.JoinedStr,
    ast.Attribute,
    ast.Subscript,
    ast.Starred,
    ast.List,
    ast.Tuple,
    ast.Slice,
)


CALLS = {  # Function inputs in python, names Blender
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
    "cmp": ((3,), "COMPARE"),  # (x, y) > c
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
    "rad": (1,),  # (x), RADIANS
    "deg": (1,),  # (x), DEGREES
}

BASIC_OPS = {
    ast.Add: "ADD",
    ast.Sub: "SUBTRACT",
    ast.Mult: "MULTIPLY",
    ast.Div: "DIVIDE",
    ast.Mod: "MODULO",
    ast.Pow: "POWER",
}


def validate_module(e: ast.Module) -> tuple[bool, str]:
    # check that the node in the ast body is just an Expr
    expr: ast.Expr

    if not isinstance((expr := e.body[0]), ast.Expr):
        return (False, "Invalid expression type. Only create math expressions!")

    for node in ast.walk(expr):
        # check if node is bad
        if any(isinstance(node, bad_node) for bad_node in BAD_NODES):
            return (False, f"Do not use node of type: {type(node)} ")

        # check if node is a constant and it is a disallowed type
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                return (
                    False,
                    f"Constants cannot be anything other than ints or floats.\n{node.value} is disallowed",
                )

        # check if node is a call and it has an allowed function name
        if isinstance(node, ast.Call):
            name: ast.Name = node.func
            allowed_nums = CALLS.get(name.id)
            if allowed_nums is None:
                return (
                    False,
                    f"Do not use a function named '{name.id}',\nOnly choose from available modes in the math node!",
                )

            # check if the number of arguments align with the number of arguments in the GOOD_CALLS
            if all(len(node.args) != x for x in allowed_nums[0]):
                return (
                    False,
                    f"Function {name.id} is allowed, but\nthe number of arguments is wrong\n({len(node.args)} is not in {allowed_nums[0]})",
                )

    return (True, "")


VARIABLE_NAME = str


@dataclass
class Operation:
    name: str
    inputs: "list[int | float | str | Operation]"

    def parse(e: ast.expr) -> int | float | str | Self:
        if isinstance(e, ast.BinOp):
            op = e.op

            if type(op) in BASIC_OPS:
                return Operation(
                    name=BASIC_OPS[type(op)],
                    inputs=[Operation.parse(e.left), Operation.parse(e.right)],
                )
            elif isinstance(op, ast.FloorDiv):
                return Operation(
                    name="FLOOR",
                    inputs=[
                        Operation(
                            name="DIVIDE",
                            inputs=[Operation.parse(e.left), Operation.parse(e.right)],
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
                return Operation(
                    name="MULTIPLY",
                    inputs=[Operation.parse(e.operand), -1],
                )

        if isinstance(e, ast.Call):
            inputs = []
            for arg in e.args:
                inputs.append(Operation.parse(arg))

            return Operation(name=CALLS[e.func.id][1], inputs=inputs)

        if isinstance(e, ast.Constant):
            return e.value

        if isinstance(e, ast.Name):
            return e.id

        if isinstance(e, ast.Expr):
            return Operation.parse(e.value)

        raise NotImplementedError(f"Unhandled expression {ast.dump(e, indent=4)}")

    def variables(self) -> set[VARIABLE_NAME]:
        vars: set[VARIABLE_NAME] = set()
        for input in self.inputs:
            if isinstance(input, Operation):
                vars.update(input.variables())
            elif isinstance(input, VARIABLE_NAME):
                vars.add(input)
        return vars


@dataclass
class Tree:
    variables: list[VARIABLE_NAME]
    toplevel_operation: Operation

    def parse(e: ast.Expr) -> Self:
        op = Operation.parse(e)
        assert isinstance(
            op, Operation
        ), f"{type(e)} is not a math operation:\n{ast.dump(e, indent=4)}"
        return Tree(variables=op.variables(), toplevel_operation=op)


def create_nodes(
    nt: bpy.types.NodeTree,
    parent: Operation,
    offset=(0, 0),
) -> tuple[
    bpy.types.ShaderNodeMath,  # parent
    tuple[tuple[int, int], tuple[int, int]],  # node bounds
    list[tuple[str, bpy.types.NodeInputs, int]],
]:
    """
    Creates nodes. returns a list of tuples of
    (
        variable name,
        bounds of (topleft, bottomright),
        input socket
    )
    """
    parent_node: bpy.types.ShaderNodeMath = nt.nodes.new("ShaderNodeMath")
    parent_node.operation = parent.name

    # I'll be honest i just tweaked the offset stuff until it worked right
    lefttop = offset
    rightbottom = (lefttop[0] + parent_node.width, lefttop[1] - parent_node.height)

    oinputs = []

    for idx, child in enumerate(parent.inputs):
        if isinstance(child, Operation):
            node, bounds, inputs = create_nodes(
                nt,
                child,
                offset=(lefttop[0], lefttop[1] - idx * (parent_node.height + 60)),
            )
            oinputs.extend(inputs)
            nt.links.new(node.outputs[0], parent_node.inputs[idx])

            # update the bounds of the total tree
            lefttop = (max(lefttop[0], bounds[0][0]), max(lefttop[1], bounds[0][1]))
            rightbottom = (
                max(rightbottom[0], bounds[1][0]),
                min(rightbottom[1], bounds[1][1]),
            )

        elif isinstance(child, str):  # add it to the oinputs
            oinputs.append((child, parent_node.inputs[idx]))
        else:
            parent_node.inputs[idx].default_value = child

    # use the topleft and bottomright to update the parent_node's location,
    # which should be  to the top-right of the children
    location = (
        rightbottom[0] + parent_node.width + 60,  # x
        # rightbottom[0],
        lefttop[1],
    )
    parent_node.location = location

    # update the bound
    rightbottom = (max(location[0], rightbottom[0]), max(location[1], rightbottom[1]))

    return parent_node, (lefttop, rightbottom), oinputs


class ComposeNodeTree(bpy.types.Operator):
    bl_idname = "node.compose_qm_nodes"
    bl_label = "Node Quick Maths: Compose nodes"

    is_in_shaders: bpy.props.BoolProperty(name="Is in shaders")
    type: bpy.props.StringProperty(description="the type of node editor being shown")
    expression: bpy.props.StringProperty(
        description="The source expression for the nodes."
    )

    tree: Tree

    def invoke(self, context: Context, event: Event) -> Set[str]:
        wm = context.window_manager
        ui_mode = context.area.ui_type
        self.is_in_shaders = ui_mode == "ShaderNodeTree"
        if not self.is_in_shaders:
            print(f"NQM: view is not a shader editor, is in {ui_mode}")

        return wm.invoke_props_dialog(self, confirm_text="Create", width=600)

    def execute(self, context: Context):
        if not self.is_in_shaders:
            return {"CANCELLED"}
        # Create nodes from tree

        bpy.ops.node.select_all(action="DESELECT")

        # Hmm...
        space_data: bpy.types.SpaceNodeEditor = bpy.context.space_data
        node_tree: bpy.types.NodeTree = space_data.edit_tree
        offset = (space_data.cursor_location[0], space_data.cursor_location[1])
        tree = self.tree

        node, bounds, inputs = create_nodes(
            node_tree, tree.toplevel_operation, offset=offset
        )

        vars = {}

        # Create the new variables as Value nodes
        for idx, variable in enumerate(tree.variables):  # create a Value
            node: bpy.types.ShaderNodeValue = node_tree.nodes.new("ShaderNodeValue")
            node.label = variable
            vars[variable] = node

            # using the bounds, place the variables to the left of the tree
            node.location = (
                bounds[0][0] - 40,
                bounds[0][1] - idx * (node.height + 40),
            )

        # Connect the Value nodes to the corresponding sockets
        for variable, input in inputs:
            node_tree.links.new(vars[variable].outputs[0], input)

        bpy.ops.node.translate_attach_remove_on_cancel("INVOKE_DEFAULT")

        return {"FINISHED"}

    def draw(self, context: Context):
        layout = self.layout

        box = layout.box()

        if not self.is_in_shaders:
            box.label(text="This node editor is currently not supported!")
            return

        col = box.column()

        src_row = col.row()
        src_row.prop(self, "expression")

        txt = self.expression
        if txt:
            try:
                mod = convert_expression(txt)
                valid, err_msg = validate_module(mod)
                expr: ast.Expr = mod.body[0]

                try:
                    tree = Tree.parse(expr)
                    self.tree = tree

                    # for line in pformat(tree, width=50).splitlines():
                    #     col.label(text=line, translate=False)
                except Exception as e:
                    traceback.print_exception(e)
                    valid, err_msg = (False, f"{e}")
            except SyntaxError:
                valid, err_msg = (False, "Could not parse expression")

            if not valid:  # print the error and a picture
                for line in err_msg.splitlines():
                    col.label(text=line, translate=False)


class Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    enable_button: BoolProperty(
        name="Quick Access Button",
        description="Enable quick access buttons for switching between different node editors",
        default=True,
    )  # type: ignore

    def draw(self, context):
        layout = self.layout

        row = layout.row(align=True)
        row.label(
            text="Check the Keymaps settings to edit activation. Default is Ctrl + M"
        )


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
