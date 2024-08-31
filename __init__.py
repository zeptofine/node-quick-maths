from __future__ import annotations

import ast

import bpy
from bpy.types import Context, Event

from .constants import PRINTABLE_SHADER_MATH_CALLS
from .node_composers import (
    ComposeCompositorMathNodes,
    ComposeGeometryMathNodes,
    ComposeNodes,
    ComposeShaderMathNodes,
    ComposeTextureMathNodes,
)
from .operations import (
    CompositorMathOperation,
    Operation,
    ShaderMathOperation,
    TextureMathOperation,
    Tree,
)
from .rustlike_result import Err, Ok, Result

InputSocketType = [
    ("VALUE", "Value", "Use values to connect variables."),
    ("REROUTE", "Reroute", "Use reroute nodes to connect variables."),
    ("GROUP", "Group", "Create a group for the generated nodes."),
]

VariableSortMode = [
    ("NONE", "None", "Variables are sorted pseudorandomly"),
    ("ALPHABET", "Alphabet", "Variables are sorted alphabetically"),
    (
        "INSERTION",
        "Insertion",
        "Variables are sorted based on when they were added in the chain",
    ),
]


class ComposeNodeTree(bpy.types.Operator):
    bl_idname = "node.compose_qm_nodes"
    bl_label = "Node Quick Maths: Compose nodes"

    show_available_functions: bpy.props.BoolProperty(
        name="Show available functions",
        default=False,
    )
    hide_nodes: bpy.props.BoolProperty(
        name="Hide nodes",
        description="Hides nodes to preserve grid space",
    )
    center_nodes: bpy.props.BoolProperty(
        name="Center nodes",
        description="Centers the generated nodes. This is a personal preference",
    )

    editor_type: bpy.props.StringProperty(name="Editor Type")
    expression: bpy.props.StringProperty(
        description="The source expression for the nodes"
    )
    input_socket_type: bpy.props.EnumProperty(items=InputSocketType, default="REROUTE")

    generate_previews: bool
    var_sort_mode: str

    def invoke(self, context: Context, event: Event) -> set[str]:
        wm = context.window_manager
        ui_mode = context.area.ui_type
        self.editor_type = ui_mode
        if context.preferences.addons[__package__].preferences.debug_prints:
            print(f"NQM: Editor type: {self.editor_type}")

        self.generate_previews = context.preferences.addons[
            __package__
        ].preferences.generate_previews
        self.var_sort_mode = context.preferences.addons[
            __package__
        ].preferences.sort_vars

        return wm.invoke_props_dialog(self, confirm_text="Create", width=600)

    def current_operation_type(
        self,
    ) -> tuple[type[Operation], type[ComposeNodes]] | None:
        if self.editor_type == "ShaderNodeTree":
            return (ShaderMathOperation, ComposeShaderMathNodes)
        elif self.editor_type == "GeometryNodeTree":
            return (ShaderMathOperation, ComposeGeometryMathNodes)
        elif self.editor_type == "CompositorNodeTree":
            return (CompositorMathOperation, ComposeCompositorMathNodes)
        elif self.editor_type == "TextureNodeTree":
            return (TextureMathOperation, ComposeTextureMathNodes)
        else:
            return None

    def generate_tree(self, expression: str) -> Result[tuple[ast.Expr, Tree], str]:
        op_type = self.current_operation_type()
        if op_type is None:
            return Err("No known operation type available")
        op, _ = op_type
        try:
            mod = ast.parse(expression.strip(), mode="exec")

            r = op.validate(mod)
            if r.is_err():
                return Err(r.unwrap_err())

            expr: ast.Expr = mod.body[0]
        except SyntaxError as e:
            print(e)
            return Err("Could not parse expression")

        try:
            parsed = op.parse(expr)
        except Exception as e:
            return Err(str(e))
        if not isinstance(parsed, Operation):
            return Err("Parsed expression is not an Operation")

        return Ok((expr, parsed.to_tree(sort_mode=self.var_sort_mode)))

    def execute(self, context: Context):
        # Create nodes from tree

        bpy.ops.node.select_all(action="DESELECT")

        tree = self.generate_tree(self.expression)
        o = self.current_operation_type()
        if tree.is_err() or o is None:
            return {"CANCELLED"}

        expr, tree = tree.unwrap()
        _, composer_class = o

        composer = composer_class(
            socket_type=self.input_socket_type,
            center_nodes=self.center_nodes,
            hide_nodes=self.hide_nodes,
        )

        return composer.run(expr, tree, context)

    def draw(self, context: Context):
        layout = self.layout

        o = self.current_operation_type()

        if o is None:
            layout.label(text="This node editor is currently not supported!")
            return

        opt, comp = o

        layout.prop(self, "expression")

        options_box = layout.box()
        options = options_box.column(align=True)

        options.row().prop(self, "input_socket_type", expand=True)
        options = options.row()
        col1 = options.column(align=True)
        col1.prop(self, "hide_nodes")
        col1.prop(
            self,
            "center_nodes",
        )
        col2 = options.column(align=True)
        col2.prop(self, "show_available_functions")

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
            r = self.generate_tree(txt)
            if r.is_err():  # print the error
                msg = r.unwrap_err()

                if context.preferences.addons[__package__].preferences.debug_prints:
                    print(msg)

                b = layout.box()
                err_col = b.column()
                err_col.label(text="Error:")
                for line in msg.splitlines():
                    err_col.label(text=line, translate=False)
            elif (
                self.generate_previews
            ):  # create a representation of the node tree under the settings
                preview_box = layout.box()

                composer = comp(
                    socket_type=self.input_socket_type,
                    center_nodes=self.center_nodes,
                    hide_nodes=self.hide_nodes,
                )
                composer.preview_generate(r.unwrap()[1].root, preview_box.row())


class Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    debug_prints: bpy.props.BoolProperty(
        name="Debug Print",
        description="Enables debug prints in the terminal",
        default=False,
    )

    generate_previews: bpy.props.BoolProperty(
        name="Generate Previews",
        description="Generates previews of node trees before creating them",
        default=True,
    )

    sort_vars: bpy.props.EnumProperty(
        items=VariableSortMode,
        name="Sort variables",
        description="The order which to sort variables",
        default="INSERTION",
    )

    def draw(self, context):
        layout = self.layout

        row = layout.column(align=True)
        row.label(
            text="Check the Keymaps settings to edit activation. Default is Ctrl + M"
        )
        row.prop(self, "debug_prints")
        row.prop(self, "generate_previews")

        r = row.row()
        r.label(text="Sort variables by...")
        r.prop(self, "sort_vars", expand=True)


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


if __name__ == "__main__":
    register()
