import ast
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import bpy
from bpy.types import UILayout

from .constants import ASSUMABLE_CONSTANTS
from .operations import Operation, Tree


def _new_reroute(nt: bpy.types.NodeTree, name: str):
    node = nt.nodes.new("NodeReroute")
    if name:
        node.label = name
    return node


@dataclass
class ComposeNodes:
    socket_type: str = "VALUE"
    center_nodes: bool = True
    hide_nodes: bool = False

    tree_type: ClassVar[str]
    group_type: ClassVar[str]

    def preview_generate(self, root: Operation, layout: UILayout) -> None:
        """Generates a node tree, but inside a uiLayout for previewing"""
        child_col = layout.column(align=True)
        for input in root.inputs:
            input_row = child_col.box().row()
            if isinstance(input, Operation):
                # generates preview for the child operation
                self.preview_generate(input, input_row)
            elif not self.hide_nodes:
                input_row.label(text=str(input))

        # create a label for the current node's name
        namecol = layout.column()
        namecol.label(text=root.name)
        namecol.separator(type="LINE")

    def run(
        self,
        source: ast.Expr,
        tree: Tree,
        context: bpy.types.Context,
    ):
        bpy.ops.node.select_all(action="DESELECT")
        # Hmm...
        space_data: bpy.types.SpaceNodeEditor = context.space_data
        nt: bpy.types.NodeTree = space_data.edit_tree

        layers = self.execute(source, tree, nt, group_offset=space_data.cursor_location)

        if self.socket_type != "GROUP":
            offset = space_data.cursor_location
        else:
            offset = (0.0, 0.0)

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
        for l_idx, (layer, height) in enumerate(zip(layers, layer_heights)):
            # offset the layer by user-defined rules
            if self.center_nodes:
                layer_offset = (
                    offset[0],
                    offset[1] - ((max_height / 2) - (height / 2)),
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

    def execute(
        self,
        source: ast.Expr,
        tree: Tree,
        nt: bpy.types.NodeTree,
        group_offset=(0, 0),
    ) -> list[list[bpy.types.Node]]:
        sublayers: list[list[bpy.types.Node]]

        if self.socket_type == "GROUP":
            group = bpy.data.node_groups.new(ast.unparse(source), self.tree_type)

            interface: bpy.types.NodeTreeInterface = group.interface

            node, sublayers, inputs = tree.root.generate(group)

            sublayers.insert(0, [node])

            # create the input sockets
            group_input = group.nodes.new("NodeGroupInput")
            sublayers.append([group_input])

            # create the output socket
            group_output = group.nodes.new("NodeGroupOutput")
            interface.new_socket(
                name="Output", in_out="OUTPUT", socket_type="NodeSocketFloat"
            )
            group.links.new(node.outputs[0], group_output.inputs[0])
            sublayers.insert(0, [group_output])

            # add the variables to the interface
            sockets = {}
            for variable in tree.variables:
                socket = interface.new_socket(
                    name=variable, in_out="INPUT", socket_type="NodeSocketFloat"
                )
                if variable in ASSUMABLE_CONSTANTS:
                    socket.default_value = ASSUMABLE_CONSTANTS[variable]
                sockets[variable] = group_input.outputs[variable]

            # add group to node tree
            node = nt.nodes.new(self.group_type)
            node.location = group_offset
            node.node_tree = group

            nt = group

        else:
            node, sublayers, inputs = tree.root.generate(nt)

            sublayers.insert(0, [node])

            # Create the variables nodes
            sockets = {}
            layer = []
            for variable in tree.variables:
                if self.socket_type == "REROUTE":
                    node = _new_reroute(nt, name=variable)
                elif self.socket_type == "VALUE":
                    node = self._new_value(nt, name=variable)
                layer.append(node)
                sockets[variable] = node.outputs[0]
            sublayers.append(layer)

        # Connect the nodes to the corresponding sockets
        for variable, input in inputs:
            nt.links.new(sockets[variable], input)

        return sublayers


class ComposeShaderMathNodes(ComposeNodes):
    tree_type = "ShaderNodeTree"
    group_type = "ShaderNodeGroup"

    @staticmethod
    def _new_value(nt: bpy.types.NodeTree, name: str = ""):
        node = nt.nodes.new("ShaderNodeValue")
        if name:
            node.label = name
            if name in ASSUMABLE_CONSTANTS:
                node.outputs[0].default_value = ASSUMABLE_CONSTANTS[name]
        return node


class ComposeGeometryMathNodes(ComposeShaderMathNodes):
    tree_type = "GeometryNodeTree"
    group_type = "GeometryNodeGroup"


class ComposeCompositorMathNodes(ComposeNodes):
    tree_type = "CompositorNodeTree"
    group_type = "CompositorNodeGroup"

    @staticmethod
    def _new_value(nt: bpy.types.NodeTree, name: str = ""):
        node = nt.nodes.new("CompositorNodeValue")
        if name:
            node.label = name
            if name in ASSUMABLE_CONSTANTS:
                node.outputs[0].default_value = ASSUMABLE_CONSTANTS[name]
        return node


class ComposeTextureMathNodes(ComposeNodes):
    tree_type = "TextureNodeTree"
    group_type = "TextureNodeGroup"
