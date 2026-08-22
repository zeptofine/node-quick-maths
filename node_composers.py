from dataclasses import dataclass

import bpy
from bpy.types import Node, UILayout

from .constants import ASSUMABLE_CONSTANTS
from .operations import LayerList, Operation, Tree


@dataclass
class ComposeNodes:
    socket_type: str = "VALUE"
    center_nodes: bool = True
    hide_nodes: bool = False

    def preview(self, root: Operation, layout: UILayout) -> None:
        """Generates a node tree, but inside a uiLayout for previewing"""
        child_col = layout.column(align=True)
        for input in root.inputs:
            input_row = child_col.box().row()
            if isinstance(input, Operation):
                # generates preview for the child operation
                self.preview(input, input_row)
            elif not self.hide_nodes:
                input_row.label(text=str(input))

        # create a label for the current node's name
        namecol = layout.column()
        namecol.label(text=root.name)
        namecol.separator(type="LINE")

    def run(
        self,
        tree: Tree,
        context: bpy.types.Context,
    ):
        bpy.ops.node.select_all(action="DESELECT")

        space_data: bpy.types.SpaceNodeEditor = context.space_data
        nt: bpy.types.NodeTree = space_data.edit_tree

        # create nodes in layers
        layers: list[list[Node]]
        if self.socket_type == "GROUP":
            group, layers = self._group_layers(tree, nt, group_offset=space_data.cursor_location)

            # add group to parent node tree
            node = nt.nodes.new(tree.group_type)
            node.location = space_data.cursor_location
            node.node_tree = group
        else:
            layers = self._node_layers(tree, nt)

        # Move them around so they look nice in the editor
        self.translate_nodes(space_data, layers)

        bpy.ops.node.translate_attach_remove_on_cancel("INVOKE_DEFAULT")

        return {"FINISHED"}

    def translate_nodes(
        self,
        space_data: bpy.types.SpaceNodeEditor,
        layers: LayerList,
    ):
        if self.socket_type != "GROUP":
            offset = space_data.cursor_location
        else:
            offset = (0.0, 0.0)

        # node.height is too small to represent the actual visible height of the nodes
        # for some reason, so these are a little bigger than the width margin
        # UPDATE: the proper way is to get node.dimensions but they are only populated after redraw.
        #         Worth it?
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
            height -= height_margin
            layer_heights.append(height)

        max_height = max(layer_heights)

        # loop through every layer in the subtrees, move them to the correct location
        for l_idx, (layer, height) in enumerate(zip(layers, layer_heights)):
            # offset the layer by user-defined rules
            layer_offset = (offset[0], offset[1] - (max_height / 2 - height / 2)) if self.center_nodes else offset

            # move the nodes
            h = 0.0
            for n_idx, node in enumerate(layer):
                position = (
                    (layer_offset[0] - (node.width + width_margin) * l_idx),
                    (layer_offset[1] - (node.height + height_margin) * n_idx),
                )
                node.location = position

                nodeheight = node.height
                h += nodeheight + height_margin

    def _group_layers(self, tree: Tree, nt: bpy.types.NodeTree, group_offset: tuple[float, float]):

        group = tree.new_group_tree()
        interface = group.interface

        layers, connections = tree.layers_and_connections(group)

        # create the input sockets
        group_input = group.nodes.new("NodeGroupInput")
        layers.append([group_input])

        # create the output socket
        group_output = group.nodes.new("NodeGroupOutput")
        interface.new_socket(name="Output", in_out="OUTPUT", socket_type="NodeSocketFloat")
        group.links.new(layers[0][0].outputs[0], group_output.inputs[0])
        layers.insert(0, [group_output])

        # add the variables to the interface
        sockets: dict[str, bpy.types.NodeSocket] = {}
        for variable in tree.variables:
            socket = interface.new_socket(name=variable, in_out="INPUT", socket_type="NodeSocketFloat")
            if variable in ASSUMABLE_CONSTANTS:
                socket.default_value = ASSUMABLE_CONSTANTS[variable]
            sockets[variable] = group_input.outputs[variable]

        # Connect the nodes to the corresponding group incoming sockets
        for input in connections:
            group.links.new(sockets[input.var], input.socket)

        return group, layers

    def _node_layers(
        self,
        tree: Tree,
        nt: bpy.types.NodeTree,
    ):
        layers, connections = tree.layers_and_connections(nt)

        # Create the variables nodes
        sockets = {}
        layer = []
        for variable in tree.variables:
            if self.socket_type == "VALUE" or variable in ASSUMABLE_CONSTANTS:
                node = tree.creator.input_value(nt, name=variable)
            elif self.socket_type == "REROUTE":
                node = tree.creator.reroute(nt, name=variable)
            layer.append(node)
            sockets[variable] = node.outputs
        layers.append(layer)

        # Connect the nodes to the corresponding sockets
        for input in connections:
            nt.links.new(sockets[input.var][-1], input.socket)

        return layers
