from abc import abstractmethod
from typing import ClassVar

import bpy

from .constants import ASSUMABLE_CONSTANTS


class NodeCreator:
    tree_type: ClassVar[str]
    group_type: ClassVar[str]

    @classmethod
    def node_group(cls, name: str) -> bpy.types.NodeTree:

        if bpy.data.node_groups.get(name):
            idx = 1
            while bpy.data.node_groups.get(new_name := f"{name}_{idx:03d}"):
                idx += 1
            name = new_name

        return bpy.data.node_groups.new(name, cls.tree_type)

    @staticmethod
    @abstractmethod
    def math_node(nt: bpy.types.NodeTree, op: str = "") -> bpy.types.Node:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def input_value(nt: bpy.types.NodeTree, name: str = "") -> bpy.types.Node:
        raise NotImplementedError

    @staticmethod
    def reroute(nt: bpy.types.NodeTree, name: str) -> bpy.types.Node:
        node = nt.nodes.new("NodeReroute")
        if name:
            node.label = name
        return node


def shader_math_node(nt: bpy.types.NodeTree, op: str = "") -> bpy.types.Node:
    node = nt.nodes.new("ShaderNodeMath")
    if op:
        node.operation = op
    return node


def shader_value(nt: bpy.types.NodeTree, name: str = "") -> bpy.types.Node:
    node = nt.nodes.new("ShaderNodeValue")
    if name:
        node.label = name
        if name in ASSUMABLE_CONSTANTS:
            node.outputs[0].default_value = ASSUMABLE_CONSTANTS[name]
    return node

class ShaderNodeCreator(NodeCreator):
    tree_type = "ShaderNodeTree"
    group_type = "ShaderNodeGroup"

    math_node = staticmethod(shader_math_node)
    input_value = staticmethod(shader_value)


class GeoNodeCreator(NodeCreator):
    tree_type = "GeometryNodeTree"
    group_type = "GeometryNodeGroup"

    math_node = staticmethod(shader_math_node)
    input_value = staticmethod(shader_value)


class CompNodeCreator(NodeCreator):
    tree_type = "CompositorNodeTree"
    group_type = "CompositorNodeGroup"

    math_node = staticmethod(shader_math_node)

    @staticmethod
    def input_value(nt: bpy.types.NodeTree, name: str = ""):
        node = nt.nodes.new("CompositorNodeValue")
        if name:
            node.label = name
            if name in ASSUMABLE_CONSTANTS:
                node.outputs[0].default_value = ASSUMABLE_CONSTANTS[name]
        return node


class TextureNodeCreator(NodeCreator):
    tree_type = "TextureNodeTree"
    group_type = "TextureNodeGroup"

    @staticmethod
    def math_node(nt: bpy.types.NodeTree, name: str) -> bpy.types.Node:
        node: bpy.types.TextureNodeMath = nt.nodes.new("TextureNodeMath")
        node.operation = name
        return node

    input_value = staticmethod(shader_value)
