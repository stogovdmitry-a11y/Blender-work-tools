"""Shrinkwrap the other selected mesh onto the active one and apply it.

Select two meshes. The active one stays put: it is the target. The other
mesh is the source, and it is the one that moves. There is no source picker.
An optional vertex group on the source limits which of its vertices move.
"""

import bpy
from bpy.props import PointerProperty, StringProperty


class ShrinkwrapError(Exception):
    """User-facing shrinkwrap failure. The message is shown in the status bar."""


def _other_selected(context, target):
    selected = list(context.selected_objects)
    if target not in selected or len(selected) != 2:
        raise ShrinkwrapError("Select two meshes. The active one is the target")
    source = selected[0] if selected[1] == target else selected[1]
    if source.type != "MESH" or target.type != "MESH":
        raise ShrinkwrapError("Select two meshes. The active one is the target")
    if source == target:
        raise ShrinkwrapError("The source and the target are the same object")
    return source


def apply_shrinkwrap(context, source, target, vertex_group=""):
    if vertex_group and vertex_group not in source.vertex_groups:
        raise ShrinkwrapError(f"Vertex group '{vertex_group}' was not found on {source.name}")

    modifier = source.modifiers.new(name="Shrinkwrap", type="SHRINKWRAP")
    modifier.target = target
    modifier.wrap_method = "NEAREST_SURFACEPOINT"
    if vertex_group:
        modifier.vertex_group = vertex_group

    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    try:
        view_layer.objects.active = source
        try:
            result = bpy.ops.object.modifier_apply(modifier=modifier.name)
        except RuntimeError as exc:
            if modifier.name in source.modifiers:
                source.modifiers.remove(modifier)
            raise ShrinkwrapError(str(exc)) from exc
    finally:
        view_layer.objects.active = previous_active
    if "FINISHED" not in result:
        if modifier.name in source.modifiers:
            source.modifiers.remove(modifier)
        raise ShrinkwrapError("Could not apply the Shrinkwrap modifier")


class ShrinkwrapSettings(bpy.types.PropertyGroup):
    vertex_group: StringProperty(
        name="Group",
        description="Vertex group on the source mesh. Empty shrinkwraps the whole source",
    )


class OBJECT_OT_apply_shrinkwrap(bpy.types.Operator):
    bl_idname = "object.apply_work_tools_shrinkwrap"
    bl_label = "Apply Shrinkwrap"
    bl_description = "Shrinkwrap the other selected mesh onto the active mesh and apply the modifier"
    bl_options = {"REGISTER", "UNDO"}

    vertex_group: StringProperty(name="Group")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode == "OBJECT"

    def invoke(self, context, event):
        self.vertex_group = context.scene.work_tools_shrinkwrap.vertex_group
        return self.execute(context)

    def execute(self, context):
        target = context.active_object
        try:
            source = _other_selected(context, target)
            apply_shrinkwrap(context, source, target, self.vertex_group)
        except ShrinkwrapError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        group = f" in '{self.vertex_group}'" if self.vertex_group else ""
        self.report({"INFO"}, f"Shrinkwrapped {source.name}{group} onto {target.name}")
        return {"FINISHED"}


class VIEW3D_PT_shrinkwrap(bpy.types.Panel):
    bl_label = "Shrinkwrap"
    bl_idname = "VIEW3D_PT_work_tools_shrinkwrap"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Work Tools"

    def draw(self, context):
        layout = self.layout
        if context.mode != "OBJECT":
            layout.label(text="Object Mode is required")
            return
        settings = context.scene.work_tools_shrinkwrap
        target = context.active_object
        col = layout.column(align=True)
        if target is None or target.type != "MESH":
            col.label(text="Target is the active mesh")
            col.prop(settings, "vertex_group", text="Group")
        else:
            col.label(text=f"Target: {target.name}")
            others = [obj for obj in context.selected_objects if obj != target and obj.type == "MESH"]
            if len(others) == 1:
                col.label(text=f"Source: {others[0].name}")
                col.prop_search(settings, "vertex_group", others[0], "vertex_groups", text="Group")
            else:
                col.prop_search(settings, "vertex_group", target, "vertex_groups", text="Group")
                col.label(text="Select one other mesh as the source")
        col.operator(OBJECT_OT_apply_shrinkwrap.bl_idname, icon="MOD_SHRINKWRAP")


def draw_object_menu(layout):
    layout.operator(OBJECT_OT_apply_shrinkwrap.bl_idname, icon="MOD_SHRINKWRAP")


_CLASSES = (
    ShrinkwrapSettings,
    OBJECT_OT_apply_shrinkwrap,
    VIEW3D_PT_shrinkwrap,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.work_tools_shrinkwrap = PointerProperty(type=ShrinkwrapSettings)


def unregister():
    del bpy.types.Scene.work_tools_shrinkwrap
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
