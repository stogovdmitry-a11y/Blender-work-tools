"""Check the active mesh for unwelded vertices, extra UV channels, and n-gons.

One button runs all three. Unwelded vertices and n-gons are selected in
Edit Mode. Extra UV channels are only reported.
"""

import bpy
from bpy.props import PointerProperty, StringProperty


_POSITION_DIGITS = 6


def _unwelded_vertices(mesh):
    groups = {}
    for vert in mesh.vertices:
        key = tuple(round(float(axis), _POSITION_DIGITS) for axis in vert.co)
        groups.setdefault(key, []).append(vert.index)
    found = []
    for indices in groups.values():
        if len(indices) > 1:
            found.extend(indices)
    return found


def _ngon_faces(mesh):
    return [poly.index for poly in mesh.polygons if len(poly.vertices) > 4]


def check_mesh(obj):
    mesh = obj.data
    return {
        "unwelded": _unwelded_vertices(mesh),
        "uv_channels": len(mesh.uv_layers),
        "ngons": _ngon_faces(mesh),
    }


def _report_lines(found):
    lines = []
    if found["unwelded"]:
        lines.append(f"Unwelded vertices: {len(found['unwelded'])}")
    if found["uv_channels"] > 1:
        lines.append(f"UV channels: {found['uv_channels']}")
    if found["ngons"]:
        lines.append(f"N-gons: {len(found['ngons'])}")
    return lines


def _select_problems(context, obj, vert_indices, face_indices):
    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    mesh = obj.data
    verts = set(vert_indices)
    faces = set(face_indices)
    for vert in mesh.vertices:
        vert.select = vert.index in verts
    for edge in mesh.edges:
        edge.select = False
    for poly in mesh.polygons:
        poly.select = poly.index in faces
    context.tool_settings.mesh_select_mode = (bool(verts), False, bool(faces))
    bpy.ops.object.mode_set(mode="EDIT")


class CheckMeshSettings(bpy.types.PropertyGroup):
    report: StringProperty(name="Report", default="")


class OBJECT_OT_check_mesh(bpy.types.Operator):
    bl_idname = "object.check_work_tools_mesh"
    bl_label = "Check Mesh"
    bl_description = "Find unwelded vertices, extra UV channels, and n-gons on the active mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode in {"OBJECT", "EDIT_MESH"}

    def execute(self, context):
        obj = context.active_object
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        found = check_mesh(obj)
        lines = _report_lines(found)
        if not lines:
            context.scene.work_tools_check_mesh.report = "OK"
            self.report({"INFO"}, "OK")
            return {"FINISHED"}
        if found["unwelded"] or found["ngons"]:
            _select_problems(context, obj, found["unwelded"], found["ngons"])
        text = "\n".join(lines)
        context.scene.work_tools_check_mesh.report = text
        self.report({"WARNING"}, text.replace("\n", ". "))
        return {"FINISHED"}


class VIEW3D_PT_check_mesh(bpy.types.Panel):
    bl_label = "Check Mesh"
    bl_idname = "VIEW3D_PT_work_tools_check_mesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Work Tools"
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        if context.mode not in {"OBJECT", "EDIT_MESH"}:
            layout.label(text="Object Mode or Edit Mode is required")
            return
        col = layout.column(align=True)
        target = context.active_object
        if target is not None and target.type == "MESH":
            col.label(text=f"Target: {target.name}")
        else:
            col.label(text="Target is the active mesh")
        col.operator(OBJECT_OT_check_mesh.bl_idname, icon="VIEWZOOM")
        report = context.scene.work_tools_check_mesh.report
        for line in report.split("\n"):
            if line:
                col.label(text=line)


def draw_object_menu(layout):
    layout.operator(OBJECT_OT_check_mesh.bl_idname, icon="VIEWZOOM")


_CLASSES = (
    CheckMeshSettings,
    OBJECT_OT_check_mesh,
    VIEW3D_PT_check_mesh,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.work_tools_check_mesh = PointerProperty(type=CheckMeshSettings)


def unregister():
    del bpy.types.Scene.work_tools_check_mesh
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
