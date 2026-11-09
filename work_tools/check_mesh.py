"""Check the active mesh for unwelded vertices, extra UV channels, and n-gons.

One button runs all three. Unwelded vertices and n-gons are selected in
Edit Mode. Extra UV channels are only reported. Show UV Seams is separate
and is not part of the check or the Maya send.
"""

import struct

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


def _uv_key(uv):
    """Same key the FBX exporter uses, so Maya's seam matches ours.

    Maya splits a UV point when the values are not the exact same float, even
    if they sit on top of each other in Blender's UV editor.
    """
    u = float(uv[0]) or 0.0
    v = float(uv[1]) or 0.0
    return struct.pack("<ff", u, v)


def _uvs_match(first, second):
    return _uv_key(first) == _uv_key(second)


def uv_seam_edges(mesh):
    """Edges that border a UV island: mesh borders and splits in the active UV map."""
    layer = mesh.uv_layers.active
    if layer is None:
        return None
    sides = {}
    for poly in mesh.polygons:
        start = poly.loop_start
        total = poly.loop_total
        for offset in range(total):
            loop = mesh.loops[start + offset]
            nxt = mesh.loops[start + (offset + 1) % total]
            side = {
                loop.vertex_index: tuple(layer.data[start + offset].uv),
                nxt.vertex_index: tuple(layer.data[start + (offset + 1) % total].uv),
            }
            sides.setdefault(loop.edge_index, []).append(side)
    seams = []
    for edge_index, records in sides.items():
        if len(records) < 2:
            seams.append(edge_index)
            continue
        base = records[0]
        split = False
        for other in records[1:]:
            for vert_index, uv in base.items():
                if vert_index not in other or not _uvs_match(uv, other[vert_index]):
                    split = True
                    break
            if split:
                break
        if split:
            seams.append(edge_index)
    return seams


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


def _select_edges(context, obj, edge_indices):
    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    mesh = obj.data
    edges = set(edge_indices)
    for vert in mesh.vertices:
        vert.select = False
    for edge in mesh.edges:
        edge.select = edge.index in edges
    for poly in mesh.polygons:
        poly.select = False
    context.tool_settings.mesh_select_mode = (False, True, False)
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


class OBJECT_OT_show_uv_seams(bpy.types.Operator):
    bl_idname = "object.show_work_tools_uv_seams"
    bl_label = "Show UV Seams"
    bl_description = "Select edges whose UVs are not exactly shared, including points that only look stitched. Not part of Check Mesh or Send to Maya"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode in {"OBJECT", "EDIT_MESH"}

    def execute(self, context):
        obj = context.active_object
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        seams = uv_seam_edges(obj.data)
        if seams is None:
            self.report({"WARNING"}, "The mesh has no UV map")
            return {"CANCELLED"}
        _select_edges(context, obj, seams)
        self.report({"INFO"}, f"UV seams: {len(seams)}")
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
        col.separator()
        col.operator(OBJECT_OT_show_uv_seams.bl_idname, icon="UV")


def draw_object_menu(layout):
    layout.operator(OBJECT_OT_check_mesh.bl_idname, icon="VIEWZOOM")
    layout.operator(OBJECT_OT_show_uv_seams.bl_idname, icon="UV")


_CLASSES = (
    CheckMeshSettings,
    OBJECT_OT_check_mesh,
    OBJECT_OT_show_uv_seams,
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
