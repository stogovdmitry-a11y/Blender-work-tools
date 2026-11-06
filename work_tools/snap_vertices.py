"""Snap selected vertices of the active mesh onto the nearest source vertices.

The objects stay separate. Only selected target vertices move, and only when
a source vertex is within the distance.
"""

import bpy
import bmesh
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty
from mathutils.kdtree import KDTree

from .seam_normals import MODE_ITEMS, OBJECT_OT_toggle_split_normals, TransferError, transfer_seam_normals


class SnapVertexError(Exception):
    """User-facing snap failure. The message is shown in the status bar."""


def _resolve_source(context, target):
    selected = list(context.selected_objects)
    if target is None or target.type != "MESH" or target not in selected or len(selected) != 2:
        raise SnapVertexError("Select two meshes. The active one is the target")
    source = selected[0] if selected[1] == target else selected[1]
    if source.type != "MESH" or source == target:
        raise SnapVertexError("Select two meshes. The active one is the target")
    return source


def _world_matrix(obj):
    return np.array(obj.matrix_world, dtype=np.float64)


def _to_world(matrix, local):
    if len(local) == 0:
        return local.reshape(0, 3)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def _to_local(matrix, world):
    if len(world) == 0:
        return world.reshape(0, 3)
    inverse = np.linalg.inv(matrix)
    return world @ inverse[:3, :3].T + inverse[:3, 3]


def _read_vertices(obj):
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        if not bm.verts:
            raise SnapVertexError(f"{obj.name} has no vertices")
        local = np.array([vert.co[:] for vert in bm.verts], dtype=np.float64)
        selected = np.array([vert.select for vert in bm.verts], dtype=bool)
        return {"kind": "BMESH", "bm": bm, "mesh": obj.data, "verts": list(bm.verts), "local": local, "selected": selected}

    mesh = obj.data
    count = len(mesh.vertices)
    if count == 0:
        raise SnapVertexError(f"{obj.name} has no vertices")
    local = np.empty(count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", local)
    selected = np.empty(count, dtype=np.bool_)
    mesh.vertices.foreach_get("select", selected)
    return {"kind": "MESH", "mesh": mesh, "local": local.reshape(-1, 3), "selected": selected}


def _write_local(access, local):
    if access["kind"] == "BMESH":
        for vert, co in zip(access["verts"], local):
            vert.co = (float(co[0]), float(co[1]), float(co[2]))
        bmesh.update_edit_mesh(access["mesh"])
        return
    flat = np.asarray(local, dtype=np.float64).reshape(-1)
    access["mesh"].vertices.foreach_set("co", flat)
    access["mesh"].update()


def snap_vertices(source, target, distance):
    if source is None or target is None:
        raise SnapVertexError("Two meshes are required")
    if source.type != "MESH" or target.type != "MESH":
        raise SnapVertexError("Two meshes are required")
    if source == target:
        raise SnapVertexError("The source and the target are the same object")

    source_access = _read_vertices(source)
    source_world = _to_world(_world_matrix(source), source_access["local"])
    target_access = _read_vertices(target)
    selected = target_access["selected"]
    if not selected.any():
        raise SnapVertexError("No vertices are selected")

    tree = KDTree(len(source_world))
    for index, co in enumerate(source_world):
        tree.insert((float(co[0]), float(co[1]), float(co[2])), index)
    tree.balance()

    target_matrix = _world_matrix(target)
    target_world = _to_world(target_matrix, target_access["local"])
    limit = max(float(distance), 0.0)
    snapped = 0
    considered = int(selected.sum())
    for index in np.flatnonzero(selected).tolist():
        co = target_world[index]
        _found, _source_index, dist = tree.find((float(co[0]), float(co[1]), float(co[2])))
        if dist <= limit:
            target_world[index] = source_world[_source_index]
            snapped += 1
    if snapped:
        _write_local(target_access, _to_local(target_matrix, target_world))
    return snapped, considered - snapped


class SnapVertexSettings(bpy.types.PropertyGroup):
    distance: FloatProperty(
        name="Distance",
        description="Maximum world distance for the snap and for the normal transfer",
        default=0.01,
        min=0.0,
        soft_max=1.0,
        precision=4,
        subtype="DISTANCE",
    )
    both: BoolProperty(
        name="Snap and Transfer",
        description="Snap the selected vertices, then transfer normals from the same source. Either button does both",
        default=True,
    )
    mode: EnumProperty(
        name="Affect",
        description="Which target vertices receive normals",
        items=MODE_ITEMS,
        default="SELECTED",
    )
    vertex_group: StringProperty(
        name="Group",
        description="Vertex group on the target",
    )
    interpolate: BoolProperty(
        name="Interpolate Along Boundary",
        description="Sample the nearest open edge on the source. Use this when the new seam has more vertices",
        default=False,
    )


def _load_settings(operator, context):
    settings = context.scene.work_tools_snap_vertices
    operator.distance = settings.distance
    operator.do_both = bool(settings.both)
    operator.mode = settings.mode
    operator.vertex_group = settings.vertex_group
    operator.interpolate = settings.interpolate


def _run_snap_seams(operator, context, action):
    target = context.active_object
    previous_mode = target.mode
    if previous_mode == "EDIT":
        bpy.ops.object.mode_set(mode="OBJECT")
    snapped = None
    snap_missed = 0
    result = None
    try:
        source = _resolve_source(context, target)
        if action == "SNAP" or operator.do_both:
            snapped, snap_missed = snap_vertices(source, target, operator.distance)
            context.view_layer.update()
        if action == "TRANSFER" or operator.do_both:
            result = transfer_seam_normals(
                context,
                source,
                target,
                distance=operator.distance,
                mode=operator.mode,
                vertex_group=operator.vertex_group,
                interpolate=operator.interpolate,
            )
    except SnapVertexError as exc:
        operator.report({"WARNING"}, str(exc))
        return {"CANCELLED"}
    except TransferError as exc:
        if snapped is not None:
            operator.report({"WARNING"}, f"Snapped vertices: {snapped} of {snapped + snap_missed}. {exc}")
            return {"FINISHED"}
        operator.report({"WARNING"}, str(exc))
        return {"CANCELLED"}
    finally:
        if previous_mode == "EDIT" and context.active_object is not None and context.active_object.mode != "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")

    parts = []
    warning = False
    if snapped is not None:
        parts.append(f"Snapped vertices: {snapped} of {snapped + snap_missed}")
        warning = bool(snap_missed)
    if result is not None:
        parts.append(f"Transferred normals: {result.matched} of {result.matched + result.missed}")
        if result.notes:
            parts.append(". ".join(result.notes))
        warning = warning or bool(result.missed or result.notes)
    nothing_moved = snapped in (None, 0)
    nothing_transferred = result is None or result.matched == 0
    if nothing_moved and nothing_transferred:
        operator.report({"WARNING"}, "No vertex fell within the distance")
        return {"CANCELLED"}
    operator.report({"WARNING"} if warning else {"INFO"}, ". ".join(parts))
    return {"FINISHED"}


class OBJECT_OT_snap_vertices(bpy.types.Operator):
    bl_idname = "object.snap_vertices"
    bl_label = "Snap"
    bl_description = "Move selected vertices of the active mesh onto the nearest vertices of the source"
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(name="Distance", default=0.01, min=0.0, soft_max=1.0, precision=4, subtype="DISTANCE")
    do_both: BoolProperty(name="Snap and Transfer", default=True)
    mode: EnumProperty(name="Affect", items=MODE_ITEMS, default="SELECTED")
    vertex_group: StringProperty(name="Group")
    interpolate: BoolProperty(name="Interpolate Along Boundary", default=False)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode in {"OBJECT", "EDIT_MESH"}

    def invoke(self, context, event):
        _load_settings(self, context)
        return self.execute(context)

    def execute(self, context):
        return _run_snap_seams(self, context, "SNAP")


class OBJECT_OT_transfer_seam_normals(bpy.types.Operator):
    bl_idname = "object.transfer_seam_normals"
    bl_label = "Transfer"
    bl_description = "Copy normals from the source onto the active mesh"
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(name="Distance", default=0.01, min=0.0, soft_max=1.0, precision=4, subtype="DISTANCE")
    do_both: BoolProperty(name="Snap and Transfer", default=True)
    mode: EnumProperty(name="Affect", items=MODE_ITEMS, default="SELECTED")
    vertex_group: StringProperty(name="Group")
    interpolate: BoolProperty(name="Interpolate Along Boundary", default=False)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            return False
        settings = getattr(context.scene, "work_tools_snap_vertices", None)
        if settings is not None and settings.both:
            return context.mode in {"OBJECT", "EDIT_MESH"}
        return context.mode == "OBJECT"

    def invoke(self, context, event):
        _load_settings(self, context)
        return self.execute(context)

    def execute(self, context):
        return _run_snap_seams(self, context, "TRANSFER")


class VIEW3D_PT_snap_seams(bpy.types.Panel):
    bl_label = "Snap Seams"
    bl_idname = "VIEW3D_PT_snap_seams"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Work Tools"

    def draw(self, context):
        layout = self.layout
        overlay = context.space_data.overlay
        row = layout.row(align=True)
        row.operator(
            OBJECT_OT_toggle_split_normals.bl_idname,
            icon="NORMALS_VERTEX_FACE",
            depress=overlay.show_split_normals,
        )
        if overlay.use_normals_constant_screen_size:
            row.prop(overlay, "normals_constant_screen_size", text="Length")
        else:
            row.prop(overlay, "normals_length", text="Length")

        if context.mode not in {"OBJECT", "EDIT_MESH"}:
            layout.label(text="Object Mode or Edit Mode is required")
            return
        settings = context.scene.work_tools_snap_vertices
        target = context.active_object
        col = layout.column(align=True)
        if target is not None and target.type == "MESH":
            col.label(text=f"Target: {target.name}")
        else:
            col.label(text="Target is the active mesh")
        try:
            source = _resolve_source(context, target)
        except SnapVertexError:
            source = None
        if source is not None:
            col.label(text=f"Source: {source.name}")
        else:
            col.label(text="Select one other mesh as the source")
        col.prop(settings, "distance")
        col.prop(settings, "mode")
        if settings.mode == "GROUP" and target is not None and target.type == "MESH":
            col.prop_search(settings, "vertex_group", target, "vertex_groups", text="Group")
        col.prop(settings, "interpolate")
        col.separator()
        col.prop(settings, "both")
        col.operator(OBJECT_OT_snap_vertices.bl_idname, icon="SNAP_VERTEX")
        col.operator(OBJECT_OT_transfer_seam_normals.bl_idname, icon="NORMALS_VERTEX")


def draw_object_menu(layout):
    layout.operator(OBJECT_OT_snap_vertices.bl_idname, icon="SNAP_VERTEX")
    layout.operator(OBJECT_OT_transfer_seam_normals.bl_idname, icon="NORMALS_VERTEX")


_CLASSES = (
    SnapVertexSettings,
    OBJECT_OT_snap_vertices,
    OBJECT_OT_transfer_seam_normals,
    VIEW3D_PT_snap_seams,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.work_tools_snap_vertices = PointerProperty(type=SnapVertexSettings)


def unregister():
    del bpy.types.Scene.work_tools_snap_vertices
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
