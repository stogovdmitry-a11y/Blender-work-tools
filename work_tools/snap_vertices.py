"""Snap selected vertices of the active mesh onto the nearest source vertices.

The objects stay separate. Only selected target vertices move, and only when
a source vertex is within the distance.
"""

import bpy
import bmesh
import numpy as np
from bpy.props import FloatProperty, PointerProperty, StringProperty
from mathutils.kdtree import KDTree

from .seam_normals import TransferError, transfer_seam_normals


class SnapVertexError(Exception):
    """User-facing snap failure. The message is shown in the status bar."""


def _poll_mesh_object(self, obj):
    return obj.type == "MESH"


def _resolve_source(context, source_name, target):
    if source_name:
        source = bpy.data.objects.get(source_name)
        if source is None or source.type != "MESH":
            raise SnapVertexError("Source not found")
        return source
    others = [obj for obj in context.selected_objects if obj.type == "MESH" and obj != target]
    if len(others) == 1:
        return others[0]
    raise SnapVertexError("Set a source in the panel, or select exactly two meshes")


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
    source: PointerProperty(
        name="Source",
        description="Mesh whose vertices are the snap targets. If empty, the other selected mesh is used",
        type=bpy.types.Object,
        poll=_poll_mesh_object,
    )
    distance: FloatProperty(
        name="Distance",
        description="Maximum world distance. Selected vertices farther than this stay where they are",
        default=0.01,
        min=0.0,
        soft_max=1.0,
        precision=4,
        subtype="DISTANCE",
    )


class OBJECT_OT_snap_vertices(bpy.types.Operator):
    bl_idname = "object.snap_vertices"
    bl_label = "Snap Vertices"
    bl_description = "Move selected vertices of the active mesh onto the nearest vertices of the source"
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(name="Distance", default=0.01, min=0.0, soft_max=1.0, precision=4, subtype="DISTANCE")
    source_name: StringProperty(options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode in {"OBJECT", "EDIT_MESH"}

    def invoke(self, context, event):
        settings = context.scene.work_tools_snap_vertices
        self.distance = settings.distance
        self.source_name = settings.source.name if settings.source else ""
        return self.execute(context)

    def execute(self, context):
        target = context.active_object
        try:
            source = _resolve_source(context, self.source_name, target)
            snapped, missed = snap_vertices(source, target, self.distance)
        except SnapVertexError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        if snapped == 0:
            self.report({"WARNING"}, "No vertex fell within the distance")
            return {"CANCELLED"}
        message = f"Snapped vertices: {snapped} of {snapped + missed}"
        self.report({"WARNING"} if missed else {"INFO"}, message)
        return {"FINISHED"}


class OBJECT_OT_snap_vertices_and_normals(bpy.types.Operator):
    bl_idname = "object.snap_vertices_and_normals"
    bl_label = "Snap and Transfer Normals"
    bl_description = "Snap the selected vertices, then copy normals from the source onto those same vertices"
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(name="Distance", default=0.01, min=0.0, soft_max=1.0, precision=4, subtype="DISTANCE")
    source_name: StringProperty(options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode in {"OBJECT", "EDIT_MESH"}

    def invoke(self, context, event):
        settings = context.scene.work_tools_snap_vertices
        self.distance = settings.distance
        self.source_name = settings.source.name if settings.source else ""
        return self.execute(context)

    def execute(self, context):
        target = context.active_object
        previous_mode = target.mode
        if previous_mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            source = _resolve_source(context, self.source_name, target)
            snapped, snap_missed = snap_vertices(source, target, self.distance)
            context.view_layer.update()
            try:
                result = transfer_seam_normals(
                    context,
                    source,
                    target,
                    distance=self.distance,
                    mode="SELECTED",
                    interpolate=False,
                )
            except TransferError as exc:
                self.report({"WARNING"}, f"Snapped vertices: {snapped} of {snapped + snap_missed}. {exc}")
                return {"FINISHED"}
        except (SnapVertexError, TransferError) as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        finally:
            if previous_mode == "EDIT" and context.active_object is not None and context.active_object.mode != "EDIT":
                bpy.ops.object.mode_set(mode="EDIT")
        if snapped == 0 and result.matched == 0:
            self.report({"WARNING"}, "No vertex fell within the distance")
            return {"CANCELLED"}
        message = (
            f"Snapped vertices: {snapped} of {snapped + snap_missed}. "
            f"Transferred normals: {result.matched} of {result.matched + result.missed}"
        )
        self.report({"WARNING"} if snap_missed or result.missed or result.notes else {"INFO"}, message)
        return {"FINISHED"}


class VIEW3D_PT_snap_vertices(bpy.types.Panel):
    bl_label = "Snap Vertices"
    bl_idname = "VIEW3D_PT_snap_vertices"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Work Tools"

    def draw(self, context):
        layout = self.layout
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
        col.prop(settings, "source")
        col.prop(settings, "distance")
        col.separator()
        col.operator(OBJECT_OT_snap_vertices.bl_idname, icon="SNAP_VERTEX")
        col.operator(OBJECT_OT_snap_vertices_and_normals.bl_idname, icon="NORMALS_VERTEX")


def draw_object_menu(layout):
    layout.operator(OBJECT_OT_snap_vertices.bl_idname, icon="SNAP_VERTEX")
    layout.operator(OBJECT_OT_snap_vertices_and_normals.bl_idname, icon="NORMALS_VERTEX")


_CLASSES = (
    SnapVertexSettings,
    OBJECT_OT_snap_vertices,
    OBJECT_OT_snap_vertices_and_normals,
    VIEW3D_PT_snap_vertices,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.work_tools_snap_vertices = PointerProperty(type=SnapVertexSettings)


def unregister():
    del bpy.types.Scene.work_tools_snap_vertices
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
