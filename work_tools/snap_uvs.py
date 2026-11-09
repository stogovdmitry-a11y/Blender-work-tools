"""Snap UV points on the active mesh to the nearest UV points of a source.

Only points inside the distance move. The rest of the UV map stays put.
The active mesh is the target. The other selected mesh is the source.
"""

import bpy
import bmesh
import numpy as np
from bpy.props import BoolProperty, FloatProperty, PointerProperty
from mathutils.kdtree import KDTree


class SnapUVError(Exception):
    """User-facing snap failure. The message is shown in the status bar."""


def _resolve_source(context, target):
    selected = list(context.selected_objects)
    if target is None or target.type != "MESH" or target not in selected or len(selected) != 2:
        raise SnapUVError("Select two meshes. The active one is the target")
    source = selected[0] if selected[1] == target else selected[1]
    if source.type != "MESH" or source == target:
        raise SnapUVError("Select two meshes. The active one is the target")
    return source


def _read_uvs(obj):
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        layer = bm.loops.layers.uv.active
        if layer is None:
            raise SnapUVError(f"{obj.name} has no UV map")
        loops = [loop for face in bm.faces for loop in face.loops]
        if not loops:
            raise SnapUVError(f"{obj.name} has no UV points")
        uvs = np.array([loop[layer].uv[:] for loop in loops], dtype=np.float64)
        selected = np.array([bool(loop[layer].select) for loop in loops], dtype=bool)
        return {
            "kind": "BMESH",
            "bm": bm,
            "mesh": obj.data,
            "layer": layer,
            "loops": loops,
            "uvs": uvs,
            "selected": selected,
        }

    layer = obj.data.uv_layers.active
    if layer is None:
        raise SnapUVError(f"{obj.name} has no UV map")
    count = len(layer.data)
    if count == 0:
        raise SnapUVError(f"{obj.name} has no UV points")
    uvs = np.empty(count * 2, dtype=np.float32)
    layer.data.foreach_get("uv", uvs)
    selected = np.empty(count, dtype=np.bool_)
    layer.data.foreach_get("select", selected)
    return {
        "kind": "MESH",
        "mesh": obj.data,
        "layer": layer,
        "uvs": uvs.reshape(-1, 2).astype(np.float64),
        "selected": selected,
    }


def _write_uvs(access, uvs):
    if access["kind"] == "BMESH":
        layer = access["layer"]
        for loop, uv in zip(access["loops"], uvs):
            loop[layer].uv = (float(uv[0]), float(uv[1]))
        bmesh.update_edit_mesh(access["mesh"])
        return
    flat = np.asarray(uvs, dtype=np.float32).reshape(-1)
    access["layer"].data.foreach_set("uv", flat)
    access["mesh"].update()


def snap_uv_points(source, target, distance, selected_only):
    if source is None or target is None:
        raise SnapUVError("Two meshes are required")
    if source.type != "MESH" or target.type != "MESH":
        raise SnapUVError("Two meshes are required")
    if source == target:
        raise SnapUVError("The source and the target are the same object")

    source_uvs = _read_uvs(source)["uvs"]
    target_access = _read_uvs(target)
    uvs = target_access["uvs"]
    if selected_only:
        mask = target_access["selected"]
        if not mask.any():
            raise SnapUVError("No UV points are selected. Select them, or turn off Selected Only")
    else:
        mask = np.ones(len(uvs), dtype=bool)

    tree = KDTree(len(source_uvs))
    for index, uv in enumerate(source_uvs):
        tree.insert((float(uv[0]), float(uv[1]), 0.0), index)
    tree.balance()

    limit = max(float(distance), 0.0)
    snapped = 0
    considered = int(mask.sum())
    for index in np.flatnonzero(mask).tolist():
        uv = uvs[index]
        _co, _source_index, dist = tree.find((float(uv[0]), float(uv[1]), 0.0))
        if dist <= limit:
            uvs[index] = source_uvs[_source_index]
            snapped += 1
    if snapped:
        _write_uvs(target_access, uvs)
    return snapped, considered - snapped


def _draw_snap_uvs(layout, context):
    settings = context.scene.work_tools_snap_uvs
    target = context.active_object
    col = layout.column(align=True)
    if target is not None and target.type == "MESH":
        col.label(text=f"Target: {target.name}")
    else:
        col.label(text="Target is the active mesh")
    try:
        source = _resolve_source(context, target)
    except SnapUVError:
        source = None
    if source is not None:
        col.label(text=f"Source: {source.name}")
    else:
        col.label(text="Select one other mesh as the source")
    col.prop(settings, "distance")
    col.prop(settings, "selected_only")
    col.separator()
    col.operator(UV_OT_snap_uv_points.bl_idname, icon="UV")


class SnapUVSettings(bpy.types.PropertyGroup):
    distance: FloatProperty(
        name="Distance",
        description="Maximum distance in UV space. Points farther than this stay where they are",
        default=0.001,
        min=0.0,
        soft_max=0.1,
        precision=4,
    )
    selected_only: BoolProperty(
        name="Selected Only",
        description="Only snap selected UV points. The distance still applies",
        default=False,
    )


class UV_OT_snap_uv_points(bpy.types.Operator):
    bl_idname = "uv.snap_uv_points"
    bl_label = "Snap UV Points"
    bl_description = "Move nearby UV points of the active mesh onto the nearest UV points of the source"
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(name="Distance", default=0.001, min=0.0, soft_max=0.1, precision=4)
    selected_only: BoolProperty(name="Selected Only", default=False)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode in {"OBJECT", "EDIT_MESH"}

    def invoke(self, context, event):
        settings = context.scene.work_tools_snap_uvs
        self.distance = settings.distance
        self.selected_only = settings.selected_only
        return self.execute(context)

    def execute(self, context):
        target = context.active_object
        try:
            source = _resolve_source(context, target)
            snapped, missed = snap_uv_points(source, target, self.distance, self.selected_only)
        except SnapUVError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        if snapped == 0:
            self.report({"WARNING"}, "No UV point fell within the distance")
            return {"CANCELLED"}
        message = f"Snapped UV points: {snapped} of {snapped + missed}"
        self.report({"WARNING"} if missed else {"INFO"}, message)
        return {"FINISHED"}


class VIEW3D_PT_snap_uvs(bpy.types.Panel):
    bl_label = "Snap UVs"
    bl_idname = "VIEW3D_PT_snap_uvs"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Work Tools"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        if context.mode not in {"OBJECT", "EDIT_MESH"}:
            layout.label(text="Object Mode or Edit Mode is required")
            return
        _draw_snap_uvs(layout, context)


class IMAGE_PT_snap_uvs(bpy.types.Panel):
    bl_label = "Snap UVs"
    bl_idname = "IMAGE_PT_snap_uvs"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Work Tools"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space is not None and getattr(space, "mode", "") == "UV"

    def draw(self, context):
        _draw_snap_uvs(self.layout, context)


def draw_object_menu(layout):
    layout.operator(UV_OT_snap_uv_points.bl_idname, icon="UV")


def _draw_uv_menu(self, _context):
    self.layout.operator(UV_OT_snap_uv_points.bl_idname, icon="UV")


_CLASSES = (
    SnapUVSettings,
    UV_OT_snap_uv_points,
    VIEW3D_PT_snap_uvs,
    IMAGE_PT_snap_uvs,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.work_tools_snap_uvs = PointerProperty(type=SnapUVSettings)
    bpy.types.IMAGE_MT_uvs.append(_draw_uv_menu)


def unregister():
    bpy.types.IMAGE_MT_uvs.remove(_draw_uv_menu)
    del bpy.types.Scene.work_tools_snap_uvs
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
