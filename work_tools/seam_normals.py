"""Перенос нормалей шва с одного меша на другой по близости.

Object Mode. Активный меш — цель. Источник выбирается в панели
или берётся как второй выделенный меш.

Выделение вершин — то, что осталось после последнего Edit Mode.
В Object Mode оно сохраняется.
"""

import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    PointerProperty,
    StringProperty,
)
from mathutils.kdtree import KDTree

MODE_ITEMS = (
    ("SELECTED", "Выделенные", "Вершины, выделенные в последнем Edit Mode"),
    ("BOUNDARY", "Граница", "Вершины на открытых рёбрах, если рядом есть источник"),
    ("GROUP", "Группа", "Вершины с ненулевым весом в группе"),
)

_EXACT = 1e-8


class TransferError(Exception):
    """User-facing transfer failure. The message is shown in the status bar."""


class TransferResult:
    def __init__(self, matched, missed, notes):
        self.matched = matched
        self.missed = missed
        self.notes = tuple(notes)


def _foreach_int(collection, name, count):
    values = np.empty(count, dtype=np.int32)
    collection.foreach_get(name, values)
    return values


def _foreach_vec(collection, name, count):
    values = np.empty(count * 3, dtype=np.float64)
    collection.foreach_get(name, values)
    return values.reshape(-1, 3)


def _normalize_rows(vectors):
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    out = np.zeros_like(vectors)
    np.divide(vectors, lengths, out=out, where=lengths > _EXACT)
    return out


def _linear_matrix(obj):
    matrix = np.array(obj.matrix_world.to_3x3(), dtype=np.float64)
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise TransferError(f"У «{obj.name}» нулевой масштаб, нормали не посчитать")
    return matrix


def _world_positions(obj, mesh):
    count = len(mesh.vertices)
    if count == 0:
        return np.zeros((0, 3), dtype=np.float64)
    local = _foreach_vec(mesh.vertices, "co", count)
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def _corner_normals_world(obj, mesh):
    loop_count = len(mesh.loops)
    if loop_count == 0 or len(mesh.corner_normals) != loop_count:
        mesh.update()
    if loop_count == 0 or len(mesh.corner_normals) != loop_count:
        raise TransferError(f"У «{obj.name}» нет нормалей по углам граней")
    # Normals ignore translation and transform by the inverse transpose.
    local = _foreach_vec(mesh.corner_normals, "vector", loop_count)
    world = local @ np.linalg.inv(_linear_matrix(obj))
    return _normalize_rows(world)


def _vertex_normals(loop_verts, corner_world, vertex_count):
    accumulated = np.zeros((vertex_count, 3), dtype=np.float64)
    np.add.at(accumulated, loop_verts, corner_world)
    return _normalize_rows(accumulated)


def _boundary_edges(mesh):
    loop_count = len(mesh.loops)
    poly_count = len(mesh.polygons)
    if loop_count == 0 or poly_count == 0:
        return np.zeros((0, 2), dtype=np.int32)
    loop_verts = _foreach_int(mesh.loops, "vertex_index", loop_count)
    starts = _foreach_int(mesh.polygons, "loop_start", poly_count)
    totals = _foreach_int(mesh.polygons, "loop_total", poly_count)
    counts = {}
    for start, total in zip(starts.tolist(), totals.tolist()):
        total = int(total)
        if total < 2:
            continue
        for offset in range(total):
            v0 = int(loop_verts[start + offset])
            v1 = int(loop_verts[start + (offset + 1) % total])
            if v0 > v1:
                v0, v1 = v1, v0
            key = (v0, v1)
            counts[key] = counts.get(key, 0) + 1
    edges = [key for key, count in counts.items() if count == 1]
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(edges, dtype=np.int32)


def _target_mask(target, mesh, mode, vertex_group):
    vertex_count = len(mesh.vertices)
    if vertex_count == 0:
        raise TransferError("У цели нет вершин")
    if mode == "SELECTED":
        selected = np.empty(vertex_count, dtype=np.bool_)
        mesh.vertices.foreach_get("select", selected)
        if not selected.any():
            raise TransferError("Нет выделенных вершин. Выдели шов в Edit Mode и вернись в Object Mode")
        return selected
    if mode == "BOUNDARY":
        edges = _boundary_edges(mesh)
        if len(edges) == 0:
            raise TransferError("На цели нет открытых рёбер")
        mask = np.zeros(vertex_count, dtype=bool)
        mask[edges[:, 0]] = True
        mask[edges[:, 1]] = True
        return mask
    if mode == "GROUP":
        group = target.vertex_groups.get(vertex_group) if vertex_group else None
        if group is None:
            raise TransferError("Укажи группу вершин на цели")
        mask = np.zeros(vertex_count, dtype=bool)
        group_index = group.index
        for vertex in mesh.vertices:
            for assignment in vertex.groups:
                if assignment.group == group_index and assignment.weight > 0.0:
                    mask[vertex.index] = True
                    break
        if not mask.any():
            raise TransferError("В группе нет вершин с весом")
        return mask
    raise TransferError(f"Неизвестный режим: {mode}")


def _match_nearest(source_pos, source_normals, source_valid, target_pos, target_index, distance):
    valid_index = np.flatnonzero(source_valid)
    tree = KDTree(len(valid_index))
    for vertex_index in valid_index.tolist():
        tree.insert(source_pos[vertex_index].tolist(), vertex_index)
    tree.balance()

    limit = max(float(distance), _EXACT)
    vertex_count = target_pos.shape[0]
    world = np.zeros((vertex_count, 3), dtype=np.float64)
    mask = np.zeros(vertex_count, dtype=bool)
    missed = 0
    for vertex_index in target_index.tolist():
        _co, found, dist = tree.find(target_pos[vertex_index].tolist())
        if found is None or dist > limit:
            missed += 1
            continue
        world[vertex_index] = source_normals[found]
        mask[vertex_index] = True
    return mask, world, missed


def _closest_on_segments(points, start, end):
    segment = end - start
    length2 = np.maximum(np.einsum("ij,ij->i", segment, segment), 1e-20)
    count = points.shape[0]
    edge_count = start.shape[0]
    best_dist = np.full(count, np.inf, dtype=np.float64)
    best_factor = np.zeros(count, dtype=np.float64)
    best_edge = np.full(count, -1, dtype=np.int32)
    bytes_per_point = max(edge_count, 1) * 3 * 8 * 3
    batch = max(1, min(256, int(64e6 / bytes_per_point)))
    for offset in range(0, count, batch):
        chunk = points[offset:offset + batch]
        factor = np.einsum("bmj,mj->bm", chunk[:, None, :] - start, segment) / length2
        np.clip(factor, 0.0, 1.0, out=factor)
        closest = start + factor[..., None] * segment
        dist2 = np.einsum("bmj,bmj->bm", chunk[:, None, :] - closest, chunk[:, None, :] - closest)
        choice = np.argmin(dist2, axis=1)
        rows = np.arange(chunk.shape[0])
        slot = slice(offset, offset + chunk.shape[0])
        best_dist[slot] = np.sqrt(dist2[rows, choice])
        best_factor[slot] = factor[rows, choice]
        best_edge[slot] = choice
    return best_edge, best_factor, best_dist


def _match_boundary(source_pos, source_normals, source_valid, edges, target_pos, target_index, distance):
    if len(edges) == 0:
        return None
    usable = source_valid[edges[:, 0]] & source_valid[edges[:, 1]]
    edges = edges[usable]
    if len(edges) == 0:
        return None

    points = target_pos[target_index]
    edge_index, factor, dist = _closest_on_segments(points, source_pos[edges[:, 0]], source_pos[edges[:, 1]])
    limit = max(float(distance), _EXACT)
    vertex_count = target_pos.shape[0]
    world = np.zeros((vertex_count, 3), dtype=np.float64)
    mask = np.zeros(vertex_count, dtype=bool)
    missed = 0
    start_normal = source_normals[edges[:, 0]]
    end_normal = source_normals[edges[:, 1]]
    for row, vertex_index in enumerate(target_index.tolist()):
        if dist[row] > limit or edge_index[row] < 0:
            missed += 1
            continue
        blend = (1.0 - factor[row]) * start_normal[edge_index[row]] + factor[row] * end_normal[edge_index[row]]
        length = np.linalg.norm(blend)
        if length <= _EXACT:
            missed += 1
            continue
        world[vertex_index] = blend / length
        mask[vertex_index] = True
    return mask, world, missed


def _read_source(context, source):
    context.view_layer.update()
    depsgraph = context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        loop_verts = _foreach_int(mesh.loops, "vertex_index", len(mesh.loops)) if mesh.loops else np.zeros(0, dtype=np.int32)
        corner_world = _corner_normals_world(evaluated, mesh)
        vertex_count = len(mesh.vertices)
        normals = _vertex_normals(loop_verts, corner_world, vertex_count)
        valid = np.linalg.norm(normals, axis=1) > _EXACT
        if not valid.any():
            raise TransferError("У источника нет нормалей")
        return {
            "positions": _world_positions(evaluated, mesh),
            "normals": normals,
            "valid": valid,
            "boundary": _boundary_edges(mesh),
        }
    finally:
        evaluated.to_mesh_clear()


def _world_to_local(obj, world_normals):
    # Inverse of the inverse-transpose map: local = M^T * world, then normalize.
    return _normalize_rows(world_normals @ _linear_matrix(obj))


def _apply_normals(mesh, local_normals, write_mask):
    loop_count = len(mesh.loops)
    if loop_count == 0:
        raise TransferError("У цели нет граней")
    # A zero vector tells Blender to keep the computed normal for that corner.
    # Existing custom normals are real vectors, so they have to be written back.
    if mesh.has_custom_normals and len(mesh.corner_normals) == loop_count:
        normals = _foreach_vec(mesh.corner_normals, "vector", loop_count).copy()
    else:
        normals = np.zeros((loop_count, 3), dtype=np.float64)
    loop_verts = _foreach_int(mesh.loops, "vertex_index", loop_count)
    hit = write_mask[loop_verts]
    normals[hit] = local_normals[loop_verts[hit]]
    try:
        mesh.normals_split_custom_set(normals.tolist())
    except RuntimeError as exc:
        raise TransferError(f"Не удалось записать нормали: {exc}") from exc


def transfer_seam_normals(context, source, target, *, distance, mode, vertex_group="", interpolate=False):
    if source is None or target is None:
        raise TransferError("Нужны два меша")
    if source.type != "MESH" or target.type != "MESH":
        raise TransferError("Нужны два меша")
    if source == target:
        raise TransferError("Источник и цель — один и тот же объект")
    if source.mode != "OBJECT" or target.mode != "OBJECT":
        raise TransferError("Оба объекта должны быть в объектном режиме")

    mesh = target.data
    mask = _target_mask(target, mesh, mode, vertex_group)
    if len(mesh.loops) == 0:
        raise TransferError("У цели нет граней")

    notes = []
    if mode == "SELECTED" and bool(mask.all()):
        notes.append("выделены все вершины, проверь что это шов, а не весь меш")

    source_data = _read_source(context, source)
    target_pos = _world_positions(target, mesh)
    target_index = np.flatnonzero(mask)

    if interpolate:
        matched = _match_boundary(
            source_data["positions"],
            source_data["normals"],
            source_data["valid"],
            source_data["boundary"],
            target_pos,
            target_index,
            distance,
        )
        if matched is None:
            notes.append("у источника нет открытых рёбер, взяты ближайшие вершины")
            matched = _match_nearest(
                source_data["positions"],
                source_data["normals"],
                source_data["valid"],
                target_pos,
                target_index,
                distance,
            )
    else:
        matched = _match_nearest(
            source_data["positions"],
            source_data["normals"],
            source_data["valid"],
            target_pos,
            target_index,
            distance,
        )

    write_mask, world_normals, missed = matched
    if not write_mask.any():
        return TransferResult(0, missed, notes)

    local_normals = np.zeros_like(world_normals)
    local_normals[write_mask] = _world_to_local(target, world_normals[write_mask])
    _apply_normals(mesh, local_normals, write_mask)
    return TransferResult(int(write_mask.sum()), missed, notes)


def _poll_mesh_object(self, obj):
    return obj.type == "MESH"


def _resolve_source(context, source_name, target):
    if source_name:
        source = bpy.data.objects.get(source_name)
        if source is None or source.type != "MESH":
            raise TransferError("Источник не найден")
        return source
    others = [obj for obj in context.selected_objects if obj.type == "MESH" and obj != target]
    if len(others) == 1:
        return others[0]
    raise TransferError("Укажи источник в панели или выдели ровно два меша")


class TransferSeamNormalsSettings(bpy.types.PropertyGroup):
    source: PointerProperty(
        name="Источник",
        description="Меш, с которого копируются нормали. Если пусто — берётся второй выделенный меш",
        type=bpy.types.Object,
        poll=_poll_mesh_object,
    )
    distance: FloatProperty(
        name="Дистанция",
        description="Максимальное расстояние в мире между вершиной цели и парой на источнике",
        default=0.001,
        min=0.0,
        soft_max=1.0,
        precision=4,
        subtype="DISTANCE",
    )
    mode: EnumProperty(
        name="Область",
        description="Какие вершины цели менять",
        items=MODE_ITEMS,
        default="SELECTED",
    )
    vertex_group: StringProperty(
        name="Группа",
        description="Группа вершин на цели",
    )
    interpolate: BoolProperty(
        name="Интерполяция по ребру границы",
        description="Брать нормаль с ближайшего открытого ребра источника. Нужно, если на новом шве вершин больше",
        default=False,
    )


class OBJECT_OT_transfer_seam_normals(bpy.types.Operator):
    bl_idname = "object.transfer_seam_normals"
    bl_label = "Перенести нормали шва"
    bl_description = "Копирует нормали с источника на вершины шва активного меша"
    bl_options = {"REGISTER", "UNDO"}

    distance: FloatProperty(
        name="Дистанция",
        default=0.001,
        min=0.0,
        soft_max=1.0,
        precision=4,
        subtype="DISTANCE",
    )
    mode: EnumProperty(name="Область", items=MODE_ITEMS, default="SELECTED")
    vertex_group: StringProperty(name="Группа")
    interpolate: BoolProperty(name="Интерполяция по ребру границы", default=False)
    source_name: StringProperty(options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode == "OBJECT"

    def invoke(self, context, event):
        settings = context.scene.work_tools_seam_normals
        self.distance = settings.distance
        self.mode = settings.mode
        self.vertex_group = settings.vertex_group
        self.interpolate = settings.interpolate
        self.source_name = settings.source.name if settings.source else ""
        return self.execute(context)

    def execute(self, context):
        target = context.active_object
        try:
            source = _resolve_source(context, self.source_name, target)
            result = transfer_seam_normals(
                context,
                source,
                target,
                distance=self.distance,
                mode=self.mode,
                vertex_group=self.vertex_group,
                interpolate=self.interpolate,
            )
        except TransferError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}

        if result.matched == 0:
            message = "Ни одна вершина не попала в дистанцию"
            if result.notes:
                message = f"{message}. {'. '.join(result.notes)}"
            self.report({"WARNING"}, message)
            return {"CANCELLED"}

        message = f"Перенесено вершин: {result.matched} из {result.matched + result.missed}"
        if result.notes:
            message = f"{message}. {'. '.join(result.notes)}"
        level = "WARNING" if result.missed or result.notes else "INFO"
        self.report({level}, message)
        return {"FINISHED"}


class VIEW3D_PT_transfer_seam_normals(bpy.types.Panel):
    bl_label = "Нормали шва"
    bl_idname = "VIEW3D_PT_transfer_seam_normals"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Work Tools"

    def draw(self, context):
        layout = self.layout
        if context.mode != "OBJECT":
            layout.label(text="Нужен объектный режим")
            return

        settings = context.scene.work_tools_seam_normals
        target = context.active_object
        col = layout.column(align=True)
        if target is not None and target.type == "MESH":
            col.label(text=f"Цель: {target.name}")
        else:
            col.label(text="Цель — активный меш")
        col.prop(settings, "source")
        col.prop(settings, "distance")
        col.prop(settings, "mode")
        if settings.mode == "GROUP" and target is not None and target.type == "MESH":
            col.prop_search(settings, "vertex_group", target, "vertex_groups", text="Группа")
        col.prop(settings, "interpolate")
        col.separator()
        col.operator(OBJECT_OT_transfer_seam_normals.bl_idname, icon="NORMALS_VERTEX")


def draw_object_menu(layout):
    layout.operator(OBJECT_OT_transfer_seam_normals.bl_idname, icon="NORMALS_VERTEX")


_CLASSES = (
    TransferSeamNormalsSettings,
    OBJECT_OT_transfer_seam_normals,
    VIEW3D_PT_transfer_seam_normals,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.work_tools_seam_normals = PointerProperty(type=TransferSeamNormalsSettings)


def unregister():
    del bpy.types.Scene.work_tools_seam_normals
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
