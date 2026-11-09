"""Send selected meshes to an open Maya scene.

Maya must be listening on a Python command port. The mesh is exported as FBX
with edge smoothing and custom normals. Maya imports the normals locked, so
the original values can be read, then unlocks them and writes those values
back. Sharp edges from Blender are applied before the unlock, so the mesh
does not come in faceted and the normals stay editable.
"""

import json
import os
import socket
import tempfile
import threading
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, IntProperty, PointerProperty

from .check_mesh import _report_lines, _select_problems, check_mesh

PORT_COMMAND = (
    "import maya.cmds as cmds\n"
    "cmds.commandPort(name=\"127.0.0.1:4434\", sourceType=\"python\")"
)


class SendToMayaError(Exception):
    """User-facing send failure. The message is shown in the status bar."""


def selected_meshes(context):
    return [obj for obj in context.selected_objects if obj.type == "MESH"]


def _sharp_edge_pairs(mesh):
    edge_count = len(mesh.edges)
    if edge_count == 0:
        return []
    vertices = np.empty(edge_count * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", vertices)
    vertices = vertices.reshape(-1, 2)

    sharp = np.zeros(edge_count, dtype=bool)
    sharp_attr = mesh.attributes.get("sharp_edge")
    if sharp_attr is not None:
        sharp_attr.data.foreach_get("value", sharp)

    face_count = len(mesh.polygons)
    if face_count and len(mesh.loops):
        loop_count = len(mesh.loops)
        loop_edges = np.empty(loop_count, dtype=np.int32)
        mesh.loops.foreach_get("edge_index", loop_edges)
        sharp[np.bincount(loop_edges, minlength=edge_count) > 2] = True

        sharp_face = np.zeros(face_count, dtype=bool)
        face_attr = mesh.attributes.get("sharp_face")
        if face_attr is not None:
            face_attr.data.foreach_get("value", sharp_face)
        if sharp_face.any():
            starts = np.empty(face_count, dtype=np.int32)
            totals = np.empty(face_count, dtype=np.int32)
            mesh.polygons.foreach_get("loop_start", starts)
            mesh.polygons.foreach_get("loop_total", totals)
            for face in np.flatnonzero(sharp_face):
                start = int(starts[face])
                sharp[loop_edges[start:start + int(totals[face])]] = True

    pairs = np.sort(vertices[sharp], axis=1)
    return pairs.tolist()


def mesh_payload(obj, depsgraph, apply_modifiers):
    if apply_modifiers:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            return {
                "name": obj.name,
                "verts": len(mesh.vertices),
                "sharp": _sharp_edge_pairs(mesh),
            }
        finally:
            evaluated.to_mesh_clear()
    return {
        "name": obj.name,
        "verts": len(obj.data.vertices),
        "sharp": _sharp_edge_pairs(obj.data),
    }


def ensure_fbx_exporter():
    if hasattr(bpy.ops.export_scene, "fbx"):
        return
    import addon_utils
    addon_utils.enable("io_scene_fbx")
    if not hasattr(bpy.ops.export_scene, "fbx"):
        raise SendToMayaError("The FBX exporter is not available")


def _fbx_global_scale(scene):
    """Cancel the exporter's meter-to-centimeter scale.

    FBX_SCALE_NONE multiplies the scene by 100, so a Blender mesh arrives in
    Maya a hundred times bigger. Dividing it back out keeps Blender numbers.
    """
    ensure_fbx_exporter()
    from io_scene_fbx.fbx_utils import units_blender_to_fbx_factor
    factor = units_blender_to_fbx_factor(scene)
    if not factor:
        return 1.0
    return 1.0 / factor


def _delete_objects(objects):
    for obj in objects:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _copies_with_applied_transforms(context, meshes, apply_modifiers):
    """Duplicates used only for export. The source objects are not edited."""
    depsgraph = context.evaluated_depsgraph_get()
    copies = []
    try:
        for obj in meshes:
            if apply_modifiers:
                mesh = bpy.data.meshes.new_from_object(
                    obj,
                    preserve_all_data_layers=True,
                    depsgraph=depsgraph,
                )
            else:
                mesh = obj.data.copy()
            copy = bpy.data.objects.new(obj.name, mesh)
            context.scene.collection.objects.link(copy)
            copy.matrix_world = obj.matrix_world.copy()
            copies.append(copy)
        view_layer = context.view_layer
        for obj in view_layer.objects:
            obj.select_set(obj in copies)
        view_layer.objects.active = copies[0]
        with context.temp_override(
            active_object=copies[0],
            selected_objects=list(copies),
            selected_editable_objects=list(copies),
            object=copies[0],
        ):
            result = bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if "FINISHED" not in result:
            raise SendToMayaError("Could not apply transforms for the Maya copy")
    except Exception:
        _delete_objects(copies)
        raise
    return copies


def _zero_world_locations(objects):
    saved = []
    for obj in objects:
        original = obj.matrix_world.copy()
        saved.append((obj, original))
        moved = original.copy()
        moved.translation = (0.0, 0.0, 0.0)
        obj.matrix_world = moved
    return saved


def _restore_matrices(saved):
    for obj, matrix in saved:
        obj.matrix_world = matrix


def _clear_materials(objects):
    saved = []
    seen = set()
    for obj in objects:
        mesh = obj.data
        if mesh is None or mesh in seen:
            continue
        seen.add(mesh)
        saved.append((mesh, list(mesh.materials)))
        mesh.materials.clear()
    return saved


def _restore_materials(saved):
    for mesh, materials in saved:
        mesh.materials.clear()
        for material in materials:
            mesh.materials.append(material)


def export_fbx(context, meshes, filepath, apply_modifiers, apply_transforms, zero_location, remove_materials):
    ensure_fbx_exporter()
    view_layer = context.view_layer
    previous = list(context.selected_objects)
    active = view_layer.objects.active
    copies = []
    saved_matrices = []
    saved_materials = []
    try:
        if zero_location:
            saved_matrices = _zero_world_locations(meshes)
            view_layer.update()
        targets = meshes
        modifiers = apply_modifiers
        if apply_transforms:
            copies = _copies_with_applied_transforms(context, meshes, apply_modifiers)
            targets = copies
            modifiers = False
        if remove_materials:
            if copies:
                for obj in copies:
                    if obj.data is not None:
                        obj.data.materials.clear()
            else:
                saved_materials = _clear_materials(targets)
        for obj in view_layer.objects:
            obj.select_set(obj in targets)
        view_layer.objects.active = targets[0]
        result = bpy.ops.export_scene.fbx(
            filepath=filepath,
            check_existing=False,
            use_selection=True,
            object_types={"MESH"},
            use_mesh_modifiers=modifiers,
            use_mesh_modifiers_render=False,
            mesh_smooth_type="EDGE",
            use_subsurf=False,
            use_triangles=False,
            use_mesh_edges=False,
            apply_unit_scale=True,
            apply_scale_options="FBX_SCALE_NONE",
            use_space_transform=True,
            bake_space_transform=True,
            axis_forward="-Z",
            axis_up="Y",
            global_scale=_fbx_global_scale(context.scene),
            bake_anim=False,
            add_leaf_bones=False,
            path_mode="AUTO",
            embed_textures=False,
            use_custom_props=False,
        )
    finally:
        for obj in context.view_layer.objects:
            obj.select_set(obj in previous)
        view_layer.objects.active = active
        _delete_objects(copies)
        if saved_materials:
            _restore_materials(saved_materials)
        if saved_matrices:
            _restore_matrices(saved_matrices)
            view_layer.update()
    if "FINISHED" not in result:
        raise SendToMayaError("FBX export failed")


MAYA_SCRIPT = r'''
import json
import os
import traceback

fbx_path = __FBX_PATH__
result_path = __RESULT_PATH__
payload = json.loads(__PAYLOAD__)
remove_materials = __REMOVE_MATERIALS__

def write_result(text):
    temporary = result_path + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, result_path)

def leaf_name(node):
    return node.split("|")[-1].split(":")[-1]

def material_snapshot():
    import maya.cmds as cmds
    return (
        set(cmds.ls(materials=True) or []),
        set(cmds.ls(type="shadingEngine") or []),
        set(cmds.ls(type=("file", "place2dTexture")) or []),
    )

def assign_shading_engine(shape, engine):
    import maya.cmds as cmds
    if not shape or not engine or not cmds.objExists(shape) or not cmds.objExists(engine):
        return
    # sets() prints an exclusivity error when a shape and its faces sit in
    # different shading groups. Rewire the shading plugs instead.
    plugs = cmds.listConnections(
        shape,
        type="shadingEngine",
        plugs=True,
        connections=True,
        source=False,
        destination=True,
    ) or []
    for src, dst in zip(plugs[0::2], plugs[1::2]):
        if dst.split(".")[0] == engine and ".objectGroups[" not in src:
            continue
        try:
            cmds.disconnectAttr(src, dst)
        except Exception:
            pass
    source_plug = shape + ".instObjGroups[0]"
    if engine not in (cmds.listConnections(source_plug, type="shadingEngine") or []):
        try:
            cmds.connectAttr(source_plug, engine + ".dagSetMembers", nextAvailable=True)
        except Exception:
            pass

def discard_imported_materials(snapshot, transforms):
    import maya.cmds as cmds
    old_materials, old_engines, old_files = snapshot
    for transform in transforms:
        for shape in cmds.listRelatives(transform, shapes=True, fullPath=True, type="mesh") or []:
            try:
                assign_shading_engine(shape, "initialShadingGroup")
            except Exception:
                pass
    for engine in cmds.ls(type="shadingEngine") or []:
        if engine not in old_engines and engine != "initialShadingGroup":
            try:
                cmds.delete(engine)
            except Exception:
                pass
    for material in cmds.ls(materials=True) or []:
        if material not in old_materials:
            try:
                cmds.delete(material)
            except Exception:
                pass
    for node in cmds.ls(type=("file", "place2dTexture")) or []:
        if node not in old_files:
            try:
                cmds.delete(node)
            except Exception:
                pass

def existing_transforms():
    found = {}
    import maya.cmds as cmds
    for node in cmds.ls(long=True, type="transform") or []:
        found.setdefault(leaf_name(node), []).append(node)
    return found

def is_imported_name(short, name):
    if short == name:
        return True
    if not short.startswith(name):
        return False
    tail = short[len(name):]
    return tail.isdigit() or tail.startswith("FBXASC")

def adopt_shape(shape, item, existing):
    """Put the imported shape on the transform that already uses this name."""
    import maya.cmds as cmds
    matches = [node for node in (existing.get(item["name"]) or []) if cmds.objExists(node)]
    new_parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [None])[0]
    if len(matches) != 1 or not new_parent or matches[0] == new_parent:
        return None
    old = matches[0]
    old_shapes = cmds.listRelatives(old, shapes=True, fullPath=True, noIntermediate=True) or []
    old_shapes = [node for node in old_shapes if cmds.nodeType(node) == "mesh"]
    engines = []
    for node in old_shapes:
        for engine in cmds.listConnections(node, type="shadingEngine") or []:
            if engine not in engines:
                engines.append(engine)
    custom = [engine for engine in engines if engine != "initialShadingGroup"]
    engines = custom or engines
    old_shape_name = old_shapes[0].split("|")[-1] if old_shapes else None
    world = cmds.xform(new_parent, query=True, matrix=True, worldSpace=True)
    cmds.xform(old, matrix=world, worldSpace=True)
    moved = cmds.parent(shape, old, shape=True, relative=True)[0]
    if old_shapes:
        cmds.delete([node for node in old_shapes if cmds.objExists(node)])
    if old_shape_name and cmds.objExists(moved):
        try:
            moved = cmds.rename(moved, old_shape_name)
        except Exception:
            pass
    for engine in engines:
        try:
            assign_shading_engine(moved, engine)
        except Exception:
            pass
    children = cmds.listRelatives(new_parent, children=True) or []
    if not children and cmds.objExists(new_parent):
        cmds.delete(new_parent)
    return old

def run():
    import maya.cmds as cmds
    import maya.mel as mel
    import maya.api.OpenMaya as om

    if not cmds.pluginInfo("fbxmaya", query=True, loaded=True):
        cmds.loadPlugin("fbxmaya")

    def mel_try(command):
        try:
            mel.eval(command)
        except Exception:
            pass

    already = existing_transforms()
    materials_before = material_snapshot() if remove_materials else None
    before = set(cmds.ls(long=True, type="mesh") or [])
    mel_try("FBXImportMode -v add")
    mel_try("FBXImportScaleFactor 1")
    mel_try("FBXImportUnlockNormals -v false")
    mel_try("FBXImportHardEdges -v true")
    mel_try("FBXImportSmoothingGroups -v true")
    mel_try("FBXImportSkins -v false")
    mel_try("FBXImportConstraints -v false")
    mel_try("FBXImportCameras -v false")
    mel_try("FBXImportLights -v false")
    mel.eval('FBXImport -f "%s"' % fbx_path.replace("\\", "/"))
    after = set(cmds.ls(long=True, type="mesh") or [])
    shapes = [shape for shape in after if shape not in before]
    if not shapes:
        raise RuntimeError("Maya imported the file but no new mesh appeared")

    def short_name(shape):
        parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [shape])[0]
        return parent.split("|")[-1].split(":")[-1]

    unused = list(shapes)
    notes = []
    restored = 0
    transforms = []
    fresh = []
    for item in payload:
        shape = None
        for candidate in unused:
            if is_imported_name(short_name(candidate), item["name"]):
                shape = candidate
                break
        if shape is None:
            for candidate in unused:
                selection = om.MSelectionList()
                selection.add(candidate)
                if om.MFnMesh(selection.getDagPath(0)).numVertices == item["verts"]:
                    shape = candidate
                    break
        if shape is None:
            notes.append("no Maya mesh for %s" % item["name"])
            continue
        unused.remove(shape)
        notes.extend(restore_mesh(shape, item))
        adopted = None
        if len(already.get(item["name"]) or []) > 1:
            notes.append("several objects named %s, imported a new one" % item["name"])
        else:
            try:
                adopted = adopt_shape(shape, item, already)
            except Exception:
                notes.append("could not update %s" % item["name"])
                adopted = None
        if adopted:
            transforms.append(adopted)
        else:
            parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [None])[0]
            if parent:
                short = parent.split("|")[-1].split(":")[-1]
                if short != item["name"]:
                    try:
                        parent = cmds.rename(parent, item["name"])
                    except Exception:
                        notes.append("%s kept its imported name" % short)
                transforms.append(parent)
                fresh.append(parent)
        restored += 1

    if remove_materials and materials_before is not None:
        discard_imported_materials(materials_before, fresh)
    if transforms:
        cmds.select(transforms, replace=True)
    message = "OK %d" % restored
    if notes:
        message += "\n" + "\n".join(notes)
    return message

def restore_mesh(shape, item):
    import maya.api.OpenMaya as om
    import maya.cmds as cmds

    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    notes = []
    saved = []
    for face in range(mesh.numPolygons):
        for vertex in mesh.getPolygonVertices(face):
            normal = mesh.getFaceVertexNormal(face, vertex, om.MSpace.kObject)
            saved.append((face, vertex, normal.x, normal.y, normal.z))

    faces = [entry[0] for entry in saved]
    vertices = [entry[1] for entry in saved]
    normals = [om.MVector(entry[2], entry[3], entry[4]) for entry in saved]
    # Unlocking clears the imported values, so they are written back afterwards.
    # Edge smoothing has to be last: unlocking marks every edge hard again.
    if faces:
        try:
            mesh.unlockFaceVertexNormals(faces, vertices)
        except Exception:
            cmds.polyNormalPerVertex(shape + ".vtxFace[*][*]", unFreezeNormal=True)
        mesh.setFaceVertexNormals(normals, faces, vertices, om.MSpace.kObject)

    if mesh.numVertices == item["verts"]:
        smooth = [True] * mesh.numEdges
        lookup = {}
        for edge in range(mesh.numEdges):
            a, b = mesh.getEdgeVertices(edge)
            lookup[(a, b) if a < b else (b, a)] = edge
        for pair in item["sharp"]:
            edge = lookup.get((pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0]))
            if edge is not None:
                smooth[edge] = False
        if mesh.numEdges:
            mesh.setEdgeSmoothings(list(range(mesh.numEdges)), smooth)
    else:
        notes.append("%s vertex count changed, sharp edges were left to the FBX importer" % item["name"])
    mesh.updateSurface()
    return notes

try:
    write_result(run())
except Exception:
    write_result("FAIL\n" + traceback.format_exc())
'''


def write_maya_script(fbx_path, payload, remove_materials):
    directory = tempfile.mkdtemp(prefix="work_tools_maya_")
    script_path = os.path.join(directory, "send.py")
    result_path = os.path.join(directory, "result.txt")
    script = (
        MAYA_SCRIPT
        .replace("__FBX_PATH__", json.dumps(fbx_path))
        .replace("__RESULT_PATH__", json.dumps(result_path))
        .replace("__PAYLOAD__", json.dumps(json.dumps(payload)))
        .replace("__REMOVE_MATERIALS__", "True" if remove_materials else "False")
    )
    with open(script_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(script)
    return script_path, result_path


def result_ready(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def _listen(port, script_path, result_path, state):
    """Talk to Maya off Blender's main thread.

    The command port sends its reply only after the script returns. Leaving
    that reply unread, or waiting for it on the UI thread, freezes Blender
    for the whole import.
    """
    # Maya 2023 runs the port command with separate globals and locals.
    # Functions in the script then cannot see result_path or fbx_path.
    command = "ns={}\nexec(open(%s, encoding='utf-8').read(), ns, ns)\n" % repr(script_path)
    sock = None
    try:
        sock = socket.create_connection(("127.0.0.1", int(port)), timeout=10)
        sock.settimeout(0.5)
        sock.sendall(command.encode("utf-8"))
        chunks = []
        while not state["cancel"]:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            chunks.append(chunk)
            if b"\x00" in chunk:
                break
        reply = b"".join(chunks).split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
        if reply and reply != "None" and not result_ready(result_path):
            state["error"] = reply
    except ConnectionRefusedError:
        state["error"] = (
            "Maya is not listening on port %d. In Maya's Script Editor (Python), run:\n%s"
            % (port, PORT_COMMAND.replace(":4434", ":%d" % int(port)))
        )
    except OSError as exc:
        if not state["cancel"]:
            state["error"] = "Could not reach Maya on port %d: %s" % (port, exc)
    finally:
        state["done"] = True
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def read_result(path):
    if not os.path.exists(path):
        raise SendToMayaError("Maya did not report a result. Check the Script Editor.")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def prepare_send(context, apply_modifiers, apply_transforms, zero_location, remove_materials):
    meshes = selected_meshes(context)
    if not meshes:
        raise SendToMayaError("Select at least one mesh")
    if context.mode != "OBJECT":
        raise SendToMayaError("Object Mode is required")

    context.view_layer.update()
    depsgraph = context.evaluated_depsgraph_get()
    payload = [mesh_payload(obj, depsgraph, apply_modifiers) for obj in meshes]

    fbx_handle = tempfile.NamedTemporaryFile(prefix="work_tools_", suffix=".fbx", delete=False)
    fbx_path = fbx_handle.name
    fbx_handle.close()
    try:
        export_fbx(context, meshes, fbx_path, apply_modifiers, apply_transforms, zero_location, remove_materials)
        script_path, result_path = write_maya_script(fbx_path, payload, remove_materials)
    except Exception:
        cleanup_send(fbx_path, None, None)
        raise
    return fbx_path, script_path, result_path


def cleanup_send(fbx_path, script_path, result_path):
    for path in (fbx_path, script_path, result_path, (result_path + ".part") if result_path else None):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    if script_path:
        directory = os.path.dirname(script_path)
        try:
            os.rmdir(directory)
        except OSError:
            pass


def interpret_result(result, mesh_count):
    if result.startswith("FAIL"):
        detail = result.split("\n", 1)[1].strip() if "\n" in result else result
        raise SendToMayaError(detail or "Maya import failed")
    lines = [line for line in result.splitlines() if line]
    sent = lines[0].split()[-1] if lines else str(mesh_count)
    return sent, lines[1:]


class WorkToolsMayaSettings(bpy.types.PropertyGroup):
    port: IntProperty(
        name="Port",
        description="Python command port Maya is listening on",
        default=4434,
        min=1,
        max=65535,
    )
    apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Send the evaluated mesh. Modifiers stay on the Blender object",
        default=True,
    )
    apply_transforms: BoolProperty(
        name="Apply Transforms",
        description="Bake location, rotation and scale into the mesh sent to Maya. The Blender object is not changed",
        default=False,
    )
    zero_location: BoolProperty(
        name="Zero Location",
        description="Move the object to the world origin for the export, then put it back. Rotation and scale stay",
        default=False,
    )
    remove_materials: BoolProperty(
        name="Remove Materials",
        description="Do not send materials to Maya. Materials on the Blender object stay. A material already on the Maya object is kept when updating",
        default=True,
    )
    check_mesh: BoolProperty(
        name="Check Mesh",
        description="Run Check Mesh before sending. Unwelded vertices, extra UV channels, or n-gons cancel the send",
        default=True,
    )


class OBJECT_OT_send_to_maya(bpy.types.Operator):
    bl_idname = "object.send_to_maya"
    bl_label = "Send to Maya"
    bl_description = "Export the selected meshes into the open Maya scene"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        if getattr(cls, "_busy", False):
            return True
        return context.mode == "OBJECT" and bool(selected_meshes(context))

    def execute(self, context):
        if getattr(self.__class__, "_busy", False):
            self.report({"WARNING"}, "Already sending to Maya")
            return {"CANCELLED"}
        settings = context.scene.work_tools_maya
        if settings.check_mesh and self._block_on_mesh_problems(context):
            return {"CANCELLED"}
        try:
            fbx_path, script_path, result_path = prepare_send(
                context,
                settings.apply_modifiers,
                settings.apply_transforms,
                settings.zero_location,
                settings.remove_materials,
            )
        except SendToMayaError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        self._fbx_path = fbx_path
        self._script_path = script_path
        self._result_path = result_path
        self._mesh_count = len(selected_meshes(context))
        self._state = {"done": False, "error": None, "cancel": False}
        self._closed = False
        self._started = time.time()
        self.__class__._busy = True
        threading.Thread(
            target=_listen,
            args=(settings.port, script_path, result_path, self._state),
            daemon=True,
        ).start()
        window_manager = context.window_manager
        self._timer = window_manager.event_timer_add(0.2, window=context.window)
        window_manager.modal_handler_add(self)
        self._set_status(context, "Sending to Maya...")
        return {"RUNNING_MODAL"}

    def _block_on_mesh_problems(self, context):
        reports = []
        first_bad = None
        first_found = None
        for obj in selected_meshes(context):
            found = check_mesh(obj)
            lines = _report_lines(found)
            if not lines:
                continue
            if first_bad is None:
                first_bad = obj
                first_found = found
            reports.append(f"{obj.name}: {', '.join(lines)}")
        if not reports:
            return False
        context.view_layer.objects.active = first_bad
        if first_found["unwelded"] or first_found["ngons"]:
            _select_problems(context, first_bad, first_found["unwelded"], first_found["ngons"])
        text = "\n".join(reports)
        context.scene.work_tools_check_mesh.report = text
        self.report({"WARNING"}, text.replace("\n", ". "))
        return True

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            self._state["cancel"] = True
            self._close(context, delete=False)
            self.report({"WARNING"}, "Stopped waiting. Maya may still finish the import.")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        elapsed = int(time.time() - self._started)
        self._set_status(context, "Sending to Maya... %ds (Esc to stop waiting)" % elapsed)
        if result_ready(self._result_path):
            try:
                sent, notes = interpret_result(read_result(self._result_path), self._mesh_count)
            except SendToMayaError as exc:
                self._close(context)
                self.report({"WARNING"}, str(exc))
                return {"CANCELLED"}
            self._close(context)
            message = "Sent %s mesh(es) to Maya" % sent
            if notes:
                message = "%s. %s" % (message, " ".join(notes))
            self.report({"WARNING"} if notes else {"INFO"}, message)
            return {"FINISHED"}
        if self._state["done"]:
            message = self._state["error"] or "Maya did not report a result. Check the Script Editor."
            self._close(context)
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        if elapsed > 600:
            self._state["cancel"] = True
            self._close(context, delete=False)
            self.report({"WARNING"}, "Maya timed out. Check the Script Editor.")
            return {"CANCELLED"}
        return {"PASS_THROUGH"}

    def cancel(self, context):
        if getattr(self, "_state", None) is not None:
            self._state["cancel"] = True
        self._close(context, delete=False)

    def _close(self, context, delete=True):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.__class__._busy = False
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        if delete:
            cleanup_send(
                getattr(self, "_fbx_path", None),
                getattr(self, "_script_path", None),
                getattr(self, "_result_path", None),
            )

    @staticmethod
    def _set_status(context, text):
        context.workspace.status_text_set(text)


class OBJECT_OT_copy_maya_port_command(bpy.types.Operator):
    bl_idname = "object.copy_maya_port_command"
    bl_label = "Copy Port Command"
    bl_description = "Copy the Python command that opens Maya's command port"

    def execute(self, context):
        port = context.scene.work_tools_maya.port
        command = PORT_COMMAND.replace(":4434", ":%d" % port)
        context.window_manager.clipboard = command
        self.report({"INFO"}, "Maya port command copied")
        return {"FINISHED"}


class VIEW3D_PT_send_to_maya(bpy.types.Panel):
    bl_label = "Maya"
    bl_idname = "VIEW3D_PT_send_to_maya"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Work Tools"
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        if context.mode != "OBJECT":
            layout.label(text="Object Mode is required")
            return
        settings = context.scene.work_tools_maya
        col = layout.column(align=True)
        col.prop(settings, "port")
        col.prop(settings, "apply_modifiers")
        col.prop(settings, "apply_transforms")
        col.prop(settings, "zero_location")
        col.prop(settings, "remove_materials")
        col.prop(settings, "check_mesh")
        col.operator(OBJECT_OT_copy_maya_port_command.bl_idname, icon="COPYDOWN")
        col.separator()
        col.operator(OBJECT_OT_send_to_maya.bl_idname, icon="EXPORT")


def draw_object_menu(layout):
    layout.operator(OBJECT_OT_send_to_maya.bl_idname, icon="EXPORT")


_CLASSES = (
    WorkToolsMayaSettings,
    OBJECT_OT_send_to_maya,
    OBJECT_OT_copy_maya_port_command,
    VIEW3D_PT_send_to_maya,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.work_tools_maya = PointerProperty(type=WorkToolsMayaSettings)


def unregister():
    del bpy.types.Scene.work_tools_maya
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
