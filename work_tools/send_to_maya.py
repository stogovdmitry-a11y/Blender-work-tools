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
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, IntProperty, PointerProperty

PORT_COMMAND = (
    "import maya.cmds as cmds\n"
    "cmds.commandPort(name=\":4434\", sourceType=\"python\")"
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


def export_fbx(context, meshes, filepath, apply_modifiers):
    ensure_fbx_exporter()
    view_layer = context.view_layer
    previous = list(context.selected_objects)
    active = view_layer.objects.active
    try:
        for obj in previous:
            obj.select_set(False)
        for obj in meshes:
            obj.select_set(True)
        view_layer.objects.active = meshes[0]
        result = bpy.ops.export_scene.fbx(
            filepath=filepath,
            check_existing=False,
            use_selection=True,
            object_types={"MESH"},
            use_mesh_modifiers=apply_modifiers,
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
            global_scale=1.0,
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
    if "FINISHED" not in result:
        raise SendToMayaError("FBX export failed")


MAYA_SCRIPT = r'''
import json
import traceback

fbx_path = __FBX_PATH__
result_path = __RESULT_PATH__
payload = json.loads(__PAYLOAD__)

def write_result(text):
    with open(result_path, "w", encoding="utf-8") as handle:
        handle.write(text)

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

    before = set(cmds.ls(long=True, type="mesh") or [])
    mel_try("FBXImportMode -v add")
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
    for item in payload:
        shape = None
        for candidate in unused:
            if short_name(candidate) == item["name"] or short_name(candidate).startswith(item["name"]):
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
        restored += 1

    transforms = []
    for shape in shapes:
        transforms.extend(cmds.listRelatives(shape, parent=True, fullPath=True) or [])
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


def write_maya_script(fbx_path, payload):
    directory = tempfile.mkdtemp(prefix="work_tools_maya_")
    script_path = os.path.join(directory, "send.py")
    result_path = os.path.join(directory, "result.txt")
    script = (
        MAYA_SCRIPT
        .replace("__FBX_PATH__", json.dumps(fbx_path))
        .replace("__RESULT_PATH__", json.dumps(result_path))
        .replace("__PAYLOAD__", json.dumps(json.dumps(payload)))
    )
    with open(script_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(script)
    return script_path, result_path


def command_port(port, script_path, result_path):
    command = "exec(open(%s, encoding='utf-8').read())\n" % repr(script_path)
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=10) as sock:
            sock.settimeout(120)
            sock.sendall(command.encode("utf-8"))
            deadline = time.time() + 120
            while time.time() < deadline:
                if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                    return
                time.sleep(0.05)
    except ConnectionRefusedError as exc:
        raise SendToMayaError(
            "Maya is not listening on port %d. In Maya's Script Editor (Python), run:\n%s"
            % (port, PORT_COMMAND)
        ) from exc
    except OSError as exc:
        raise SendToMayaError("Could not reach Maya on port %d: %s" % (port, exc)) from exc
    raise SendToMayaError("Maya timed out. Check the Script Editor.")


def read_result(path):
    if not os.path.exists(path):
        raise SendToMayaError("Maya did not report a result. Check the Script Editor.")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def send_selected_to_maya(context, port, apply_modifiers):
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
    script_path = result_path = None
    try:
        export_fbx(context, meshes, fbx_path, apply_modifiers)
        script_path, result_path = write_maya_script(fbx_path, payload)
        command_port(port, script_path, result_path)
        result = read_result(result_path)
    finally:
        for path in (fbx_path, script_path, result_path):
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

    if result.startswith("FAIL"):
        detail = result.split("\n", 1)[1].strip() if "\n" in result else result
        raise SendToMayaError(detail or "Maya import failed")
    lines = [line for line in result.splitlines() if line]
    sent = lines[0].split()[-1] if lines else str(len(meshes))
    notes = lines[1:]
    return sent, notes


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


class OBJECT_OT_send_to_maya(bpy.types.Operator):
    bl_idname = "object.send_to_maya"
    bl_label = "Send to Maya"
    bl_description = "Export the selected meshes into the open Maya scene"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and bool(selected_meshes(context))

    def execute(self, context):
        settings = context.scene.work_tools_maya
        try:
            sent, notes = send_selected_to_maya(context, settings.port, settings.apply_modifiers)
        except SendToMayaError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        message = "Sent %s mesh(es) to Maya" % sent
        if notes:
            message = "%s. %s" % (message, " ".join(notes))
        self.report({"WARNING"} if notes else {"INFO"}, message)
        return {"FINISHED"}


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

    def draw(self, context):
        layout = self.layout
        if context.mode != "OBJECT":
            layout.label(text="Object Mode is required")
            return
        settings = context.scene.work_tools_maya
        col = layout.column(align=True)
        col.prop(settings, "port")
        col.prop(settings, "apply_modifiers")
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
