"""Work Tools, a set of working tools for Blender.

A new tool is its own module with register and unregister, plus an entry in
the menu below. Panels go in the Work Tools sidebar tab.
"""

bl_info = {
    "name": "Work Tools",
    "author": "",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Work Tools",
    "description": "A set of working tools for Blender",
    "category": "3D View",
}

import bpy
from bpy.types import Menu

from . import check_mesh, seam_normals, send_to_maya, shrinkwrap, snap_uvs, snap_vertices

modules = (
    check_mesh,
    seam_normals,
    send_to_maya,
    shrinkwrap,
    snap_uvs,
    snap_vertices,
)


class VIEW3D_MT_work_tools(Menu):
    bl_label = "Work Tools"
    bl_idname = "VIEW3D_MT_work_tools"

    def draw(self, _context):
        layout = self.layout
        check_mesh.draw_object_menu(layout)
        send_to_maya.draw_object_menu(layout)
        shrinkwrap.draw_object_menu(layout)
        snap_uvs.draw_object_menu(layout)
        snap_vertices.draw_object_menu(layout)


def _draw_object_menu(self, _context):
    self.layout.menu(VIEW3D_MT_work_tools.bl_idname)


def register():
    for module in modules:
        module.register()
    bpy.utils.register_class(VIEW3D_MT_work_tools)
    bpy.types.VIEW3D_MT_object.append(_draw_object_menu)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(_draw_object_menu)
    bpy.utils.unregister_class(VIEW3D_MT_work_tools)
    for module in reversed(modules):
        module.unregister()
