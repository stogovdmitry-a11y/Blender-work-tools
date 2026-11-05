"""Work Tools — набор рабочих инструментов для Blender.

Новый инструмент — отдельный модуль со своими register/unregister,
плюс пункт в меню ниже. Панели кладутся во вкладку N-панели «Work Tools».
"""

bl_info = {
    "name": "Work Tools",
    "author": "",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Work Tools",
    "description": "Набор рабочих инструментов",
    "category": "3D View",
}

import bpy
from bpy.types import Menu

from . import seam_normals

modules = (
    seam_normals,
)


class VIEW3D_MT_work_tools(Menu):
    bl_label = "Work Tools"
    bl_idname = "VIEW3D_MT_work_tools"

    def draw(self, _context):
        layout = self.layout
        seam_normals.draw_object_menu(layout)


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
