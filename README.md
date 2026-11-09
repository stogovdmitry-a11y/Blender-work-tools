# Work Tools

Blender tools for matching mesh seams and sending meshes to Maya. For Blender 4.5 or newer.

The active mesh is the target. The other selected mesh is the source.

![Work Tools sidebar](docs/sidebar.png)

- **Snap Seams** snaps selected vertices to the source, then copies normals onto them.
- **Snap UVs** snaps nearby UV points to the source. It does not replace the whole UV map.
- **Shrinkwrap** shrinkwraps the target onto the source and applies the modifier.
- **Check Mesh** finds unwelded vertices, extra UV channels, and n-gons.
- **Maya** sends the selected meshes to an open Maya scene, keeping custom normals and sharp edges. Check Mesh can stop the send when it finds a problem.

Install `work_tools.zip` from the repository root.
