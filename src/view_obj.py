import open3d as o3d

mesh = o3d.io.read_triangle_mesh("result.obj")
mesh.compute_vertex_normals()  # Pour un rendu plus joli

o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
