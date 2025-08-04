import open3d as o3d

mesh = o3d.io.read_triangle_mesh("reconstruction.ply")
o3d.visualization.draw_geometries([mesh])