import numpy as np
from skimage import measure
from scipy import spatial
import open3d as o3d

print("🔹 Chargement du nuage de points...")
# === Charger nuage de points avec normales (shape: [N, 6])
p = np.loadtxt('armadillo_sub.xyz')
points = p[:, 0:3]
normals = p[:, 3:]

print("✅ Nuage de points chargé.")

# === Boîte englobante + normalisation
print("🔹 Normalisation des points...")
min_coords = points.min(axis=0)
max_coords = points.max(axis=0)
center = (min_coords + max_coords) / 2
scale = (max_coords - min_coords).max()
points_normalized = (points - center) / scale * 2
print("✅ Normalisation terminée.")

# === Grille régulière
print("🔹 Construction de la grille 3D...")
res = 128
X, Y, Z = np.mgrid[-1:1:complex(res), -1:1:complex(res), -1:1:complex(res)]
grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
print(f"✅ Grille créée avec résolution {res}³.")

# === KD-Tree pour recherche rapide
print("🔹 Construction du KD-Tree...")
kdtree = spatial.KDTree(points_normalized)
print("✅ KD-Tree prêt.")

# === Calcul de la Signed Distance Function (SDF)
print("🔹 Calcul de la SDF... (peut être long)")
u = np.zeros(X.shape)

for i in range(res):
    if i % 10 == 0:
        print(f"  → Ligne {i}/{res}")
    for j in range(res):
        for k in range(res):
            query = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
            dist, idx = kdtree.query(query)
            closest_point = points_normalized[idx]
            normal = normals[idx]
            vec = query - closest_point
            signed_distance = np.dot(vec, normal)
            u[i, j, k] = signed_distance

print("✅ SDF calculée.")

# === Reconstruction de surface par Marching Cubes
print("🔹 Reconstruction par Marching Cubes...")
vertices, triangles, _, _ = measure.marching_cubes(u, level=0)
print(f"✅ Mesh généré avec {len(vertices)} sommets et {len(triangles)} triangles.")

# === Export du mesh en .obj
print("🔹 Exportation du mesh vers 'result.obj'...")
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
o3d.io.write_triangle_mesh("result.obj", mesh)
print("✅ Fichier 'result.obj' exporté avec succès.")
