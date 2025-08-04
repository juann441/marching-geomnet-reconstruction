
import numpy as np
import torch
from torch import nn
from scipy import spatial
from matplotlib import pyplot as plt
import torch.nn.functional as F

from skimage import measure
import open3d as o3d

class GeomNet(nn.Module):
    def __init__(self, nlayers, nneurons):
        super(GeomNet, self).__init__()

        layers = []

        # Layer d'entrée : de 3 (x,y,z) à nneurons
        layers.append(nn.Linear(3, nneurons))
        layers.append(nn.ReLU())

        # Couches intermédiaires
        for _ in range(nlayers - 2):  # -2 car input + output
            layers.append(nn.Linear(nneurons, nneurons))
            layers.append(nn.ReLU())

        # Couche de sortie : 1 valeur scalaire (SDF)
        layers.append(nn.Linear(nneurons, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

#Cette fonction calcule le gradient de la sortie par rapport à l 'entrée (utile pour la contrainte eikonale)    
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


#fonction qui calcule la loss d'alignement des normales
def sdf_loss_align(grad, normals):
    return (1-nn.functional.cosine_similarity(grad, normals, dim = 1)).mean()

def evaluate_loss(geomnet, p, device, lpc, leik, lambda_pc, lambda_eik, batch_size=2000, pc_batch_size=2000):
    # points aléatoires dans [-1, 1]^3
    pts_random = torch.rand((batch_size, 3), device=device) * 2 - 1
    pts_random.requires_grad = True

    # échantillonner des points du nuage avec normales
    sample = torch.randint(p.shape[0], (pc_batch_size,))
    sample_pc = p[sample, 0:3]
    sample_pc.requires_grad = True  # Important pour les gradients

    sample_nc = p[sample, 3:]

    # === SDF estimé par le réseau ===
    sdf_pc = geomnet(sample_pc).squeeze(1)       # [pc_batch_size]
    sdf_random = geomnet(pts_random).squeeze(1)  # [batch_size]

    # === Gradients pour contraintes eikonales et alignement des normales ===
    grad_pc = gradient(sdf_pc, sample_pc)        # [pc_batch_size, 3]
    grad_random = gradient(sdf_random, pts_random)  # [batch_size, 3]

    # === Losses ===
    loss_pc = sdf_loss_align(grad_pc, sample_nc)  # alignement normales
    loss_eik = ((grad_random.norm(2, dim=1) - 1) ** 2).mean()  # eikonale

    # Stocker pour affichage
    lpc.append(float(loss_pc))
    leik.append(float(loss_eik))

    # Loss totale pondérée
    loss = lambda_pc * loss_pc + lambda_eik * loss_eik

    return loss

def main():
    print("🔹 Chargement du nuage de points...")
    p = np.loadtxt('armadillo_sub.xyz')  # [N, 6] : x, y, z, nx, ny, nz

    print("🔹 Calcul de la boîte englobante et normalisation...")
    min_coords = p[:, 0:3].min(axis=0)
    max_coords = p[:, 0:3].max(axis=0)
    center = (min_coords + max_coords) / 2
    scale = (max_coords - min_coords).max()
    p[:, 0:3] = (p[:, 0:3] - center) / scale * 2
    print("✅ Normalisation terminée.")

    print("🔹 Création du réseau GeomNet...")
    geomnet = GeomNet(nlayers=10, nneurons=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    geomnet.to(device)
    points = torch.from_numpy(p).float().to(device)

    lpc, leik = [], []
    lambda_pc = 1
    lambda_eik = 1
    optim = torch.optim.Adam(params=geomnet.parameters(), lr=1e-3)
    nepochs = 5000

    print(f"🔹 Démarrage de l'entraînement pour {nepochs} epochs...")
    for epoch in range(nepochs):
        optim.zero_grad()
        loss = evaluate_loss(geomnet, points, device, lpc, leik, lambda_pc, lambda_eik)
        loss.backward()
        optim.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")

    print("✅ Entraînement terminé.")

    print("🔹 Création de la grille 3D pour l'évaluation du réseau...")
    res = 128
    x = torch.linspace(-1, 1, res)
    y = torch.linspace(-1, 1, res)
    z = torch.linspace(-1, 1, res)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)
    print(f"✅ Grille créée avec {grid_points.shape[0]} points.")

    print("🔹 Évaluation du réseau sur la grille...")
    with torch.no_grad():
        sdf_values = geomnet(grid_points).reshape(res, res, res).cpu().numpy()
    print("✅ Évaluation terminée.")
    print("🔹 Application de Marching Cubes...")
    min_sdf = sdf_values.min()
    max_sdf = sdf_values.max()
    print(f"SDF range: min={min_sdf}, max={max_sdf}")

    level = 0
    if level < min_sdf or level > max_sdf:
        level = (min_sdf + max_sdf) / 2
        print(f"Level 0 hors intervalle, niveau utilisé: {level}")

    vertices, triangles, _, _ = measure.marching_cubes(sdf_values, level=level)

    # Création du mesh Open3D
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Export du mesh au format PLY (tu peux mettre "reconstruction.obj" si tu préfères)
    o3d.io.write_triangle_mesh("reconstruction.ply", mesh)
    print("✅ Marching Cubes terminé, fichier 'reconstruction.ply' généré.")

    print("🔹 Tracé des courbes de perte...")
    plt.figure(figsize=(6, 4))
    plt.yscale('log')
    plt.plot(lpc, label='Point cloud loss ({:.2f})'.format(lpc[-1]))
    plt.plot(leik, label='Eikonal loss ({:.2f})'.format(leik[-1]))
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("loss.pdf")
    plt.close()
    print("✅ Courbes de perte sauvegardées dans 'loss.pdf'.")

# N'oublie pas d'appeler main()
if __name__ == "__main__":
    main()