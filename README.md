# 🧊 Marching Cubes + GeomNet Surface Reconstruction

This mini-project explores **surface reconstruction** from 3D point clouds using:

- A simple **Signed Distance Function (SDF)** computed from normals
- The **Marching Cubes** algorithm to generate a mesh
- (Optional) A neural network-based reconstruction method like **GeomNet**
<p align="center">
  <img src="res_32.png" width="22%" />
  <img src="res_64.png" width="22%" />
  <img src="res_128.png" width="22%" />
  <img src="res_256.png" width="22%" />
</p>

<p align="center">
  <b>32³</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>64³</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>128³</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>256³</b>
</p>


## 📁 Structure

- `src/reconstruction.py` : Main pipeline — normalization, KDTree, SDF, Marching Cubes
- `data/` : Example point cloud (with normals)
- `exports/` : Sample output (`.obj`)
- `notebooks/demo.ipynb` : Google Colab-friendly notebook version

---

## 🚀 Quickstart

### 1. Install requirements

```bash
pip install -r requirements.txt
```
### 2. Run the reconstruction
```bash
python src/reconstruction.py --input data/sample_pointcloud.txt --output exports/result.obj
```
### 🛠️ Dependencies
numpy
scipy
scikit-image
open3d
Optional (for GeomNet):
torch
torchvision
### 🧠 About GeomNet
If you activate GeomNet, the system replaces the classic SDF with a learned implicit function for better reconstruction on noisy or sparse clouds

## 📦 Dataset

📥 [Download via Google Drive](https://drive.google.com/file/d/19RXyfxno2OUtCPDOVC9p1yhROWgok5Jv/view?usp=drive_link)

