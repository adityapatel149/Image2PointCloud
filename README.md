# Image2PointCloud
Generate 3D point clouds and meshes from 2D images without depth data.

## Overview
**Image2PointCloud** is a Python-based tool that generates 3D point clouds and meshes from one or more 2D images without requiring depth data. It leverages the **DepthAnythingV2** model to estimate depth maps from input images and constructs point clouds accordingly. For multi-view images, the tool performs global and non-rigid registration to refine the point cloud alignment, followed by mesh generation using Poisson surface reconstruction.

## Features
- **Depth Estimation**: Utilizes DepthAnythingV2 to infer depth maps from 2D images.
- **Point Cloud Generation**: Constructs a 3D point cloud from the estimated depth maps.
- **Multi-View Registration**: Aligns multiple point clouds using:
  - Fast Point Feature Histograms (FPFH) for global registration.
  - Bayesian Coherent Point Drift (BCPD) for non-rigid alignment.
  - Point-to-Plane Iterative Closest Point (ICP) registration.
  - Colored point cloud registration for final refinement.
- **Mesh Reconstruction**: Generates a 3D mesh using Poisson surface reconstruction.

## Installation
Ensure you have Python installed along with the necessary dependencies. Install required libraries using:

```bash
pip install -r requirements.txt
```

## Usage
1. **Prepare Input Images**:
   - Place the images in the `multiview_data/` folder.
   - Only include images that will be used for point cloud generation.

2. **Generate Point Clouds**:
   - Run the following command to generate the point cloud:
     ```bash
     python multiview_pcd_generator.py
     ```
   - The generated point clouds will be saved in the `pointclouds/` folder.

3. **Generate Meshes** (Optional):
   - To create a mesh from a point cloud, run:
     ```bash
     python generate_mesh_from_pcd.py
     ```
   - The Poisson surface reconstruction method is used for mesh generation.

## Output
- **Point Clouds**: Stored in `pointclouds/` as `.ply` files.
- **Meshes**: Generated from point clouds and saved in the corresponding format.

## Contributions
Contributions and improvements are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.


