import open3d
import numpy as np

# pcd = open3d.io.read_point_cloud(".\meshes\test (15).jpg.ply")
pcd = open3d.io.read_point_cloud(".\pointclouds\merged.ply")
mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, n_threads=4)
rotation = mesh.get_rotation_matrix_from_xyz((np.pi,0,0))
mesh.rotate(rotation, center=(0,0,0))
# remove vertices with low densities
vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)
    
# visualize the mesh
open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
