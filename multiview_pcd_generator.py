from generate_pcd import generate_pcd, icp
import open3d as o3d
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def main():
    feature_extractor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf", use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

    directory_path = "./multiview_data/"
    pcd_list = []
    
    # Generate and Save Point Clouds
    pcd_list = generate_pcd(directory_path,feature_extractor, model)
    for i, p in pcd_list:
        o3d.io.write_point_cloud(f"./pointclouds/pcd{i}.ply", p.to_legacy())
    
    # Merge and Display Point Clouds
    if len(pcd_list) > 1:
        print("Starting registration...")
        merged_pcd = icp(pcd_list[0], pcd_list[1])
        o3d.io.write_point_cloud("./merged.ply", merged_pcd.to_legacy())
        o3d.visualization.draw_geometries([merged_pcd.to_legacy()])
    else:
        o3d.visualization.draw_geometries([pcd_list[0].to_legacy()])
    

if __name__== "__main__":
    main()
