import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import open3d as o3d
import os
from probreg import cpd
import probreg

def preprocess_images(directory_path, res = 720):
    images = []
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


    for filename in os.listdir(directory_path):
        img_path = os.path.join(directory_path, filename)

        if not os.path.isfile(img_path):
            continue

        if not filename.lower().endswith(valid_ext):
            continue

        image = Image.open(img_path)
        image = image.convert("RGB")
        new_height = res if image.height > res else image.height
        new_height -= (new_height % 32)
        new_width = int(new_height* image.width/image.height)
        diff = new_width % 32    
        new_width = new_width - diff if diff < 16 else new_width + 32 - diff
        new_size=(new_width, new_height)
        image = image.resize(new_size)
        images.append(image)

    return images

def generate_pcd(directory_path, feature_extractor, model, res=720):
    
    pcd_list=[]

    images = preprocess_images(directory_path, res)
    inputs = feature_extractor(images=images, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_outputs = feature_extractor.post_process_depth_estimation(
        outputs,
        target_sizes=[(img.height, img.width) for img in images],
    )

    for i in range(len(post_processed_outputs)):
        image = images[i]

        predicted_depth = post_processed_outputs[i]["predicted_depth"]
        output = predicted_depth.detach().cpu().numpy() * 100
        width, height = image.size
        depth_image = ((output - output.min()) * 255 / (output.max() - output.min())) # keep values in 0,255
        depth_image = depth_image.max() - depth_image  # Invert depth values, because depthanything model gives higher value to close objects and lower to far. Point cloud is inverted if this step not done.
    
        threshold = 125 # points below the threshold will be pushed back
        lower_bound = 75 # minimum depth to avoid convergence of nearby pixels towards principle center
        # Essentially, we want the values below 125 to gradually increase towards 125, starting from 75. closer it is to 125, lesser it will be pushed back. (threshold - 75) is basically a scale of how much to push back.
        depth_image[depth_image < threshold] = (depth_image[depth_image < threshold] / threshold) * (threshold - 75) + 75
     
        # Ensure that no value goes above 255 (in case of overflow)
        depth_image = np.clip(depth_image, 0, 255)
    
        image_np = np.array(image)
    
        # Create rgbd image
        depth_o3d = o3d.t.geometry.Image(depth_image)
        image_o3d = o3d.t.geometry.Image(image_np)
        rgbd_image = o3d.t.geometry.RGBDImage(image_o3d, depth_o3d)
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(width, height, 1000,1000, width/2, height/2)
        raw_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.core.Tensor(camera_intrinsic.intrinsic_matrix))
        # Remove outliers
        pcd, ind = raw_pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=10.0)
            
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction()

        pcd_list.append(pcd)

    return pcd_list


def preprocess_pcd(pcd, voxel_size = 0.025):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius = voxel_size*5, max_nn = 100)
    )
    return pcd_down, pcd_fpfh


def icp(source, target):
    trans_init = o3d.core.Tensor.eye(4, o3d.core.float32)    
        
    # Remove statistical outliers (removes noisy depth points)
    source, _ = source.remove_statistical_outliers(nb_neighbors=20, std_ratio=3.0)
    target, _ = target.remove_statistical_outliers(nb_neighbors=20, std_ratio=3.0)
     
    source_legacy = source.to_legacy()
    target_legacy = target.to_legacy()
    source_down, source_fpfh = preprocess_pcd(source_legacy)
    target_down, target_fpfh = preprocess_pcd(target_legacy)
    
    # Global Registration    
    # FGR (Rigid Alignment)
    result_fgr =  o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=0.025, 
        )
    )
    print("FGR Complete")
     
    source_down.transform(result_fgr.transformation)
    source_legacy.transform(result_fgr.transformation)

    slu = source_legacy.uniform_down_sample(80)
    tlu =target_legacy.uniform_down_sample(80)

    # tf_param,_,_ = cpd.registration_cpd(slu, tlu, tf_type_name= 'nonrigid',  lmd=9000.0)    
    # slu.points = tf_param.transform(slu.points)
    # tf_param,_,_ = probreg.filterreg.registration_filterreg(slu, tlu,objective_type='pt2pt'   )
    # slu.points = tf_param.transform(slu.points)

    
    # Bayesian Coherent Point Drift for non-rigid registration
    tf_param = probreg.bcpd.registration_bcpd(slu, tlu)
    slu.points = tf_param.transform(slu.points)
    print("BCPD complete")
    

    s = o3d.t.geometry.PointCloud.from_legacy(slu)
    t = o3d.t.geometry.PointCloud.from_legacy(tlu)


    # Point to Plane ICP Registration
    reg_p2plane = o3d.t.pipelines.registration.icp(
        s,t, 
        max_correspondence_distance=0.02,  # Smaller distance for more precise alignment
        init_source_to_target=o3d.core.Tensor(result_fgr.transformation),  # Initial transformation tensor
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane())

    print("P2Plane Complete")
    s.transform(reg_p2plane.transformation)
    source.transform(reg_p2plane.transformation)

        
    # Colored registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()    
    voxel_sizes = o3d.utility.DoubleVector([0.06, 0.02, 0.005])
    max_correspondence_distances = o3d.utility.DoubleVector([0.08, 0.025, 0.008])
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(  
            relative_fitness=0.0001,
            relative_rmse=0.0001,
            max_iteration=50
        ),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 30),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 14)
    ]
    
    reg_multiscale_icp = o3d.t.pipelines.registration.multi_scale_icp(
        s, t, voxel_sizes,
        criteria_list,
        max_correspondence_distances,
        trans_init, 
        estimation
    )

    print("Colored Registration Complete")

    s.transform(reg_multiscale_icp.transformation)
    source.transform(reg_multiscale_icp.transformation)
      
    # Merge the two point clouds into a single point cloud
    return s+t