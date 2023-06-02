# import open3d as o3d
import cv2
import numpy as np
import os
from os.path import join, dirname, realpath
import json
from scipy.spatial.transform import Rotation
import sys

def combine_clouds(buffer=100, stride=8, max_depth=1):

    project_id = 1
    home = os.environ.get('HOME')
    dataset_folder = os.path.join(f"{home}/datasets", f"project_{project_id}")

    with open(join(dataset_folder, 'transforms.json')) as f:
        transforms = json.load(f)

    with open(join(dataset_folder, 'output/poses.json')) as f:
        poses = json.load(f)  

    with open(join(dataset_folder, 'output/keyframes.json')) as f:
        kf_idx_to_f_idx = json.load(f)                   

    K2 = np.eye(3)

    K2[0,0] = transforms["fl_x"]
    K2[1,1] = transforms["fl_y"]
    K2[0,2] = transforms["cx"]   
    K2[1,2] = transforms["cy"]   
    w0      = transforms["w"]    

    scale_l = []

    pcd = o3d.geometry.PointCloud()

    for img_id in range(0,buffer,stride):

        depth = cv2.imread(join(dataset_folder,'output',f'est_depth_viz{img_id}.png'),cv2.IMREAD_UNCHANGED) 
        depth = np.float32(depth)
        depth[depth>max_depth*1000] = 0

        color = cv2.cvtColor(cv2.imread(join(dataset_folder,'output',f'est_image_viz{img_id}.jpg')), cv2.COLOR_BGR2RGB)

        height, width, _ = color.shape
        scale = width / w0

        fx = K2[0,0]*scale
        fy = K2[1,1]*scale
        cx = K2[0,2]*scale
        cy = K2[1,2]*scale

        depth = o3d.geometry.Image(depth)
        color = o3d.geometry.Image(color)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=100000, convert_rgb_to_intensity=False)

        intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        pose = np.array(poses[str(img_id)])
        T_wc = pose2matrix(pose)
        T_cw = np.linalg.inv(T_wc)

        pcd += o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr, extrinsic=T_cw)

        # gt pose
        f_idx = kf_idx_to_f_idx[str(img_id)] * 2
        t = transforms["frames"][f_idx]
        T_wc = np.array(t["transform_matrix"])
        pose_gt = matrix2pose(T_wc)        

        if img_id == 0:
            pose0 = pose
            pose_gt0 = pose_gt
        else:
            scale_l.append(np.linalg.norm(pose_gt[:3]-pose_gt0[:3]) / np.linalg.norm(pose[:3]-pose0[:3]))

    scale = np.mean(scale_l)
    print(f'scale: {scale}')
    T_scale = np.diag([scale, scale, scale, 1])
    pcd.transform(T_scale)

    # pcd_down = pcd.voxel_down_sample(0.01)
    o3d.io.write_point_cloud(join(dataset_folder, f'cloud.ply'),pcd)

def pose2matrix(pose):
    p = pose[:3]
    q = pose[3:]
    R = Rotation.from_quat(q)
    T_m_c = np.eye(4)
    T_m_c[:3, :3] = R.as_matrix()
    T_m_c[:3, 3] = p
    return T_m_c


def matrix2pose(T_m_c):
    R = T_m_c[:3,:3]
    p = T_m_c[:3,3]
    q = Rotation.from_matrix(R).as_quat()

    pose = np.concatenate([p,q])

    return pose


if __name__ == '__main__':
    combine_clouds()