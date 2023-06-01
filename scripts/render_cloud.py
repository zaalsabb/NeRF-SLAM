import open3d as o3d
import cv2
import numpy as np
import os
from os.path import join, dirname, realpath
import json
from scipy.spatial.transform import Rotation
import sys

def main(img_id):

    project_id = 1
    home = os.environ.get('HOME')
    dataset_folder = os.path.join(f"{home}/datasets", f"project_{project_id}")

    with open(join(dataset_folder, 'transforms.json')) as f:
        transforms = json.load(f)

    K2 = np.eye(3)

    K2[0,0] = transforms["fl_x"]
    K2[1,1] = transforms["fl_y"]
    K2[0,2] = transforms["cx"]   
    K2[1,2] = transforms["cy"]   
    w0      = transforms["w"]    

    depth = cv2.imread(join(dataset_folder,'output',f'est_depth_viz{img_id}.png'),cv2.IMREAD_UNCHANGED) 
    color = cv2.cvtColor(cv2.imread(join(dataset_folder,'output',f'est_image_viz{img_id}.jpg')), cv2.COLOR_BGR2RGB)

    # color = cv2.cvtColor(cv2.imread(f'/home/uwcviss/datasets/steel_tree/iphone/rgb/{int(img_id)+1:06}.jpg'),cv2.COLOR_BGR2RGB)
    # depth = cv2.imread(f'/home/uwcviss/datasets/steel_tree/iphone/depth/{int(img_id)+1:06}.png',cv2.IMREAD_UNCHANGED)
    # color = cv2.resize(color, (depth.shape[1],depth.shape[0]))

    height, width, _ = color.shape
    scale = width / w0

    fx = K2[0,0]*scale
    fy = K2[1,1]*scale
    cx = K2[0,2]*scale
    cy = K2[1,2]*scale

    depth = o3d.geometry.Image(np.float32(depth))
    # depth = o3d.geometry.Image(np.float32(depth)/1000)
    color = o3d.geometry.Image(color)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,depth_trunc=100000, convert_rgb_to_intensity=False)

    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # t = transforms["frames"][int(img_id)]
    # T_wc = np.array(t["transform_matrix"])
    # # T_wc[0:3, 1] *= -1  # flip the y axis
    # # T_wc[0:3, 2] *= -1  # flip the z axis    
    # # T_wc[:,3] /= 1.784667785881502    
    # T_cw = np.linalg.inv(T_wc)
    
    with open(join(dataset_folder, 'output/poses.json')) as f:
        poses = json.load(f)    
    pose = np.array(poses[img_id])
    T_wc = pose2matrix(pose)
    # T_lc = pose2matrix([0,0,0,-0.5,0.5,-0.5,0.5])
    # T_wl = T_wc
    # T_wc = T_wl.dot(T_lc)
    T_cw = np.linalg.inv(T_wc)


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr, extrinsic=T_cw)

    o3d.io.write_point_cloud(join(dataset_folder, f'cloud{img_id}.ply'),pcd)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    # vis.run()

def pose2matrix(pose):
    p = pose[:3]
    q = pose[3:]
    R = Rotation.from_quat(q)
    T_m_c = np.eye(4)
    T_m_c[:3, :3] = R.as_matrix()
    T_m_c[:3, 3] = p
    return T_m_c

if __name__ == '__main__':
    main(sys.argv[1])