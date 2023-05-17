import open3d as o3d
import cv2
import numpy as np
import os
from os.path import join, dirname, realpath
import json
from scipy.spatial.transform import Rotation

dataset_folder = '/home/uwcviss/datasets/project_1'

if os.path.exists(join(dataset_folder, 'intrinsics.json')):
    with open(join(dataset_folder, 'intrinsics.json')) as f:
        intrinsics = json.load(f)
    K2 = np.array(intrinsics['camera_matrix'])
    w0 = np.array(intrinsics['width'])
else:
    raise Exception('No intrinsics file found')

img_id = 0

depth = cv2.imread(join(dataset_folder,'output',f'est_depth_viz{img_id}.png'),cv2.IMREAD_UNCHANGED) 
color = cv2.cvtColor(cv2.imread(join(dataset_folder,'output',f'est_image_viz{img_id}.jpg')), cv2.COLOR_BGR2RGB)

# depth = cv2.imread(join(dataset_folder,f'depth000000.png'),cv2.IMREAD_UNCHANGED) 
# color = cv2.cvtColor(cv2.imread(join(dataset_folder,f'frame000000.jpg')), cv2.COLOR_BGR2RGB)
height, width, _ = color.shape

scale = width / w0

fx = K2[0,0]*scale
fy = K2[1,1]*scale
cx = K2[0,2]*scale
cy = K2[1,2]*scale

depth = o3d.geometry.Image(np.float32(depth)/1000)
color = o3d.geometry.Image(color)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth,depth_trunc=100000, convert_rgb_to_intensity=False)


intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# poses = np.loadtxt(join(dataset_folder,'poses.csv'), delimiter=',')
# pose = poses[img_id-1,:]
pose = np.array([1, 0,0,0, 0,0,0,1])
pose = pose.reshape(-1)
q = pose[4:]
ext = np.eye(4)
ext[:3,:3] = Rotation.from_quat(q).as_matrix()
ext[:3,3] = np.array(pose[1:4])
ext = np.linalg.inv(ext)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr, ext)

o3d.io.write_point_cloud(join(dataset_folder, 'cloud.ply'),pcd)


# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
# vis.run()
