import open3d as o3d
import cv2
import numpy as np
import os
from os.path import join, dirname, realpath
import json
from scipy.spatial.transform import Rotation
import sys
import time

def main(project_id,img_id):

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

    # pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # while True:
    depth = cv2.imread(join(dataset_folder,'output',f'est_depth_viz.png'),cv2.IMREAD_UNCHANGED) 
    color = cv2.cvtColor(cv2.imread(join(dataset_folder,'output',f'est_image_viz.jpg')), cv2.COLOR_BGR2RGB)

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

    T_wc = np.loadtxt(join(dataset_folder, 'output/T_w_c.txt'))
    # T_wc[[0,1,2],[0,1,2]] *= 3.41168844461 * 3
    # T_wc = np.eye(4)
    T_cw = np.linalg.inv(T_wc)
    # T_cw[[0,1,2],[0,1,2]] /= 3.41168844461

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr, extrinsic=T_cw)

    # vis.add_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()

    # time.sleep(0.5)

    o3d.io.write_point_cloud(join(dataset_folder, f'cloud{img_id}.ply'),pcd)

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
    main(sys.argv[1],sys.argv[2])