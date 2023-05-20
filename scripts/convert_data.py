import cv2
import requests
import numpy as np
import yaml
import json
import os
import sys
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nerf_slam import NerfSLAM

def load_nerf(project_id: int):     
    home = os.environ.get('HOME')
    dataset_dir = os.path.join(f"{home}/datasets", f"project_{project_id}")
    shutil.rmtree(dataset_dir, ignore_errors=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    nerf_slam = NerfSLAM(dataset_dir)

    return nerf_slam

def main(dataset_dir):
    
    ok_depth = os.path.exists(dataset_dir+'/depth')

    nerf = load_nerf(1)

    f_intrinsics = dataset_dir + '/intrinsics.json'
    with open(f_intrinsics) as f:
        intrinsics = json.load(f)

    poses = np.loadtxt(dataset_dir + '/poses.csv', delimiter=',')

    K = np.array(intrinsics['camera_matrix'])
    w = int(intrinsics['width'])
    h = int(intrinsics['height'])
    d = np.array(intrinsics['dist_coeff'], dtype=np.float64)
    nerf.set_intrinsics(K, w, h, d)

    N = len(os.listdir(os.path.join(dataset_dir , 'rgb')))

    for i in range(1,N):
        f_image = os.path.join(dataset_dir , 'rgb', f'{i:06}.jpg')
        if not os.path.exists(f_image):
            f_image = os.path.join(dataset_dir , 'rgb', f'{i}.png')
        I = cv2.imread(f_image)
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2RGBA)

        if ok_depth:
            f_depth = os.path.join(dataset_dir , 'depth', f'{i:06}.png')
            if not os.path.exists(f_depth):
                f_depth = os.path.join(dataset_dir , 'depth', f'{i}.png')            
            D = cv2.imread(f_depth, cv2.IMREAD_UNCHANGED)
        else:
            D = None

        pose = poses[i-1, 1:]
        nerf.save_image(I, D, pose, i-1)

        print(f"Processing frame {i}/{N}", end="\r")

    nerf.save_intrinsics_file()

if __name__ == '__main__':
    main(sys.argv[1])