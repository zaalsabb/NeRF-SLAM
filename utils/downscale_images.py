import os
from threading import Thread
import sys
import yaml
import json
import numpy as np
import cv2
import re
import shutil

class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

class NerfSLAM():
    def __init__(self, dataset_dir):
        
        self.dataset_dir = dataset_dir
        self.intrinsics_file = os.path.join(self.dataset_dir, 'transforms.json')
        self.intrinsics = {}
        self.K = np.eye(3)
        self.w = 0
        self.h = 0

        shutil.rmtree(dataset_dir, ignore_errors=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir,'images'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir,'output'), exist_ok=True)

        self.fusion_module = None
        self.slam_module = None
        self.data_module = None    

        self.output_dir = os.path.join(self.dataset_dir,'output')
    

    def convert(self, dataset_dir2, scale, max_num):

        shutil.copy(os.path.join(dataset_dir2,'intrinsics.json'), os.path.join(dataset_dir,'intrinsics.json'))
        with open(os.path.join(dataset_dir,'intrinsics.json')) as f:
            intrinsics = json.load(f)
            
        K = np.array(intrinsics["camera_matrix"])
        K[:3,:3] = scale*K[:3,:3]
        w = intrinsics["width"]*scale
        h = intrinsics["height"]*scale
        self.set_intrinsics(K, w, h)

        for fpath in os.listdir(os.path.join(dataset_dir2, 'images')):

            k = int(re.sub('[^0-9,]', "", fpath))
            if k > max_num:
                continue

            f_image2 = os.path.join(dataset_dir2, 'images', fpath)
            f_image1 = 'images/'+fpath            
            I = cv2.imread(f_image2)
            I = cv2.resize(I, (int(I.shape[1]*scale),int(I.shape[0]*scale)))
            self.save_image(I, f_image1)               
            print(f_image2)
        
        self.save_intrinsics_file()  


    def save_intrinsics_file(self):
        with open(self.intrinsics_file, 'w') as f:
            json.dump(self.intrinsics, f)    

    def load_intrinsics_file(self):
        with open(self.intrinsics_file, 'r') as f:
            self.intrinsics = json.load(f)

        self.K[0,0] = self.intrinsics["fl_x"]
        self.K[1,1] = self.intrinsics["fl_y"]
        self.K[0,2] = self.intrinsics["cx"]   
        self.K[1,2] = self.intrinsics["cy"]   
        self.w      = self.intrinsics["w"]    
        self.h      = self.intrinsics["h"]    

    def set_intrinsics(self, K, w, h):

        self.K = K
        self.w = w
        self.h = h

        self.intrinsics["fl_x"] = K[0,0]
        self.intrinsics["fl_y"] = K[1,1]
        self.intrinsics["k1"]   = 0
        self.intrinsics["k2"]   = 0
        self.intrinsics["p1"]   = 0
        self.intrinsics["p2"]   = 0
        self.intrinsics["cx"]   = K[0,2]
        self.intrinsics["cy"]   = K[1,2]
        self.intrinsics["w"]    = w
        self.intrinsics["h"]    = h

        self.intrinsics["aabb"] = (2*np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])).tolist() # Computed automatically in to_nerf()?
        self.intrinsics["aabb_scale"] = 1.0

        self.save_intrinsics_file()


    def save_image(self, image, f_name):
        if 'frames' not in self.intrinsics:
            self.intrinsics['frames'] = []
        
        frame = {}
        frame['file_path'] = f_name
        frame['transform_matrix'] = np.eye(4).tolist()
        self.intrinsics['frames'].append(frame)

        cv2.imwrite(os.path.join(self.dataset_dir,frame['file_path']), image)


    
    def load_args(self):

        with open('config/params.yaml') as f:
            args = yaml.safe_load(f)
        args["dataset_dir"] = self.dataset_dir

        args = Struct(**args)

        return args


if __name__ == '__main__':
    project_id = 1
    dataset_dir = sys.argv[1]
    dataset_dir2 = sys.argv[2]
    k = float(sys.argv[3])
    max_num = int(sys.argv[4])
    nerf = NerfSLAM(dataset_dir)
    nerf.convert(dataset_dir2, k, max_num)