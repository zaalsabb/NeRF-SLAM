import os
from threading import Thread
import sys
import yaml
import json
import numpy as np
import cv2
from utils.utils import *

from icecream import ic

sys.settrace
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    from torch.multiprocessing import Process

    from datasets.data_module import DataModule
    from gui.gui_module import GuiModule
    from slam.slam_module import SlamModule
    from fusion.fusion_module import FusionModule
except:
    print('cannot import modules..')

# torch.multiprocessing.set_start_method('spawn')
# torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True
# torch.set_grad_enabled(False)


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
        self.d = np.zeros(5)
        self.poses_t = []

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir,'images'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir,'output'), exist_ok=True)

        self.fusion_module = None
        self.slam_module = None
        self.data_module = None    

        self.output_dir = os.path.join(self.dataset_dir,'output')

    def run_nerf(self):
        args = self.load_args()
        # torch.multiprocessing.set_start_method('spawn')
        # torch.backends.cudnn.benchmark = True
        # torch.set_grad_enabled(False)        
        # torch.cuda.empty_cache()
        self.run(args)        


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
        self.d[0]   = self.intrinsics["k1"]
        self.d[1]   = self.intrinsics["k2"]
        self.d[2]   = self.intrinsics["p1"]
        self.d[3]   = self.intrinsics["p2"]

    def set_intrinsics(self, K, w, h, d):

        self.K = K
        self.w = w
        self.h = h

        self.intrinsics["fl_x"] = K[0,0]
        self.intrinsics["fl_y"] = K[1,1]
        self.intrinsics["k1"]   = d[0]
        self.intrinsics["k2"]   = d[1]
        self.intrinsics["p1"]   = d[2]
        self.intrinsics["p2"]   = d[3]
        self.intrinsics["cx"]   = K[0,2]
        self.intrinsics["cy"]   = K[1,2]
        self.intrinsics["w"]    = w
        self.intrinsics["h"]    = h

        self.intrinsics["aabb_scale"] = 1.0

        self.save_intrinsics_file()


    def save_image(self, image, depth, pose, k, save_intrinsics=False):
        if 'frames' not in self.intrinsics:
            self.intrinsics['frames'] = []
        
        self.poses_t.append(pose)
        if len(self.poses_t) > 1: 
            delta_t = 1.0 # 1 meter extra to allow for the depth of the camera
            poses_t = np.array(self.poses_t)
            t_max = np.amax(poses_t, 0).flatten()
            t_min = np.amin(poses_t, 0).flatten()
            self.intrinsics["aabb"] = np.array([t_min-delta_t, t_max+delta_t]).tolist()
        else:
            self.intrinsics["aabb"]  = np.array([[-2, -2, -2], [2, 2, 2]]).tolist() # Computed automatically

        frame = {}
        frame['file_path'] = f"images/frame{k:05}.png"
        if depth is not None:
            frame['depth_path'] = f"images/depth{k:05}.png"

        frame['transform_matrix'] = pose2matrix(pose).tolist()
        self.intrinsics['frames'].append(frame)

        cv2.imwrite(os.path.join(self.dataset_dir,frame['file_path']), cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        if depth is not None:
            h = image.shape[0]
            w = image.shape[1]
            depth = np.array(depth, dtype=np.uint16)
            depth = cv2.resize(depth, (w,h), cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.dataset_dir,frame['depth_path']), depth)

        # ic(self.intrinsics)
        if save_intrinsics:
            self.save_intrinsics_file()  

    
    def load_args(self):

        with open('config/params.yaml') as f:
            args = yaml.safe_load(f)
        args["dataset_dir"] = self.dataset_dir

        args = Struct(**args)

        return args

    
    def create_view(self, pose):
        self.load_intrinsics_file()
        self.fusion_module.fusion.create_view(pose, self.w, self.h, self.output_dir)
    
    def run(self, args):
        if args.parallel_run and args.multi_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
            cpu = 'cpu'
            cuda_slam = 'cuda:0'
            cuda_fusion = 'cuda:1' # you can also try same device as in slam.
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            cpu = 'cpu'
            cuda_slam = cuda_fusion = 'cuda:0'
        print(f"Running with GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

        if not args.parallel_run:
            from queue import Queue
        else:
            from torch.multiprocessing import Queue

        # Create the Queue object
        data_for_viz_output_queue = Queue()
        data_for_fusion_output_queue = Queue()
        data_output_queue = Queue()
        slam_output_queue_for_fusion = Queue()
        slam_output_queue_for_o3d = Queue()
        fusion_output_queue_for_gui = Queue()
        gui_output_queue_for_fusion = Queue()

        # Create Dataset provider
        self.data_provider_module = DataModule(args.dataset_name, args, device=cpu)

        # Create MetaSLAM pipeline
        # The SLAM module takes care of creating the SLAM object itself, to avoid pickling issues
        # (see initialize_module() method inside)
        slam = args.slam
        if slam:
            self.slam_module = SlamModule("VioSLAM", args, device=cuda_slam)
            self.data_provider_module.register_output_queue(data_output_queue)
            self.slam_module.register_input_queue("data", data_output_queue)

        # Create Neural Volume
        fusion = args.fusion != ""
        if fusion:
            self.fusion_module = FusionModule(args.fusion, args, device=cuda_fusion)
            if slam:
                self.slam_module.register_output_queue(slam_output_queue_for_fusion)
                self.fusion_module.register_input_queue("slam", slam_output_queue_for_fusion)
            
            if (args.fusion == 'nerf' and not slam) or (args.fusion != 'nerf' and args.eval):
                # Only used for evaluation, or in case we do not use slam (for nerf)
                self.data_provider_module.register_output_queue(data_for_fusion_output_queue)
                self.fusion_module.register_input_queue("data", data_for_fusion_output_queue)


        # Create interactive Gui
        gui = args.gui and args.fusion != 'nerf' # nerf has its own gui
        print(f'fusion: {fusion}')
        if gui:        
            gui_module = GuiModule("Open3DGui", args, device=cuda_slam) # don't use cuda:1, o3d doesn't work...
            self.data_provider_module.register_output_queue(data_for_viz_output_queue)
            if slam:
                self.slam_module.register_output_queue(slam_output_queue_for_o3d)
            gui_module.register_input_queue("data", data_for_viz_output_queue)
            gui_module.register_input_queue("slam", slam_output_queue_for_o3d)
            if fusion and (self.fusion_module.name == "tsdf" or self.fusion_module.name == "sigma"):
                self.fusion_module.register_output_queue(fusion_output_queue_for_gui)
                gui_module.register_input_queue("fusion", fusion_output_queue_for_gui)
                gui_module.register_output_queue(gui_output_queue_for_fusion)
                self.fusion_module.register_input_queue("gui", gui_output_queue_for_fusion)

        # Run
        if args.parallel_run:
            print("Running pipeline in parallel mode.")

            data_provider_thread = Process(target=self.data_provider_module.spin, args=())
            if fusion: fusion_thread = Process(target=self.fusion_module.spin) # FUSION NEEDS TO BE IN A PROCESS
            #if slam: slam_thread = Process(target=slam_module.spin, args=())
            if gui: gui_thread = Process(target=gui_module.spin, args=())

            data_provider_thread.start()
            if fusion: fusion_thread.start()
            #if slam: slam_thread.start()
            if gui: gui_thread.start()

            # Runs in main thread
            if slam: 
                self.slam_module.spin() # visualizer should be the main spin, but pytorch has a memory bug/leak if threaded...
                self.slam_module.shutdown_module()
                ic("Deleting SLAM module to free memory")
                torch.cuda.empty_cache()
                # slam_module.slam. # add function to empty all matrices?
                del self.slam_module
            print("FINISHED RUNNING SLAM")
            while (fusion and fusion_thread.exitcode == None):
                continue
            print("FINISHED RUNNING FUSION")
            while (gui and not gui_module.shutdown):
                continue
            print("FINISHED RUNNING GUI")

            # This is not doing what you think, because Process has another module
            if gui: gui_module.shutdown_module()
            if fusion: self.fusion_module.shutdown_module()
            self.data_provider_module.shutdown_module()

            if gui: gui_thread.terminate() # violent, should be join()
            #if slam: slam_thread.terminate() # violent, should be join()
            if fusion: fusion_thread.terminate() # violent, should be join()
            data_provider_thread.terminate() # violent, should be a join(), but I don't know how to flush the queue
        else:
            print("Running pipeline in sequential mode.")

            # Initialize all modules first (and register 3D volume)
            # print(f'test:{data_provider_module.spin()}')
            if self.data_provider_module.spin() \
                and (not slam or self.slam_module.spin()) \
                and (not fusion or self.fusion_module.spin()):                
                    if gui:
                        gui_module.spin()
                        #gui_module.register_volume(fusion_module.fusion.volume)

            # Run sequential, dataprovider fills queue and gui empties it
            while self.data_provider_module.spin() \
                and (not slam or self.slam_module.spin()) \
                and (not fusion or self.fusion_module.spin()) \
                and (not gui or gui_module.spin()):
                continue

            # # Then gui runs indefinitely until user closes window
            ok = True
            while ok:
                if gui: ok &= gui_module.spin()
                if fusion: ok &= self.fusion_module.spin()

        # Delete everything and clean memory

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)    

    project_id = 1
    dataset_dir = os.path.join(f"/datasets", f"project_{project_id}")
    nerf = NerfSLAM(dataset_dir)
    nerf.run_nerf()