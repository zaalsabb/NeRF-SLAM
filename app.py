import logging
import os
import flask
import typing
import shutil
import zipfile
from threading import Thread
import subprocess
import sys
sys.settrace
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import torch
# from torch.multiprocessing import Process

# from datasets.data_module import DataModule
# from gui.gui_module import GuiModule
# from slam.slam_module import SlamModule
# from fusion.fusion_module import FusionModule
import yaml
import io
import tempfile

# from icecream import ic

class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

class NerfSLAM():
    def __init__(self):
        self.fusion_module = None
        self.slam_module = None
        self.data_module = None
        
        args = self.load_params()
        self.run(args)

    def load_params(self):

        with open('config/params.yaml') as f:
            args = yaml.safe_load(f)

        args = Struct(**args)

        # parser = argparse.ArgumentParser(description="Instant-SLAM")

        # # SLAM ARGS
        # parser.add_argument("--parallel_run", action="store_true", help="Whether to run in parallel")
        # parser.add_argument("--multi_gpu", action="store_true", help="Whether to run with multiple (two) GPUs")
        # parser.add_argument("--initial_k", type=int, help="Initial frame to parse in the dataset", default=0)
        # parser.add_argument("--final_k", type=int, help="Final frame to parse in the dataset, -1 is all.", default=-1)
        # parser.add_argument("--img_stride", type=int, help="Number of frames to skip when parsing the dataset", default=1)
        # parser.add_argument("--stereo", action="store_true", help="Use stereo images")
        # parser.add_argument("--weights", default="droid.pth", help="Path to the weights file")
        # parser.add_argument("--buffer", type=int, default=512, help="Number of keyframes to keep")

        # parser.add_argument("--dataset_dir", type=str,
        #                     help="Path to the dataset directory",
        #                     default="/home/tonirv/Datasets/euroc/V1_01_easy")
        # parser.add_argument('--dataset_name', type=str, default='euroc',
        #                     choices=['euroc', 'nerf', 'replica', 'real'],
        #                     help='Dataset format to use.')

        # parser.add_argument("--mask_type", type=str, default='ours', choices=['no_depth', 'raw', 'ours', 'ours_w_thresh'])

        # #parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
        # parser.add_argument("--slam", action="store_true", help="Run SLAM.")
        # parser.add_argument("--fusion", type=str, default='', choices=['tsdf', 'sigma', 'nerf', ''],
        #                     help="Fusion approach ('' for none):\n\
        #                         -`tsdf' classical tsdf-fusion using Open3D\n \
        #                         -`sigma' tsdf-fusion with uncertainty values (Rosinol22wacv)\n \
        #                         -`nerf' radiance field reconstruction using Instant-NGP.")

        # # GUI ARGS
        # parser.add_argument("--gui", action="store_true", help="Run O3D Gui, use when volume='tsdf'or'sigma'.")
        # parser.add_argument("--width",  "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
        # parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

        # # NERF ARGS
        # parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

        # parser.add_argument("--eval", action="store_true", help="Evaluate method.")

        # return parser.parse_args()
        return args

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
            gui_module = GuiModule("NoGui", args, device=cuda_slam) # don't use cuda:1, o3d doesn't work...
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
            # while (gui and not gui_module.shutdown):
            #     continue
            # print("FINISHED RUNNING GUI")

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
            # ok = True
            # while ok:
            #     if gui: ok &= gui_module.spin()
            #     if fusion: ok &= self.fusion_module.spin()

        # Delete everything and clean memory
    

NerfSLAM()


# def load_localizer(project_id: int) -> typing.Tuple[localization.Localizer]:
     
#     home = os.environ.get('HOME')
#     sample_data = os.path.join(f"{home}/datasets", f"project_{project_id}")
#     if not os.path.exists(sample_data):
#         return None
#     l = LocalLoader(sample_data)
#     loc = localization.Localizer(l)

#     return loc

# app.logger.info("API server ready")

# @app.route("/api/v1/project/<int:project_id>/load")
# def load_project(project_id):
#     loc = load_localizer(project_id)
#     if loc is None:
#         return flask.make_response("Project not found", 404)
#     localizers[project_id] = loc
#     intrinsics[project_id] = loc.camera_matrix
#     print(f'Loaded project {project_id}')    
#     return flask.make_response("Project loaded successfully")

# @app.route("/api/v1/project/<int:project_id>/intrinsics", methods=["POST"])
# def add_intrinsics(project_id):
#     if flask.request.method == "POST":
#         if "camera_matrix" in flask.request.json:
#             intrinsics[project_id] = np.array(flask.request.json["camera_matrix"])    
#             return flask.make_response("Query camera intrinsics added successfully")  
#         else:
#             return flask.make_response("Query camera intrinsics not found", 404)
#     else:
#         return flask.make_response("Invalid request", 404)

# @app.route("/api/v1/project/<int:project_id>/localize", methods=["POST"])
# def localize_request(project_id):
#     if project_id not in localizers:
#         return flask.make_response("Project not loaded", 404)
    
#     if flask.request.method == "POST":

#         if flask.request.files.get("camera_matrix"):
#             json_str=flask.request.files["camera_matrix"].read()
#             camera_matrix = np.frombuffer(json_str, dtype="float").reshape(3,3)
#         else:
#             return flask.make_response("Intrinsics not found", 404)

#         if flask.request.files.get("image"):
#             loc = localizers[project_id]

#             img = Image.open(io.BytesIO(flask.request.files["image"].read()))    
#             img = np.array(img)
            
#             T_m1_c2, inliers = loc.callback_query(img, camera_matrix)
#             if T_m1_c2 is None:
#                 res = {'success':False}
#             else:
#                 pose = matrix2pose(T_m1_c2)
#                 res = {'pose':tuple(pose.tolist()), 'inliers':inliers, 'success':True}

#             return flask.make_response(res)

#         else:
#             return flask.make_response("Image not found", 404)
#     else:
#         return flask.make_response("Invalid request", 404)

# @app.route("/api/v1/project/<int:project_id>/localize_multiple", methods=["POST"])
# def localize_multiple_request(project_id):
#     if project_id not in localizers:
#         return flask.make_response("Project not loaded", 404)
    
#     if flask.request.method == "POST":

#         N_imgs = len(flask.request.files) - 2
#         if N_imgs < 1:
#             return flask.make_response("Incorrect number of inputs", 404)
        
#         if flask.request.files.get("camera_matrix"):
#             json_str=flask.request.files["camera_matrix"].read()
#             camera_matrix = np.frombuffer(json_str, dtype="float").reshape(3,3)
#         else:
#             return flask.make_response("Intrinsics not found", 404)        

#         if flask.request.files.get("poses"):
#             json_str=flask.request.files["poses"].read()
#             poses_l = np.frombuffer(json_str, dtype="float").reshape(-1,7)
#             poses_l = poses_l.tolist()
#         else:
#             return flask.make_response("Poses not found", 404)  

#         loc = localizers[project_id]
#         I2_l = []
        
#         for fkey in flask.request.files.keys():
#             if "image" in fkey:            
#                 img = Image.open(io.BytesIO(flask.request.files[fkey].read()))    
#                 img = np.array(img)
#                 I2_l.append(img)

#         T_m1_m2, inliers = loc.callback_query_multiple(I2_l, poses_l, camera_matrix)
#         if T_m1_m2 is None:
#             res = {'success':False}
#         else:
#             pose = matrix2pose(T_m1_m2)
#             res = {'pose':tuple(pose.tolist()), 'inliers':inliers, 'success':True}

#         return flask.make_response(res)

#     else:
#         return flask.make_response("Invalid request", 404)

# # post request method for uploading data to local filesystem for development
# @app.route("/api/v1/project/<int:project_id>/upload", methods=["POST"])
# def upload(project_id):
#     if flask.request.method == "POST":
#         if len(flask.request.data) > 0:

#             with open(os.path.join("/Data.zip"), "wb") as f:
#                 f.write(flask.request.data)

#             home = os.environ.get('HOME')
#             sample_data = os.path.join(home, "datasets", f"project_{project_id}")
#             os.makedirs(sample_data, exist_ok=True)

#             thread = Thread(
#                 target=preprocess_task, args=(sample_data,project_id,)
#             )
#             thread.start()

#             return "success"
#         else:
#             return flask.make_response("Data not found", 404)
#     else:
#         return flask.make_response("Invalid request", 404)

# def preprocess_task(sample_data,project_id):
#     print("started preprocessing...")
#     shutil.rmtree(os.path.join("/Data"), ignore_errors=True)
#     with zipfile.ZipFile("/Data.zip", "r") as zip_ref:
#         zip_ref.extractall("/Data")
#     shutil.rmtree(sample_data, ignore_errors=True)

#     # remove output directory folders if they exist
#     frame_rate = 2
#     max_depth = 5
#     voxel = 0.01
#     # create preprocessor object
#     home = os.environ.get('HOME')
#     process = subprocess.Popen(["python3", "preprocessor/cli.py",
#                                 "-i", "/Data", "-o", sample_data, "-f", str(frame_rate), "-d", str(max_depth), "-v", str(voxel),
#                                 "--mobile_inspector"])
#     process.wait()

#     print('Reloading project...')  
#     loc_1 = load_localizer(project_id) 
#     loc_1.build_database()
#     localizers[project_id] = loc_1
#     intrinsics[project_id] = loc_1.camera_matrix     
#     print(f'Loaded project {project_id}')    

# if __name__ == "__main__":
#     #app.run(host='0.0.0.0',port=5000)
#     init_ip_address()
#     app.run(host='::',port=5000, debug=True)    
