#!/usr/bin/env python3

import torch
from lietorch import SE3

import numpy as np
import cv2

from icecream import ic

from utils.flow_viz import *
from utils.utils import *

import os
import sys
import glob
import json

from scipy.optimize import least_squares
# from plyfile import PlyData, PlyElement
from sklearn.utils.random import sample_without_replacement

# Search for pyngp in the build folder.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(
    os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(
    os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]

import pyngp as ngp
import pandas

from utils.utils import *

def round_up_to_base(x, base=10):
    return x + (base - x) % base

def get_marching_cubes_res(res_1d: int, aabb:  ngp.BoundingBox ) -> np.ndarray:
    scale = res_1d / (aabb.max - aabb.min).max()
    res3d = (aabb.max - aabb.min) * scale + 0.5
    res3d = round_up_to_base(res3d.astype(np.int32), 16)
    return res3d


class NerfFusion:
    def __init__(self, name, args, device) -> None:
        ic(os.environ['CUDA_VISIBLE_DEVICES'])

        self.name = name
        self.args = args
        self.device = device

        self.viz = False
        self.kf_idx_to_f_idx = {}
        self.poses = {}
        self.n_kf = 0
        self.T_d_w = None
        self.nerf_scale = 1
        self.prev_pose = np.array([0,0,0,0,0,0,1])

        # self.render_path_i = 0
        import json
        with open(os.path.join(args.dataset_dir, "transforms.json"), 'r') as f:
            self.json = json.load(f)

        output_dir = os.path.join(self.args.dataset_dir,'output')
        K = np.eye(3)
        K[0,0] = self.json["fl_x"]
        K[1,1] = self.json["fl_y"]
        K[0,2] = self.json["cx"]   
        K[1,2] = self.json["cy"]      
        w      = self.json["w"]    
        h      = self.json["h"]               
        intrinsics = {'camera_matrix': K.tolist(), 'width': w, 'height':h}
        with open(os.path.join(output_dir, 'intrinsics.json'), 'w') as f:
            json.dump(intrinsics,f)

        # self.render_path = []
        self.gt_to_slam_scale = 1 # We should be calculating this online.... Sim(3) pose alignment
        # sf = self.json["scale_trans"]
        # aabb = np.array(self.json["aabb"])
        # scale, offset = get_scale_and_offset(aabb)

        # for frame in self.json["frames"]:
        #     c2w = np.array(frame['transform_matrix'])
        #     # c2w[:3,3] /= sf
        #     c2w = nerf_matrix_to_ngp(c2w, scale=scale, offset=offset)
        #     w2c = np.linalg.inv(c2w)
        #     # print(w2c)
        #     self.render_path += [w2c]

        self.iters = 1
        self.iters_if_none = 1
        self.total_iters = 0
        self.stop_iters  = 25000
        self.old_training_step = 0

        mode = ngp.TestbedMode.Nerf
        configs_dir = os.path.join(ROOT_DIR, "thirdparty/instant-ngp/configs", "nerf")

        base_network = os.path.join(configs_dir, "base.json")
        network = args.network if args.network else base_network
        if not os.path.isabs(network):
            network = os.path.join(configs_dir, network)

        self.ngp = ngp.Testbed(mode, 0) # NGP can only use device = 0

        n_images = args.buffer

        if args.eval:
            train_split = 0.8
            test_split = 1 - train_split
            self.test_images_ids = sample_without_replacement(n_images, int(test_split*n_images))
        else:
            self.test_images_ids = []

        aabb_scale = 4
        nerf_scale = 1.0 # Not needed unless you call self.ngp.load_training_data() or you render depths!
        offset = np.array([np.inf, np.inf, np.inf]) # Not needed unless you call self.ngp.load_training_data()
        render_aabb = ngp.BoundingBox(np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf])) # a no confundir amb self.ngp.aabb/raw_aabb/render_aabb
        self.ngp.create_empty_nerf_dataset(n_images, nerf_scale, offset, aabb_scale, render_aabb)

        self.ngp.nerf.training.n_images_for_training = 0;


        if args.gui:
            # Pick a sensible GUI resolution depending on arguments.
            sw = args.width or 1920
            sh = args.height or 1080
            while sw*sh > 1920*1080*4:
                sw = int(sw / 2)
                sh = int(sh / 2)
            self.ngp.init_window(640, 480, second_window=False)

            # Gui params:
            self.ngp.display_gui = True
            self.ngp.nerf.visualize_cameras = True
            self.ngp.visualize_unit_cube = False

        self.ngp.reload_network_from_file(network)

        # NGP Training params:
        self.ngp.shall_train = True
        self.ngp.dynamic_res = True
        self.ngp.dynamic_res_target_fps = 15
        self.ngp.camera_smoothing = True
        #self.ngp.nerf.training.near_distance = 0.2
        #self.ngp.nerf.training.density_grid_decay = 1.0
        self.ngp.nerf.training.optimize_extrinsics = True
        self.ngp.nerf.training.depth_supervision_lambda = 1.0
        self.ngp.nerf.training.depth_loss_type = ngp.LossType.L2
        self.mask_type = args.mask_type # "ours", "ours_w_thresh" or "raw", "no_depth"

        # Keeps track of frame_ids being reconstructed
        self.ref_frames = {}

        self.mesh_renderer = None

        self.anneal = False
        self.anneal_every_iters = 200
        self.annealing_rate = 0.95

        self.evaluate = args.eval
        self.eval_every_iters = 1000
        if self.evaluate:
            self.df = pandas.DataFrame(columns=['Iter', 'Dt','PSNR', 'L1', 'count'])

        # Fit vol once to init gui
        self.fit_volume_once()

    def process_data(self, packet):
        # GROUND_TRUTH Fitting
        self.ngp.nerf.training.optimize_extrinsics = False

        calib = packet["calibs"][0]
        scale, offset = get_scale_and_offset(calib.aabb)
        gt_depth_scale = calib.depth_scale

        packet["poses"]            = scale_offset_poses(np.linalg.inv(packet["poses"]), scale=scale, offset=offset)
        packet["images"]           = (packet["images"].astype(np.float32) / 255.0)
        packet["depths"]           = (packet["depths"].astype(np.float32))
        packet["gt_depths"]        = (packet["depths"].astype(np.float32))
        packet["depth_scale"]      = gt_depth_scale * scale
        packet["depths_cov"]       = np.ones_like(packet["depths"])
        packet["depths_cov_scale"] = 1.0

        # timestamps     = packet["t_cams"]
        # print(timestamps)        

        self.send_data(packet)
        return False

    def process_slam(self, packet):
        # SLAM_TRUTH Fitting

        # No slam output, just fit for some iters
        if not packet:
            print("Missing fusion input packet from SLAM module...")
            return True

        # Slam output is None, just fit for some iters
        slam_packet = packet[1]
        if slam_packet is None:
            print("Fusion packet from SLAM module is None...")
            return True

        if slam_packet["is_last_frame"]:
            return True

        # Get new data and fit volume
        viz_idx        = slam_packet["viz_idx"]
        cam0_T_world   = slam_packet["cam0_poses"]
        images         = slam_packet["cam0_images"]
        idepths_up     = slam_packet["cam0_idepths_up"]
        depths_cov_up  = slam_packet["cam0_depths_cov_up"]
        calibs         = slam_packet["calibs"]
        gt_depths      = slam_packet["gt_depths"]
        kf_idx_to_f_idx = slam_packet["kf_idx_to_f_idx"]
        
        self.kf_idx_to_f_idx = kf_idx_to_f_idx
        for k in self.kf_idx_to_f_idx.keys():
            self.kf_idx_to_f_idx[k] = self.kf_idx_to_f_idx[k].tolist()

        calib = calibs[0]
        scale, offset = get_scale_and_offset(calib.aabb) # if we happen to change aabb, we are screwed...
        gt_depth_scale = calib.depth_scale
        scale = 1.0 # We manually set the scale to 1.0 bcs the automatic get_scale_and_offset sets the scale too small for high-quality recons.
        offset = np.array([0.0, 0.0, 0.0])

        # Mask out depths that have too much uncertainty
        if self.mask_type == "ours":
            pass
        elif self.mask_type == "raw":
            depths_cov_up[...] = 1.0
        elif self.mask_type == "ours_w_thresh":
            masks = (depths_cov_up.sqrt() > depths_cov_up.quantile(0.50))
            idepths_up[masks] = -1.0
        elif self.mask_type == "no_depth":
            idepths_up[...] = -1.0
        else:
            raise NotImplementedError(f"Unknown mask type: {self.mask_type}")

        #TODO: 
        # poses -> matrix
        # images -> [N,H,W,4] float cpu
        # depths -> [N,H,W,1] float cpu up-sampled
        # calibs -> up-sampled
        assert(images.dtype == torch.uint8)
        assert(idepths_up.dtype == torch.float)
        assert(depths_cov_up.dtype == torch.float)

        if self.viz:
            viz_depth_sigma(depths_cov_up.unsqueeze(-1).sqrt(), fix_range=True, bg_img=images, sigma_thresh=20.0, name="Depth Sigma for Fusion")
            cv2.waitKey(1)

        N, _, H, W = images.shape
        alpha_padding = 255 * torch.ones(N, 1, H, W, dtype=images.dtype, device=images.device) # we could avoid this if we didn't remove the alpha channel in the frontend
        images = torch.cat((images, alpha_padding), 1)

        cam0_T_world = SE3(cam0_T_world).matrix().contiguous().cpu().numpy()
        world_T_cam0 = scale_offset_poses(np.linalg.inv(cam0_T_world), scale=scale, offset=offset)
        images = (images.permute(0,2,3,1).float() / 255.0)
        depths = (1.0 / idepths_up[..., None])
        depths_cov = depths_cov_up[..., None]
        gt_depths = gt_depths.permute(0, 2, 3, 1) * gt_depth_scale * scale

        idx = viz_idx.cpu().numpy()           
        for i,k in enumerate(idx):            
            w2c = cam0_T_world[i]
            self.poses[int(k)] = matrix2pose(np.linalg.inv(w2c)).tolist()

        # This is extremely slow.
        # TODO: we could do it in cpp/cuda: send the uint8_t image instead of float, and call srgb_to_linear inside the convert_rgba32 function
        if images.shape[2] == 4:
            images[...,0:3] = srgb_to_linear(images[...,0:3], self.device)
            images[...,0:3] *= images[...,3:4] # Pre-multiply alpha
        else:
            images = srgb_to_linear(images, self.device)

        data_packets = {"k":            viz_idx.cpu().numpy(),
                    "poses":            world_T_cam0,  # needs to be c2w
                    "images":           images.contiguous().cpu().numpy(),
                    "depths":           depths.contiguous().cpu().numpy(),
                    "depth_scale":      scale, # This should be scale, since we scale the poses... # , 1.0, #np.mean(depths), #* self.ngp.nerf.training.dataset.scale,
                    "depths_cov":       depths_cov.contiguous().cpu().numpy(), # do not use up
                    "depths_cov_scale": scale, # , 1.0, #np.mean(depths), #* self.ngp.nerf.training.dataset.scale, 
                    "gt_depths":        gt_depths.contiguous().cpu().numpy(), 
                    "calibs":           calibs,
                }

        # Uncomment in case you want to use ground-truth poses
        # batch["poses"] = np.linalg.inv(gt_poses.cpu().numpy())
        # batch["depths"] = (gt_depths.permute(0,2,3,1).float()).contiguous().cpu().numpy()
        # batch["depth_scale"] = 4.5777065690089265e-05 * self.ngp.nerf.training.dataset.scale
        # gt_pose = np.linalg.inv(gt_poses[0].cpu().numpy()) # c2w gt poses in ngp format

        self.send_data(data_packets)
        return False

    # Main LOOP
    def fuse(self, data_packets):
        fit = False
        if data_packets:  # data_packets is a dict of data_packets
            for name, packet in data_packets.items():
                if name == "data":
                    fit = self.process_data(packet)
                elif name == "slam":
                    fit = self.process_slam(packet)
                    #self.ngp.set_camera_to_training_view(self.ngp.nerf.training.n_images_for_training-1) 
                else:
                    raise NotImplementedError(f"process_{name} not implemented...")
            if fit:
                self.fit_volume()
        else:
            #print("No packet received in fusion module.")
            self.fit_volume()

        # Set the gui to a given pose, and follow the gt pose, but modulate the speed somehow...
        # a) allow to provide a pose (from gt)
        # b) position gui cam there (need the pybind for that) No need! It's camera_matrix!
        # c) allow to speed/slow cam mov (use slerp)
        # TODO: ideally set it slightly ahead!
        #self.render_path_i += 1
        #self.ngp.camera_matrix = self.render_path[self.render_path_i][:3,:]
        return True  # return None if we want to shutdown

    def stop_condition(self):
        return self.total_iters > self.stop_iters if self.evaluate else False

    def send_data(self, batch):
        frame_ids       = batch["k"]
        poses           = batch["poses"]
        images          = batch["images"]
        depths          = batch["depths"]
        depth_scale     = batch["depth_scale"]
        depths_cov      = batch["depths_cov"]
        depth_cov_scale = batch["depths_cov_scale"]
        gt_depths       = batch["gt_depths"]
        calib           = batch["calibs"][0]  # assumes all the same calib

        intrinsics = calib.camera_model.numpy()
        resolution = calib.resolution.numpy()

        focal_length = intrinsics[:2]
        principal_point = intrinsics[2:]

        # TODO: we need to restore the self.ref_frames[frame_id] = [image, gt, etc] for evaluation....
        for i, id in enumerate(frame_ids):
            self.ref_frames[id.item()] = [images[i], depths[i], gt_depths[i], depths_cov[i]]        
            
        frame_ids = list(frame_ids)
        poses = list(poses[:, :3, :4])
        images = list(images)
        depths = list(depths)
        depths_cov = list(depths_cov)

        frame_ids   = [p for i, p in enumerate(frame_ids) if i not in self.test_images_ids]
        poses       = [p for i, p in enumerate(poses) if i not in self.test_images_ids]
        images      = [p for i, p in enumerate(images) if i not in self.test_images_ids]
        depths      = [p for i, p in enumerate(depths) if i not in self.test_images_ids]
        depths_cov  = [p for i, p in enumerate(depths_cov) if i not in self.test_images_ids]        

        self.ngp.nerf.training.update_training_images(frame_ids,
                                                      poses, 
                                                      images, 
                                                      depths, 
                                                      depths_cov, resolution, principal_point, focal_length, depth_scale, depth_cov_scale)

        # On the first frame, set the viewpoint
        if self.ngp.nerf.training.n_images_for_training == 1:
            self.ngp.set_camera_to_training_view(0) 


    def fit_volume(self):
        print(f"Fitting volume for {self.iters} iters")
        self.fps = 30
        for _ in range(self.iters):
            self.fit_volume_once()
            self.ngp.apply_camera_smoothing(1000.0/self.fps)
        # if self.evaluate and self.total_iters % self.eval_every_iters == 0:
        #     self.create_training_views()

    def fit_volume_once(self):
        self.ngp.frame()
        dt = self.ngp.elapsed_training_time
        ic(f"Iter={self.total_iters}; Dt={dt}; Loss={self.ngp.loss}")
        if self.anneal and self.total_iters % self.anneal_every_iters == 0:
            self.ngp.nerf.training.depth_supervision_lambda *= self.annealing_rate
        if self.evaluate and self.total_iters % self.eval_every_iters == 0:
            print("Evaluate.")
            # self.eval_gt_traj()
            self.create_training_views()
        if self.total_iters % 1 == 0 and self.total_iters > 0:
            try:
                output_dir = os.path.join(self.args.dataset_dir,'output')
                if os.path.exists(os.path.join(self.args.dataset_dir, "query_pose.json")):               
                    with open(os.path.join(self.args.dataset_dir, "query_pose.json"), 'r') as f:
                        pose_dict = json.load(f)
                    pose = np.array(pose_dict['pose'])
                    if not np.all(pose == self.prev_pose):
                        self.create_view(pose, self.json["w"], self.json["h"], output_dir)
            except Exception as e:
                ic(e)

        self.total_iters += 1

    def evaluate_depth(self):
        self.mesh_renderer.render_depth()

    def print_ngp_info(self):
        print("NGP Info")
        ic(self.ngp.dynamic_res)
        ic(self.ngp.dynamic_res_target_fps)
        ic(self.ngp.fixed_res_factor)
        ic(self.ngp.background_color)
        ic(self.ngp.shall_train)
        ic(self.ngp.shall_train_encoding)
        ic(self.ngp.shall_train_network)
        ic(self.ngp.render_groundtruth)
        ic(self.ngp.groundtruth_render_mode)
        ic(self.ngp.render_mode)
        ic(self.ngp.slice_plane_z)
        ic(self.ngp.dof)
        ic(self.ngp.aperture_size)
        ic(self.ngp.autofocus)
        ic(self.ngp.autofocus_target)
        ic(self.ngp.floor_enable)
        ic(self.ngp.exposure)
        ic(self.ngp.scale)
        ic(self.ngp.bounding_radius)
        ic(self.ngp.render_aabb)
        ic(self.ngp.render_aabb_to_local)
        ic(self.ngp.aabb)
        ic(self.ngp.raw_aabb)
        ic(self.ngp.fov)
        ic(self.ngp.fov_xy)
        ic(self.ngp.fov_axis)
        ic(self.ngp.zoom)
        ic(self.ngp.screen_center)

    def print_training_info(self):
        print("Training Info")
        ic(self.ngp.nerf.training.n_images_for_training)
        ic(self.ngp.nerf.training.depth_supervision_lambda)

    def print_dataset_info(self):
        print("Dataset Info")
        ic(self.ngp.nerf.training.dataset.render_aabb)
        ic(self.ngp.nerf.training.dataset.render_aabb.min)
        ic(self.ngp.nerf.training.dataset.render_aabb.max)
        ic(self.ngp.nerf.training.dataset.render_aabb_to_local)
        ic(self.ngp.nerf.training.dataset.up)
        ic(self.ngp.nerf.training.dataset.offset)
        ic(self.ngp.nerf.training.dataset.n_images)
        ic(self.ngp.nerf.training.dataset.envmap_resolution)
        ic(self.ngp.nerf.training.dataset.scale)
        ic(self.ngp.nerf.training.dataset.aabb_scale)
        ic(self.ngp.nerf.training.dataset.from_mitsuba)
        ic(self.ngp.nerf.training.dataset.is_hdr)

    def print_dataset_metadata_info(self):
        print("Meta Info")
        metadatas = self.ngp.nerf.training.dataset.metadata
        ic(len(metadatas))
        for metadata in metadatas:
            ic(metadata.focal_length)
            ic(metadata.camera_distortion)
            ic(metadata.principal_point)
            ic(metadata.rolling_shutter)
            ic(metadata.light_dir)
            ic(metadata.resolution)

        ic(self.ngp.nerf.training.dataset.paths[0])
        #ic(self.ngp.nerf.training.dataset.transforms[0].start)
        #ic(self.ngp.nerf.training.dataset.transforms[0].end)


    def compute_scale(self):

        scale_l = []

        for img_id in self.poses.keys():

            pose = np.array(self.poses[img_id])

            # gt pose
            f_idx = self.kf_idx_to_f_idx[img_id] * self.args.img_stride
            t = self.json["frames"][f_idx]
            T_wc = np.array(t["transform_matrix"])
            pose_gt = matrix2pose(T_wc)        

            if img_id == 0:
                pose0 = pose
                pose_gt0 = pose_gt
            else:
                scale_l.append(np.linalg.norm(pose_gt[:3]-pose_gt0[:3]) / np.linalg.norm(pose[:3]-pose0[:3]))

        if len (scale_l) == 0:
            scale = 0
        else:
            scale = np.mean(scale_l)
        # print(scale_l)
        print(f'scale: {scale}')

        return scale        

    def fit_scale_error(self,x,t1,t2):
        s = x[0]
        pose = x[1:]
        T_m1_m2 = pose2matrix(pose, scale=s)
        t2_ = np.hstack([t2, np.ones((t2.shape[0],1))])
        t2_ = (T_m1_m2 @ t2_.T).T
        t2 = t2_[:,:3] / (t2_[:,3].reshape((-1,1)))
        t = t1 - t2
        return np.sum(np.linalg.norm(t,axis=1))

    def fit_scale(self):
        
        poses_d = []
        poses_m1 = []

        for img_id in self.poses.keys():

            pose_d = np.array(self.poses[img_id])

            # gt pose
            f_idx = self.kf_idx_to_f_idx[img_id] * self.args.img_stride
            t = self.json["frames"][f_idx]
            T_wc = nerf_matrix_to_ngp(np.array(t["transform_matrix"]))
            pose_m1 = matrix2pose(T_wc)        
            
            poses_m1.append(pose_m1)
            poses_d.append(pose_d)

        poses_d = np.array(poses_d)
        poses_m1 = np.array(poses_m1)

        t_m1 = poses_m1[:,:3]
        t_d = poses_d[:,:3]
        
        x0 = [1,0,0,0,0,0,0,1]
        res=least_squares(self.fit_scale_error, x0,args=(t_d, t_m1))

        s = res.x[0]
        pose = res.x[1:]
        T_d_m1 = pose2matrix(pose, scale=s)
        # T_d_m1 = pose2matrix(pose, scale=1/s)        
        # T_d_m1 = pose2matrix(pose, scale=1)
        # s = 1        

        # s = self.compute_scale()
        
        return T_d_m1, s

    def create_view(self, pose, w, h, output_dir):

        if len(self.poses) < 7:
            return

        if len(self.poses) > self.n_kf or self.T_d_w is None:
            self.n_kf = len(self.poses)
            self.T_d_w, self.nerf_scale = self.fit_scale()

        
        ic(f'scale: {self.nerf_scale }')

        spp = 1 # samples per pixel
        linear = True
        fps = 20.0

        s = float(w) / 640.0
        w = int(float(w) / s)
        h = int(float(h) / s)

        # Save the state before evaluation
        import copy
        tmp_shall_train = copy.deepcopy(self.ngp.shall_train)
        tmp_background_color = copy.deepcopy(self.ngp.background_color)
        tmp_snap_to_pixel_centers = copy.deepcopy(self.ngp.snap_to_pixel_centers)
        tmp_snap_to_pixel_centers = copy.deepcopy(self.ngp.snap_to_pixel_centers)
        tmp_rendering_min_transmittance = copy.deepcopy(self.ngp.nerf.rendering_min_transmittance)
        tmp_cam = self.ngp.camera_matrix.copy()
        tmp_render_mode = copy.deepcopy(self.ngp.render_mode)

        # Modify the state for evaluation
        self.ngp.background_color = [0.0, 0.0, 0.0, 1.0]
        self.ngp.snap_to_pixel_centers = True
        self.ngp.nerf.rendering_min_transmittance = 1e-4
        self.ngp.shall_train = False  

        T_w_c0 = pose2matrix(pose)
        
        T_link_c0 = pose2matrix([0,0,0,-0.5,0.5,-0.5,0.5])
        # T_link_c0 = pose2matrix([0,0,0,1,0,0,0])
        T_w_link = T_w_c0
        T_w_c0 = T_w_link.dot(T_link_c0)

        T_w_c = nerf_matrix_to_ngp(T_w_c0)
        T_d_c = self.T_d_w @ T_w_c

        self.ngp.camera_matrix = T_d_c[:3,:]
        # self.ngp.set_camera_to_training_view(0)
        # ic(self.ngp.camera_matrix) 

        # Get ref/est RGB images
        self.ngp.render_mode = ngp.Shade
        # ref_image = self.ref_frames[i][0]
        est_image = self.ngp.render(w, h, spp, linear, fps=fps)

        ic(est_image.shape)

        # ref_image_viz = 255*cv2.cvtColor(ref_image, cv2.COLOR_BGRA2RGBA)
        est_image_viz = 255*cv2.cvtColor(est_image, cv2.COLOR_BGRA2RGBA)
        est_image_viz = cv2.rotate(est_image_viz, cv2.ROTATE_180)

        # TODO: Get ref/est Depth images
        self.ngp.render_mode = ngp.Depth
        est_depth = self.ngp.render(w, h, spp, linear, fps=fps)
        est_depth = est_depth[...,0] # The rest of the channels are the same (and last is 1)
        # ref_depth = self.ref_frames[i][2].squeeze()
        # est_to_ref_depth_scale = ref_depth.mean() / est_depth.mean()

        # est_depth_viz = np.array(est_depth * 1000, dtype=np.uint16)
        # est_depth_viz = np.array(est_depth * 1000 / self.nerf_scale, dtype=np.uint16)
        est_depth_viz = np.array(est_depth * 1000 / self.nerf_scale / self.nerf_scale , dtype=np.uint16)

        est_depth_viz = cv2.rotate(est_depth_viz, cv2.ROTATE_180)

        # cv2.imwrite(os.path.join(output_dir,f'ref_image_viz_{i}.jpg'), ref_image_viz)        
        cv2.imwrite(os.path.join(output_dir,f'est_image_viz.jpg'), est_image_viz)        
        cv2.imwrite(os.path.join(output_dir,f'est_depth_viz.png'), est_depth_viz)        
        np.savetxt( os.path.join(output_dir,'T_w_c.txt') , T_w_c0)
        np.savetxt( os.path.join(output_dir,'scale.txt') , np.array(self.nerf_scale).reshape((1,1)))

        # Reset the state
        self.ngp.shall_train                 = tmp_shall_train
        self.ngp.background_color            = tmp_background_color
        self.ngp.snap_to_pixel_centers       = tmp_snap_to_pixel_centers
        self.ngp.snap_to_pixel_centers       = tmp_snap_to_pixel_centers
        self.ngp.nerf.rendering_min_transmittance = tmp_rendering_min_transmittance
        self.ngp.camera_matrix               = tmp_cam
        self.ngp.render_mode                 = tmp_render_mode
        
        # set prev pose
        self.prev_pose = pose

    def marching_cubes(self,output_dir):
        mc = self.ngp.compute_marching_cubes_mesh(resolution=get_marching_cubes_res(512, self.ngp.aabb), aabb=self.ngp.aabb, thresh=2)
        vertex = np.array(list(zip(*mc["V"].T)), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_color = np.array(list(zip(*((mc["C"] * 255).T))), dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        n = len(vertex)
        assert len(vertex_color) == n

        vertex_all = np.empty(n, vertex.dtype.descr + vertex_color.dtype.descr)

        for prop in vertex.dtype.names:
            vertex_all[prop] = vertex[prop]

        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

        ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False)

        ply.write( os.path.join(output_dir,'nerf_pc.ply'))        
        return mc

    def create_training_views(self, output_dir='/datasets/project_1/output'):


        if len(self.poses) < 7:
            return

        if len(self.poses) > self.n_kf or self.T_d_w is None:
            self.n_kf = len(self.poses)
            # depth_scale = self.compute_scale()
            _, depth_scale = self.fit_scale()

        spp = 1 # samples per pixel
        linear = True
        fps = 20.0

        # Save the state before evaluation
        import copy
        tmp_shall_train = copy.deepcopy(self.ngp.shall_train)
        tmp_background_color = copy.deepcopy(self.ngp.background_color)
        tmp_snap_to_pixel_centers = copy.deepcopy(self.ngp.snap_to_pixel_centers)
        tmp_snap_to_pixel_centers = copy.deepcopy(self.ngp.snap_to_pixel_centers)
        tmp_rendering_min_transmittance = copy.deepcopy(self.ngp.nerf.rendering_min_transmittance)
        tmp_cam = self.ngp.camera_matrix.copy()
        tmp_render_mode = copy.deepcopy(self.ngp.render_mode)

        # Modify the state for evaluation
        self.ngp.background_color = [0.0, 0.0, 0.0, 1.0]
        self.ngp.snap_to_pixel_centers = True
        self.ngp.nerf.rendering_min_transmittance = 1e-4
        self.ngp.shall_train = False        

        # stride = 300
        stride = 2

        ic('Creating training views...')
        for i in range(0, self.ngp.nerf.training.n_images_for_training, stride):
        # for i in range(0, len(self.render_path), stride):

            # Use GT trajectory for evaluation to have consistent metrics.

            # self.ngp.camera_matrix = self.render_path[i][:3,:]
            self.ngp.set_camera_to_training_view(i) 

            if len(self.ref_frames) == 0: break
            ref_image = self.ref_frames[0][0]
            h = ref_image.shape[0]
            w = ref_image.shape[1]            

            # Get ref/est RGB images
            self.ngp.render_mode = ngp.Shade
            est_image = self.ngp.render(w, h, spp, linear, fps=fps)
            # ic(est_image.shape)

            est_image_viz = 255*cv2.cvtColor(est_image, cv2.COLOR_BGRA2RGBA)

            # TODO: Get ref/est Depth images
            self.ngp.render_mode = ngp.Depth
            est_depth = self.ngp.render(w, h, spp, linear, fps=fps)
            est_depth = est_depth[...,0] # The rest of the channels are the same (and last is 1)

            est_depth_viz =  np.array(depth_scale*est_depth*1000, dtype=np.uint16)
        
            if not i in self.kf_idx_to_f_idx.keys(): continue
            
            cv2.imwrite(os.path.join(output_dir,f'est_image_viz{i}.jpg'), est_image_viz)        
            cv2.imwrite(os.path.join(output_dir,f'est_depth_viz{i}.png'), est_depth_viz) 

            ic(f'view: {i}')
        
        with open(os.path.join(output_dir, 'poses.json'), 'w') as f:
            json.dump(self.poses,f)

        with open(os.path.join(output_dir, 'keyframes.json'), 'w') as f:
            json.dump(self.kf_idx_to_f_idx,f)            


        # Reset the state
        self.ngp.shall_train                 = tmp_shall_train
        self.ngp.background_color            = tmp_background_color
        self.ngp.snap_to_pixel_centers       = tmp_snap_to_pixel_centers
        self.ngp.snap_to_pixel_centers       = tmp_snap_to_pixel_centers
        self.ngp.nerf.rendering_min_transmittance = tmp_rendering_min_transmittance
        self.ngp.camera_matrix               = tmp_cam
        self.ngp.render_mode                 = tmp_render_mode

    def eval_gt_traj(self):

        output_dir='/datasets/project_1/output'
        # self.marching_cubes(output_dir)

        ic(self.total_iters)

        spp = 1 # samples per pixel
        linear = True
        fps = 20.0

        # Save the state before evaluation
        import copy
        tmp_shall_train = copy.deepcopy(self.ngp.shall_train)
        tmp_background_color = copy.deepcopy(self.ngp.background_color)
        tmp_snap_to_pixel_centers = copy.deepcopy(self.ngp.snap_to_pixel_centers)
        tmp_snap_to_pixel_centers = copy.deepcopy(self.ngp.snap_to_pixel_centers)
        tmp_rendering_min_transmittance = copy.deepcopy(self.ngp.nerf.rendering_min_transmittance)
        tmp_cam = self.ngp.camera_matrix.copy()
        tmp_render_mode = copy.deepcopy(self.ngp.render_mode)

        # Modify the state for evaluation
        self.ngp.background_color = [0.0, 0.0, 0.0, 1.0]
        self.ngp.snap_to_pixel_centers = True
        self.ngp.nerf.rendering_min_transmittance = 1e-4
        self.ngp.shall_train = False

        stride = 2

        # Evaluate
        count = 0
        total_l1 = 0
        total_psnr = 0
        ic(f'test: {len(self.ref_frames) }')

        poses = {}

        assert(len(self.ref_frames) == self.ngp.nerf.training.n_images_for_training)
        for i in range(0, self.ngp.nerf.training.n_images_for_training, stride):
            # Use GT trajectory for evaluation to have consistent metrics.
            self.ngp.set_camera_to_training_view(i) 
            # print(self.ngp.camera_matrix)

            # get pose
            w2c = np.eye(4)
            w2c[:3,:] = self.ngp.camera_matrix            
            c2w = np.linalg.inv(w2c)        
            c2w = ngp_matrix_to_nerf(c2w, scale=self.gt_to_slam_scale, offset=0.0) # THIS multiplies by scale = 1 and offset = 0.5
            pose = matrix2pose(c2w)
            poses[i] = pose.tolist()


            # Get ref/est RGB imageswwwwwwww
            self.ngp.render_mode = ngp.Shade
            ref_image = self.ref_frames[i][0]
            est_image = self.ngp.render(ref_image.shape[1], ref_image.shape[0], spp, linear, fps=fps)

            if self.viz:
                cv2.imshow("Color Error", np.sum(ref_image - est_image, axis=-1))

            # TODO: Get ref/est Depth images
            self.ngp.render_mode = ngp.Depth
            ref_depth = self.ref_frames[i][2].squeeze()
            est_depth = self.ngp.render(ref_image.shape[1], ref_image.shape[0], spp, linear, fps=fps)
            est_depth = est_depth[...,0] # The rest of the channels are the same (and last is 1)

            # Calc metrics
            mse = float(compute_error(est_image, ref_image))
            psnr = mse2psnr(mse)
            total_psnr += psnr

            # Calc L1 metrics
            if self.viz:
                frontend_depth = self.ref_frames[i][1].squeeze()
                depths_cov_up = torch.tensor(self.ref_frames[i][3], dtype=torch.float32, device="cpu")
                viz_depth_sigma(depths_cov_up.unsqueeze(0).sqrt(), fix_range=True,
                                bg_img=torch.tensor(ref_image[...,:3]*255, dtype=torch.uint8, device="cpu").permute(2,0,1).unsqueeze(0),
                                sigma_thresh=20.0, name="Depth Sigma")
                #import matplotlib.pyplot as plt
                #plt.hist(depths_cov_up.view(-1).cpu().numpy(), bins=50, density=False, histtype='barstacked',  # weights=weights_u,
                #        alpha=0.25, color=['steelblue'], edgecolor='none', label='cov', range=[0,256])
                #plt.legend(loc='upper right')
                #plt.xlabel('cov')
                #plt.ylabel('count')
                #plt.draw()
                #plt.pause(1)
                #plt.show()
                viz_depth_map(torch.tensor(frontend_depth, dtype=torch.float32, device="cpu"), fix_range=False, name="Frontend Depth", colormap=cv2.COLORMAP_TURBO, invert=False)
                viz_depth_map(torch.tensor(ref_depth, dtype=torch.float32, device="cpu"), fix_range=False, name="Ref Depth", colormap=cv2.COLORMAP_TURBO, invert=False)
                viz_depth_map(torch.tensor(est_depth, dtype=torch.float32, device="cpu"), fix_range=False, name="Est Depth", colormap=cv2.COLORMAP_TURBO, invert=False)

            est_to_ref_depth_scale = ref_depth.mean() / est_depth.mean()
            ic(est_to_ref_depth_scale)
            diff_depth_map = np.abs(est_to_ref_depth_scale * est_depth - ref_depth)
            diff_depth_map[diff_depth_map > 2.0] = 2.0 # Truncate outliers to 1m, otw biases metric, this can happen either bcs depth is not estimated or bcs gt depth is wrong. 
            if self.viz:
                viz_depth_map(torch.tensor(diff_depth_map), fix_range=False, name="Depth Error", colormap=cv2.COLORMAP_TURBO, invert=False)
            l1 = diff_depth_map.mean() * 100 # From m to cm AND use the mean (as in Nice-SLAM)
            total_l1 += l1
            count += 1

            ref_image_viz = 255*cv2.cvtColor(ref_image, cv2.COLOR_BGRA2RGBA)
            est_image_viz = 255*cv2.cvtColor(est_image, cv2.COLOR_BGRA2RGBA)

            est_depth_viz = np.array(est_depth*1000, dtype=np.uint16)
            # est_depth_viz /= self.json['scale_trans'] # scale back to origin size            

            if not i in self.kf_idx_to_f_idx.keys(): continue
            f_idx = self.kf_idx_to_f_idx[i]*self.args.img_stride
            cv2.imwrite(os.path.join(output_dir,f'est_image_viz{f_idx}.jpg'), est_image_viz)        
            cv2.imwrite(os.path.join(output_dir,f'est_depth_viz{f_idx}.png'), est_depth_viz) 

            if self.viz:
                ref_image_viz = cv2.cvtColor(ref_image, cv2.COLOR_BGRA2RGBA) # Required for Nerf Fusion, perhaps we can put it in there
                est_image_viz = cv2.cvtColor(est_image, cv2.COLOR_BGRA2RGBA) # Required for Nerf Fusion, perhaps we can put it in there
                cv2.imshow("Ref img", ref_image_viz)
                cv2.imshow("Est img", est_image_viz)

            if self.viz:
                cv2.waitKey(1)

        with open(os.path.join(output_dir, 'poses.json'), 'w') as f:
            json.dump(poses,f)
            
            
        dt = self.ngp.elapsed_training_time
        psnr = total_psnr / (count or 1)
        l1 = total_l1 / (count or 1)
        print(f"Iter={self.total_iters}; Dt={dt}; PSNR={psnr}; L1={l1}; count={count}")
        self.df.loc[len(self.df.index)] = [self.total_iters, dt, psnr, l1, count]
        self.df.to_csv("results.csv")

        # Reset the state
        self.ngp.shall_train                 = tmp_shall_train
        self.ngp.background_color            = tmp_background_color
        self.ngp.snap_to_pixel_centers       = tmp_snap_to_pixel_centers
        self.ngp.snap_to_pixel_centers       = tmp_snap_to_pixel_centers
        self.ngp.nerf.rendering_min_transmittance = tmp_rendering_min_transmittance
        self.ngp.camera_matrix               = tmp_cam
        self.ngp.render_mode                 = tmp_render_mode

