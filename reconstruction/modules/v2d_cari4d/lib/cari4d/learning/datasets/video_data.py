# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import os
import joblib
from os import path as osp
import torch
import cv2, time
import numpy as np
from glob import glob
import h5py
import trimesh
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy
import Utils
import kornia
from behave_data.const import _sub_gender, get_test_view_id
import learning.datasets.augmentations as augm
from render_h5_codec import load_pickled_dataset


def build_video_input_augmenter(H_ori=224, W_ori=224, *, depth_corruption_probability=None):
    augmentations = [
        augm.RGBGaussianNoise(max_noise=15, prob=0.3),
        augm.ChangeBright(mag=[0.5, 1.5], prob=0.5, augment_imgA=True),
        augm.ChangeContrast(mag=[0.8, 1.2], prob=0.5),
        augm.GaussianBlur(max_kernel_size=7, min_kernel_size=3, sigma_range=(0, 3.0), prob=0.3),
        augm.JpegAug(prob=0.5, compression_range=[0, 20]),
    ]
    operation_probability = 0.5 if depth_corruption_probability is None else 1.0
    depth_augmentations = augm.ComposedAugmenter([
        augm.DepthCorrelatedGaussianNoise(prob=operation_probability, H_ori=H_ori, W_ori=W_ori, noise_range=[0, 0.01], rescale_factor_min=2, rescale_factor_max=10),
        augm.DepthMissing(prob=operation_probability, H_ori=H_ori, W_ori=W_ori, max_missing_percent=0.5, down_scale=1),
        augm.DepthRoiMissing(prob=operation_probability, H_ori=H_ori, W_ori=W_ori, max_missing_ratio=0.5, downscale_range=[0.1, 1]),
        augm.DepthEllipseMissing(prob=operation_probability, H_ori=H_ori, W_ori=W_ori, max_num_ellipse=20, max_radius=30),
    ])
    if depth_corruption_probability is None:
        augmentations.extend(depth_augmentations.augs)
    else:
        augmentations.append(augm.RandomApply(depth_augmentations, prob=depth_corruption_probability))
    return augm.ComposedAugmenter(augmentations)


def augment_observed_video_components(input_data, augmenter):
    data_temp = {'depth': input_data['xyzB'][:, :, -1].copy(), 'seg': input_data['xyzB'][:, :, -1] > 0, 'rgb': input_data['rgbmB'][:, :, :3].copy()}
    augmented = augmenter(data_temp)
    return augmented['depth'], augmented['rgb']


def augment_observed_video_input(input_data, K_roi, augmenter, *, augment_depth):
    augmented_depth, augmented_rgb = augment_observed_video_components(input_data, augmenter)
    xyz = Utils.depth2xyzmap(augmented_depth, K_roi) if augment_depth else input_data['xyzB']
    return xyz, augmented_rgb


class VideoDataset(Dataset):
    def __init__(self, cfg, seqs, split='val'):
        super().__init__()
        self.cfg = cfg
        self.render_h5_root = cfg['render_root']
        self.split = split

        if self.cfg.nlf_root is not None:
            from lib_smpl import get_smpl
            from lib_smpl.body_landmark import BodyLandmarks
            from lib_smpl.const import SMPL_ASSETS_ROOT

            landmark = BodyLandmarks(SMPL_ASSETS_ROOT)
            self.smpl_male = get_smpl('male', hands=True).cuda()
            self.smpl_female = get_smpl('female', hands=True).cuda()

        # load parameters
        obj_angles, obj_transls = [], []
        data_paths, start_inds, seq_index = [], [], []  # all paths, and index to start image
        joints_body, poses_body, joints_smpl, transl_smpl, betas_gt = [], [], [], [], []
        dists_h2o = []
        occ_ratios = []
        kinect_inds = []
        fp_errors = []
        nlf_poses, nlf_betas, nlf_trans, nlf_verts, nlf_joints, nlf_betas = [], [], [], [], [], []

        exclude_frames = [] if cfg.exclude_frames is None else set(json.load(open(cfg.exclude_frames))['frames'])
        self.exclude_frames = exclude_frames
        exclude_count = 0
        exclude_views = set(json.load(open('data/train/exclude-views.json', 'r'))['exclude'])

        offset = 0
        clip_len, window = cfg['clip_len'], cfg['window']
        kids = range(4) if self.cfg.job == 'train' else [cfg.cam_id]
        print(f'Loading video data with kinects: {kids}')
        genders = []
        
        for sid, seq in enumerate(tqdm(seqs)):
            if not osp.isfile(osp.join(self.render_h5_root, f'{seq}_render.h5')):
                print(f'no render h5 file for {seq}...skipping')
                continue
            # try to open the h5 file with r mode, if fail, skip it
            try:
                _ = h5py.File(osp.join(self.render_h5_root, f'{seq}_render.h5'), 'r')
            except Exception as e:
                print(f'Opening render file failed for {seq}...skipping')
                continue
            pack_file = osp.join(cfg['packed_root'], f'{seq}_GT-packed.pkl')
            packed = joblib.load(pack_file)
            L = len(packed['frames'])

            nlf_joints_seq = []
            nlf_file =f'{cfg.nlf_root}/{seq}_params.pkl'
            if not osp.isfile(nlf_file):
                print(f'{nlf_file} does not exist, skipping')
                continue
            nlf_data = joblib.load(nlf_file)

            # get test view id 
            if self.cfg.job == 'test' and self.cfg.use_sel_view:
                view_id = get_test_view_id(seq)
                kids = [view_id] if view_id is not None else [cfg.cam_id]
                print(f'Using test views: {kids} for {seq}')

            for k in kids:
                if f'{seq}/k{k}' in exclude_views and split == 'train':
                    print("Skipping {} k{} due to large translation error/missing masks!".format(seq, k))
                    continue
                for i in range(0, L - clip_len + 1, window):
                    start_inds.append(i + offset)
                    seq_index.append(sid)
                    kinect_inds.append(k)
                offset += L
                obj_angles.append(packed['obj_angles'].copy())
                obj_transls.append(packed['obj_trans'].copy())
                joints_body.append(packed['joints_body'].copy())
                poses_body.append(packed['poses'].copy())
                transl_smpl.append(packed['trans'].copy())
                betas_gt.append(packed['betas'].copy())
                joints_smpl.append(packed['joints_smpl'].copy()) # (T, 52, 3)
                data_paths.extend([osp.join(seq, x) for x in packed['frames']])
                occ_ratios.append(packed['occ_ratios'][:, k].copy() if 'occ_ratios' in packed else np.ones((len(packed['frames']),))) # (L, 4)
                dists_h2o.append(packed['dists_h2o'][0].copy()) # (L, 52)

                fp_errors.append(np.zeros((len(packed['frames']),))) # dummy data

                # Add NLF data
                if cfg.nlf_root is not None:
                    assert len(nlf_data['frames']) == len(packed['frames']), f'inconsistent number of frames {len(nlf_data["frames"])}!={len(packed["frames"])} on {seq}!'
                    poses = nlf_data['poses'][:, k].copy()
                    transls = nlf_data['transls'][:, k].copy()
                    betas_nlf = nlf_data['betas'][:, k].copy()
                    betas = packed['betas'].astype(np.float32) # Use GT betas
                    betas_avg = np.mean(betas, axis=0)[None].repeat(len(poses), axis=0)
                    betas_avg_nlf = np.mean(betas_nlf, axis=0)[None].repeat(len(poses), axis=0)
                    gender = nlf_data['gender'] if 'gender' in nlf_data else 'neutral'
                    body_model = self.smpl_male if gender == 'male' else self.smpl_female

                    if 'joints_25' not in nlf_data:
                        verts = body_model(torch.from_numpy(poses).cuda(),
                                        torch.from_numpy(betas_avg).cuda(),
                                        torch.from_numpy(transls).cuda()
                                        )[0].cpu().numpy()
                        jts = landmark.get_body_kpts_batch(verts)
                        nlf_joints_seq.append(jts)
                    else:
                        jts = nlf_data['joints_25'][:, k].copy() # this is significantly faster!
                    nlf_joints.append(jts)
                    nlf_poses.append(poses)
                    nlf_betas.append(betas_avg)
                    nlf_trans.append(transls)
                    nlf_betas.append(betas_avg_nlf)
            
            if 'joints_25' not in nlf_data and cfg.job == 'train': # only allow training to save it, so that it computes all joints
                # add to it and save
                nlf_data['joints_25'] = np.stack(nlf_joints_seq, axis=1)
                if nlf_data['joints_25'].shape[1] == len(nlf_data['kids']):
                    joblib.dump(nlf_data, f'{cfg.nlf_root}/{seq}_params.pkl') # save it so next time no need to compute it again
                    print(f'Saved joints_25 for {seq}!')
        self.data_paths = data_paths
        self.start_inds = start_inds
        self.seq_indices = seq_index
        self.kinect_inds = kinect_inds
        self.seqs = seqs
        print(f'dataset setting: clip_len={cfg.clip_len}, window={cfg.window}, total data examples: {len(self.start_inds)}')
        print(f'{exclude_count} data examples are excluded!')

        self.obj_angles = np.concatenate(obj_angles, axis=0)
        self.obj_transls = np.concatenate(obj_transls, axis=0)
        self.joints_body = np.concatenate(joints_body, axis=0)
        self.poses_body = np.concatenate(poses_body, axis=0) # TODO: transform this to local coordinate as well
        self.betas_gt = np.concatenate(betas_gt, axis=0)
        self.joints_smpl = np.concatenate(joints_smpl, axis=0)
        self.transl_smpl = np.concatenate(transl_smpl, axis=0)
        self.occ_ratios = np.concatenate(occ_ratios, axis=0)
        self.dists_h2o = np.concatenate(dists_h2o, axis=0)
        self.fp_errors = np.concatenate(fp_errors, axis=0)
        self.clip_len, self.window = clip_len, window
        if cfg.nlf_root is not None:
            self.nlf_joints = np.concatenate(nlf_joints, axis=0)
            self.nlf_poses = np.concatenate(nlf_poses, axis=0)
            self.nlf_betas = np.concatenate(nlf_betas, axis=0)
            self.nlf_trans = np.concatenate(nlf_trans, axis=0)
            self.nlf_betas = np.concatenate(nlf_betas, axis=0)
            print("NLF data loading done!")
        # init readers as empty, and lazy read in get_item
        self.rgb_h5_handles = {}
        self.render_h5_handles = {}

        fx, fy = 979.7844, 979.840  # for original kinect coordinate system
        cx, cy = 1018.952, 779.486
        self.focal = np.array([fx, fy])
        self.principal_point = np.array([cx, cy])
        factor = 2
        self.K_half = np.array([[fx/factor, 0, cx/factor],
                                [0, fy/factor, cy/factor],
                                [0, 0, 1.]])
        factor = 1
        self.K_full_old = np.array([[fx / factor, 0, cx / factor],
                                [0, fy / factor, cy / factor],
                                [0, 0, 1.]]) # this should never be used
        self.input_resize = cfg['input_resize']

        assert self.cfg.pose_init_type in ['copy-first', 'random', 'fp-pred', 'random-keyframes']
        print('pose_init_type:', self.cfg.pose_init_type)
        print(f'render from {self.render_h5_root}, NLF root: {cfg.nlf_root}')

        # For debug, pre-load object templates
        from behave_data.utils import load_templates_all
        from behave_data.const import BEHAVE_ROOT
        templates_all = load_templates_all(BEHAVE_ROOT, orig=False) if not cfg.wild_video else {} 
        templates_meta = {}
        for k, mesh in templates_all.items():
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            templates_meta[k] = {'bbox': bbox, 'to_origin': to_origin, 'extents': extents, 'mesh': mesh}
        self.obj_templates = templates_meta
        self.pose_pred_dir = cfg.pose_pred_dir

        # back to cpu
        if cfg.nlf_root is not None:
            self.smpl_male = self.smpl_male.to('cpu')
            self.smpl_female = self.smpl_female.to('cpu')
        
        # load symmetry info 
        symm_info = joblib.load('data/train/behave-symmetries.pkl')
        self.max_symms = max([len(v) for k, v in symm_info.items()]) # maximum number of symmetries for any object
        symm_full = {}
        # for objects with less symmetries, fill with identity matrix
        for k, v in symm_info.items():
            symm_full[k] = np.eye(4)[None].repeat(self.max_symms, 0)
            symm_full[k][:len(v)] = v
        self.symm_full = symm_full

        self.augm_img = build_video_input_augmenter()

    def __len__(self):
        return len(self.start_inds)

    def get_chunk_files(self, idx):
        """
        get the list of files for this chunk in the batch
        Parameters
        ----------
        idx :

        Returns
        -------

        """
        start, end = self.start_inds[idx], self.start_inds[idx] + self.clip_len
        image_files = deepcopy(self.data_paths[start:end])
        self.check_paths(image_files)
        return image_files # format: seq_name/frame_time

    def check_paths(self, image_files):
        "sanity check, make sure all images are from the same sequence, same kinect"
        seq_name, frame = image_files[0].split(os.sep)
        for file in image_files:
            seq, f = file.split(os.sep)
            assert seq == seq_name, f'{file} seq name incompatible with {image_files[0]}'

    def __getitem__(self, idx):
        try:
            data_dict = self.get_item(idx)
            return data_dict
        except Exception as e:
            print(f'failed on loading {self.seqs[self.seq_indices[idx]]} {idx} due to {e}')
            ridx = np.random.randint(0, len(self))
            return self[ridx]

    def get_item(self, idx):
        time_start = time.time()
        seq_name = self.seqs[self.seq_indices[idx]]
        if seq_name not in self.render_h5_handles:
            self.render_h5_handles[seq_name] = h5py.File(osp.join(self.render_h5_root, f'{seq_name}_render.h5'), 'r')
        frames = self.get_chunk_files(idx)
        kid = self.kinect_inds[idx]

        # add random flip if training
        indices = np.arange(self.clip_len)
        if self.cfg.job == 'train' and np.random.uniform() > 0.5 and self.cfg.random_flip:
            indices = indices[::-1] # flip all examples in the batch
        frames = [frames[i] for i in indices]

        fp_err_chunk = self.fp_errors[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[indices]
        fp_err_chunk = (np.pi - fp_err_chunk)/np.pi
        visibility = self.occ_ratios[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[indices] # 1 is fully visible, 0 is occluded

        input_rgbs, depth_xyz = [], []
        render_rgbs, render_xyz = [], []
        w2cs = load_pickled_dataset(self.render_h5_handles[seq_name][seq_name + '_w2c'])
        w2c_k = np.eye(4)
        w2c_k[:3, :3] = w2cs['rot'][kid]
        w2c_k[:3, 3] = w2cs['trans'][kid]
        mesh_diameter = w2cs['mesh_diameter']
        rot_normalizer = w2cs['rot_normalizer'] # a deg2rad is needed
        trans_normalizer = w2cs['trans_normalizer']
        # now the rendered RGB, take first frame as init
        seq, fname = frames[0].split(os.sep)
        frame_key = f'{seq_name}+{fname}'
        pid = 0  # use the first one with min perturbation, TODO: use more sophisticated pose
        render_h5_key = frame_key + f'_k{kid}_perturb_{pid}'
        bbox_render, dmap_xyz_init, pose_init, rgb_render, render_data = self.extract_render_data(render_h5_key, seq, mesh_diameter)
        L = len(frames)
        angles_chunk, transls_chunk = self.obj_angles[self.start_inds[idx]:self.start_inds[idx] + L][indices].copy(), self.obj_transls[
                                                                                                      self.start_inds[
                                                                                                          idx]:
                                                                                                      self.start_inds[
                                                                                                          idx] + L][indices].copy()
        # now the poses
        poses_gt, poses_perturbed = [], []
        delta_transl, delta_rot = [], []  # disentangled transl and rot
        K_rois = []
        obj = seq_name.split('_')[2]

        # human info
        joints_body = [np.matmul(self.joints_body[i], w2c_k[:3, :3].T) + w2c_k[:3, 3] for i in range(idx, idx + L)]

        # add human info
        pre_dict = {}
        nlf_transl = None
        if self.cfg.nlf_root is not None:
            nlf_transl = self.nlf_trans[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[indices].copy()
            nlf_poses = self.nlf_poses[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[indices].copy() # (T, 72)
            nlf_joints = self.nlf_joints[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[indices].copy()

            J = 24
            gt_rot_smpl = self.poses_body[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[indices] # (T, 156)
            gt_smpl_trans = self.transl_smpl[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[indices]
            smpl_root_jtrs = self.joints_smpl[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[:, 0][indices]
            # Compute new pose due to w2c
            global_rot = R.from_rotvec(gt_rot_smpl[:, :3]).as_matrix()
            new_rot = np.stack([np.matmul(w2c_k[:3, :3], r) for r in global_rot], 0)
            gt_rot_smpl[:, :3] = R.from_matrix(new_rot).as_rotvec()
            joints_cent = smpl_root_jtrs - gt_smpl_trans
            gt_smpl_trans = np.matmul(gt_smpl_trans, w2c_k[:3, :3].T) + w2c_k[:3, 3] + np.matmul(joints_cent, w2c_k[:3, :3].T) - joints_cent

            gt_rot_smpl_mat = R.from_rotvec(gt_rot_smpl.reshape(-1, 3)).as_matrix()
            assert nlf_poses.shape[-1] == 156, f'NLF poses should have 156 joints, but got {nlf_poses.shape[-1]}!'
            nlf_rot = R.from_rotvec(nlf_poses.reshape(-1, 3)).as_matrix().reshape((self.clip_len, -1, 3, 3))[:, :J].reshape((-1, 3, 3)) # (TJ, 3, 3)

            # compute delta: from nlf to GT
            delta_smpl_rot = np.matmul(gt_rot_smpl_mat.reshape((self.clip_len, -1, 3, 3))[:, :J].reshape((-1, 3, 3)), nlf_rot.transpose(0, 2, 1))
            delta_smpl_trans = gt_smpl_trans - nlf_transl
            delta_smpl_rot = delta_smpl_rot.reshape((self.clip_len, -1, 3, 3))[:, :J] 

            # Compute GT SMPL joints
            t0 = time.time()
            betas_gt = self.betas_gt[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy().astype(np.float32)[indices].copy() # need to copy again to avoid negative stride tensors
            betas_nlf = self.nlf_betas[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy().astype(np.float32)[indices].copy() # need to copy again to avoid negative stride tensors
            gender = _sub_gender[seq_name.split('_')[1]]
            body_model = self.smpl_male if gender == 'male' else self.smpl_female
            jts_smpl_gt = body_model.get_joints(torch.from_numpy(gt_rot_smpl).float(),
                                         torch.from_numpy(betas_gt).float(), torch.from_numpy(gt_smpl_trans).float()) # [:, :J] #
            t1 = time.time()

            pre_dict = {
                'delta_smpl_trans': delta_smpl_trans.astype(np.float32).copy(), # do copy to avoid negative stride tensors
                'delta_smpl_rot': delta_smpl_rot.astype(np.float32).copy(),
                'joints_nlf': nlf_joints.astype(np.float32).copy(),
                'nlf_rotmat': nlf_rot.astype(np.float32).copy(),
                'nlf_poses': nlf_poses.astype(np.float32).copy(),
                'nlf_transl': nlf_transl,
                'betas_gt': betas_gt,
                'betas_nlf': betas_nlf,
                'smpl_poses_gt': gt_rot_smpl.copy().astype(np.float32),
                'smpl_rotmat_gt': gt_rot_smpl_mat.astype(np.float32).reshape(self.clip_len, -1, 3, 3).copy(),
                'smpl_transl_gt': gt_smpl_trans.astype(np.float32).copy(),
                'smpl_jtrs_gt': jts_smpl_gt,
                'is_male': np.array([gender == 'male']*len(jts_smpl_gt)),
                'frame_indices': indices.copy(), # this needs to be copied to avoid negative stride tensors
                'contact_dist_gt': self.dists_h2o[self.start_inds[idx]:self.start_inds[idx] + self.clip_len].copy()[indices].copy().astype(np.float32),
            }

        for i, frame in enumerate(frames):
            seq, fname = frame.split(os.sep)
            frame_key = f'{seq_name}+{fname}'
            # always use the first frame one
            if self.cfg.pose_init_type == 'copy-first':
                # do nothing
                pass
            elif self.cfg.pose_init_type == 'random-keyframes':
                h5_key = frame_key + f'_k{kid}_perturb_{pid}'
                assert h5_key in self.render_h5_handles[seq_name], f'{h5_key} not in {seq_name}!'
                render_h5_key = frame_key + f'_k{kid}_perturb_{pid}'
                bbox_render, dmap_xyz_init, pose_init, rgb_render, render_data = self.extract_render_data(render_h5_key, seq, mesh_diameter)

            h5_mix_key = f'{seq}+{fname}_k{kid}_input'
            input_data = load_pickled_dataset(self.render_h5_handles[seq][h5_mix_key])
            dmap_xyz, dmap_xyz_a, rgb = self.process_input(dmap_xyz_init, i, input_data, mesh_diameter, nlf_transl,
                                                           pose_init, render_data, rgb_render, frame, kid)

            render_rgbs.append(rgb_render.copy().transpose(2, 0, 1) / 255.)
            poses_perturbed.append(pose_init.copy())
            # GT pose
            rot = R.from_rotvec(angles_chunk[i]).as_matrix()
            t = transls_chunk[i]
            pose_gt = np.eye(4)
            pose_gt[:3, :3] = rot
            pose_gt[:3, 3] = t
            # add w2c transform
            pose_gt = np.matmul(w2c_k, pose_gt)
            poses_gt.append(pose_gt)

            input_rgbs.append(rgb)
            depth_xyz.append(dmap_xyz)
            render_xyz.append(dmap_xyz_a)
            delta_transl.append(pose_gt[:3, 3] - pose_init[:3, 3])
            delta_rot.append(np.matmul(pose_gt[:3, :3], pose_init[:3, :3].T))
            K_rois.append(render_data['K_roi']) # 3x3 matrix

        time_end = time.time()
        obj = seq_name.split('_')[2]

        poseA_norm = np.stack(poses_perturbed).copy()
        poseA_norm[:, :3, 3] *= 2 / mesh_diameter
        assert self.cfg['normalize_xyz'], 'must normalize xyz!'

        frame_mask = np.ones((L,)) if len(self.exclude_frames) == 0 else np.array([f'{x}/k{kid}' not in self.exclude_frames for x in frames])

        # compute pose_gt with all symmetries
        obj_key = obj
        if seq_name.startswith('2022'):
            obj_key = 'hodome+' + obj
        elif seq_name.startswith('2023'):
            obj_key = 'imhd+' + obj
        
        if obj in self.symm_full:
            pose_symm = self.symm_full[obj] # (N, 4, 4)
        else:
            # get dataset specific prefix 
            if seq_name.startswith('2022'):
                key = 'hodome+' + obj
            elif seq_name.startswith('2023'):
                key = 'imhd+' + obj
            else:
                raise ValueError(f'Unknown dataset prefix for symmetry of {seq_name}!')
            pose_symm = self.symm_full[key] # (N, 4, 4)
        poses_gt_symm = np.stack([np.matmul(p[None].repeat(pose_symm.shape[0], 0), pose_symm) for p in poses_gt], 0) # (T, N, 4, 4)

        data_dict = {
            "input_rgbs": torch.stack(input_rgbs, 0).float(), # this is rgbB
            'render_rgbs': np.stack(render_rgbs, axis=0).astype(np.float32), # this is rgbAs
            'input_xyz': torch.stack(depth_xyz, 0).float(),
            'render_xyz': torch.stack(render_xyz, 0).float(),
            'pose_gt': np.stack(poses_gt).astype(np.float32),
            'pose_gt_symm': poses_gt_symm.astype(np.float32),
            'pose_perturbed': np.stack(poses_perturbed).astype(np.float32),
            'delta_transl': np.stack(delta_transl).astype(np.float32),
            'delta_rot': np.stack(delta_rot).astype(np.float32),
            'mesh_diameter': np.array([mesh_diameter] * L).astype(np.float32),
            'rot_normalizer': np.stack([rot_normalizer] * L, 0).astype(np.float32),
            'trans_normalizer': np.stack([trans_normalizer] * L, 0).astype(np.float32),
            'image_files': frames,
            'poseA_norm': poseA_norm.astype(np.float32),
            'joints_body': np.stack(joints_body, 0).astype(np.float32), # (T, 25, 3)
            'fp_error': fp_err_chunk.astype(np.float32),
            'visibility': visibility.astype(np.float32),
            'Krois': np.stack(K_rois, 0).astype(np.float32),
            'frame_mask': frame_mask.astype(np.float32),
            

            # additional info for debug
            'obj_bbox_3d':self.obj_templates[obj_key]['bbox'],
            'to_origin': self.obj_templates[obj_key]['to_origin'],
            'K_rois': np.stack(K_rois), # (T, 3, 3)

            # human info
            **pre_dict,

        }

        # load additional input pose
        if self.pose_pred_dir is not None:
            pkl_file = osp.join(self.pose_pred_dir, f'{seq_name}.pkl')
            d = joblib.load(pkl_file)
            pose_pred, frames_pred = d['poses'], d['frames']
            frames_pred = [x.split('/')[-1] for x in frames_pred]
            id1, id2 = frames_pred.index(frames[0].split('/')[-1]), frames_pred.index(frames[-1].split('/')[-1])
            assert id2 - id1 + 1 == len(frames), f'unreliable pred pose found for seq {seq_name}!'
            data_dict['pose_pred'] = pose_pred[id1:id2+1].copy().astype(np.float32)

        return data_dict

    def process_input(self, dmap_xyz_init, i, input_data, mesh_diameter, nlf_transl, pose_init, render_data,
                      rgb_render, frame, kid, *, augment_depth=None):
        seq, fname = frame.split(os.sep)
        h5_mix_key = f'{seq}+{fname}_k{kid}_input'
        rgb = torch.from_numpy(input_data['rgbmB'][:, :, :3] / 255.).permute(2, 0, 1).float()
        dmap_xyz = torch.from_numpy(input_data['xyzB']).permute(2, 0, 1).float()
        if 'behave-fp+input' in self.render_h5_root:
            dmap_xyz /= 1000.  # legacy error

        # now apply image augmentation 
        if self.split == 'train':
            assert 'behave-fp+input' not in self.render_h5_root, 'procigen data should not use behave-fp+input!'
            augment_depth = 'Date04_Subxx' in seq if augment_depth is None else bool(augment_depth)
            dmap_xyz_augmented, rgb_augmented = augment_observed_video_input(input_data, render_data['K_roi'], self.augm_img, augment_depth=augment_depth)
            if augment_depth:
                dmap_xyz = torch.from_numpy(dmap_xyz_augmented).permute(2, 0, 1).float()
            rgb = torch.from_numpy(rgb_augmented / 255.).permute(2, 0, 1).float()
        defer_mhr_spatial_scale = False
        if getattr(self.cfg, "body_model", "smpl") == "mhr":
            from learning.datasets.mhr_input_materialization import MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT, resolve_mhr_spatial_normalization_type

            defer_mhr_spatial_scale = resolve_mhr_spatial_normalization_type(self.cfg) == MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT
        if self.cfg['normalize_xyz'] and not defer_mhr_spatial_scale:
            dmap_xyz *= 2 / mesh_diameter  # the object radius is 1m now.
            dmap_xyz_init *= 2 / mesh_diameter
        # add mask to the dmap_xyz
        if self.cfg.add_ho_mask:

            mask_ho = torch.from_numpy(input_data['rgbmB'][:, :, 3:] / 255.).permute(2, 0, 1).float()  # (2, H, W)
            if self.cfg.mask_encode_type in ['stack', 'stack-occ']:
                # simply stack them as two channels
                dmap_xyz = torch.cat([dmap_xyz, mask_ho], axis=0)  # (5, H, W)
            elif self.cfg.mask_encode_type == 'hum-obj-fullobj':
                # add one additional full object rendering mask
                mask_render_obj_full = render_data['mask_o'][:, :, 1][None]
                dmap_xyz = torch.cat([dmap_xyz, mask_ho,
                                      torch.from_numpy(mask_render_obj_full).float()], 0)
            elif self.cfg.mask_encode_type == 'obj-fullobj':
                # no human mask, only obj + obj full rendering mask
                mask_render_obj_full = render_data['mask_o'][:, :, 1][None]
                dmap_xyz = torch.cat([dmap_xyz, mask_ho[1:], torch.from_numpy(mask_render_obj_full).float()], 0)
            elif self.cfg.mask_encode_type == 'one-channel':
                comb = self.get_one_channel_mask(mask_ho)
                dmap_xyz = torch.cat([dmap_xyz, comb[None]], axis=0)
            else:
                raise ValueError('Unknown mask encode type: ' + self.cfg.mask_encode_type)
        if self.cfg.mask_rgb_bkg:
            assert self.cfg.add_ho_mask, 'mask add ho mask for this setup!'
            mask_fore = ((mask_ho[0:1] > 0.5) | (mask_ho[1:2] > 0.5)).expand(3, -1, -1)
            rgb[~mask_fore] = 0.  # no background
            dmap_xyz[:3][~mask_fore] = 0  # also mask out xyz map
        if self.cfg.add_ho_mask:
            # add human + object masks as well
            mask_render_full = np.mean(rgb_render, -1) > 0.01
            assert ('mask_o' in render_data) | ('fp+smpl' not in self.render_h5_root), 'incorrect data format!'
            if 'mask_o' in render_data:
                mask_render_obj = render_data['mask_o']
                if len(mask_render_obj.shape) == 3:
                    mask_render_obj = mask_render_obj[:, :, 0]  # keep only the object mask
                mask_render_hum = mask_render_full & (~mask_render_obj)  # subtract obj mask
            else:
                # no human render
                mask_render_obj = mask_render_full
                mask_render_hum = mask_ho[0].cpu().numpy()
            mask_ho_render = torch.from_numpy(np.stack([mask_render_hum, mask_render_obj], 0)).float()
            if self.cfg.mask_encode_type == 'stack':
                # simply stack them as two channels
                dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render], 0)
            elif self.cfg.mask_encode_type == 'one-channel':
                comb = self.get_one_channel_mask(mask_ho_render)
                dmap_xyz_a = torch.cat([dmap_xyz_init, comb[None]], axis=0)
            elif self.cfg.mask_encode_type == 'hum-obj-fullobj': # final model 
                # add one additional full object rendering mask
                mask_render_obj_full = render_data['mask_o'][:, :, 1][None]
                dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render,
                                        torch.from_numpy(mask_render_obj_full).float()], 0)
            elif self.cfg.mask_encode_type == 'obj-fullobj':
                # add one additional full object rendering mask, no human
                mask_render_obj_full = render_data['mask_o'][:, :, 1][None]
                dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render[1:],
                                        torch.from_numpy(mask_render_obj_full).float()], 0)
            elif self.cfg.mask_encode_type == 'stack-occ':
                # stack but take occlusion into account for the rendered obj
                mask_o_render = torch.from_numpy(mask_render_obj)
                mask_o_render[mask_ho[0] > 0.5] = 0  # mask out occlusion by human
                mask_ho_render = torch.stack([mask_ho[0], mask_o_render], 0)
                dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render], 0)
            else:
                raise ValueError('Unknown mask encode type: ' + self.cfg.mask_encode_type)
        else:
            dmap_xyz_a = dmap_xyz_init.clone()
        # one more normalization
        if self.cfg.subtract_transl:
            assert self.cfg['normalize_xyz']
            invalid = dmap_xyz_a[2:3] < 0.01
            spatial_scale = 1.0 if defer_mhr_spatial_scale else 2 / mesh_diameter
            trans_ref = torch.from_numpy(pose_init[:3, 3]).reshape((3, 1, 1)) * spatial_scale
            if nlf_transl is not None:
                if self.cfg.trans_ref_type == 'frame':
                    trans_ref = torch.from_numpy(nlf_transl[i].copy()).reshape((3, 1, 1)) * spatial_scale
                elif self.cfg.trans_ref_type == '1st-frame':
                    trans_ref = torch.from_numpy(nlf_transl[0].copy()).reshape((3, 1, 1)) * spatial_scale
                else:
                    raise ValueError(f'Unknown translation reference type: {self.cfg.trans_ref_type}')
            dmap_xyz_a[:3] = dmap_xyz_a[:3] - trans_ref
            dmap_xyz_a[:3][invalid.repeat(3, 1, 1)] = 0.0

            # same for B
            invalid = dmap_xyz[2:3] < 0.01
            dmap_xyz[:3] = dmap_xyz[:3] - trans_ref
            dmap_xyz[:3][invalid.repeat(3, 1, 1)] = 0.0
            if self.cfg.crop_xyz_3d:
                bound_min, bound_max = np.array([-1, -1, -1.]), np.array([1, 1, 1.])
                m = ((dmap_xyz[0] < bound_max[0]) & (dmap_xyz[0] > bound_min[0])
                     & (dmap_xyz[1] < bound_max[1]) & (dmap_xyz[1] > bound_min[1])
                     & (dmap_xyz[1] < bound_max[2]) & (dmap_xyz[1] > bound_min[2]))
                dmap_xyz[:3][~m[None].repeat(3, 1, 1)] = 0.  # mask out all other points outside the box
        return dmap_xyz, dmap_xyz_a, rgb

    def get_one_channel_mask(self, mask_ho):
        "mask_ho: (2, h, w)"
        mask_h = mask_ho[0] > 0.5
        mask_o = mask_ho[1] > 0.5
        comb = torch.zeros_like(mask_h).float()
        comb[mask_o] = 1.
        comb[mask_h] = -1. # bool -1 is also true!
        return comb

    def extract_render_data(self, render_h5_key, seq, mesh_diameter):
        render_data = load_pickled_dataset(self.render_h5_handles[seq][render_h5_key])
        dmap_xyz_init = Utils.depth2xyzmap(render_data['depth'] , render_data['K_roi'])
        pose_init = render_data['pose']
        bbox_render = render_data['bbox']
        rgb_render = render_data['rgba']
        if self.input_resize != rgb_render.shape[:2]:
            rgb_render = cv2.resize(rgb_render, self.input_resize)
            # use kornia to resize
            h, w= dmap_xyz_init.shape[:2]
            left, right, top, bottom = torch.tensor([0, ]), torch.tensor([w, ]), torch.tensor([0, ]), torch.tensor([h, ])
            out_size = self.input_resize
            tf_full = Utils.compute_tf_batch(left=left, right=right, top=top, bottom=bottom, out_size=out_size)
            dmap_xyz_init = kornia.geometry.transform.warp_perspective(
                torch.as_tensor(dmap_xyz_init[None], device='cpu', dtype=torch.float).permute(0, 3, 1, 2),
                tf_full, dsize=self.input_resize, mode='nearest', align_corners=False)[0]
        else:
            # simply convert to torch tensor
            dmap_xyz_init = torch.from_numpy(dmap_xyz_init).permute(2, 0, 1).float() # (3, H, W)

        return bbox_render, dmap_xyz_init, pose_init, rgb_render, render_data

    def extract_render_data_compact(self, render_h5_key, seq):
        render_data = load_pickled_dataset(self.render_h5_handles[seq][render_h5_key])
        pose_init = render_data['pose']
        bbox_render = render_data['bbox']
        rgb_render = render_data['rgba']
        if self.input_resize != rgb_render.shape[:2]:
            raise ValueError(f'gpu_compact_v1 requires stored render resolution {self.input_resize}, got {rgb_render.shape[:2]} for {render_h5_key}')
        return bbox_render, pose_init, rgb_render, render_data


class VideoDataProcessor(VideoDataset):
    "do not load any data, simply set the configs"
    def __init__(self, cfg, seqs, split='val', *, depth_corruption_probability=None):
        # call the Dataset init function, skip other heavy data loading steps
        Dataset.__init__(self)
        self.cfg = cfg
        self.render_h5_root = cfg['render_root']
        self.split = split 

        fx, fy = 979.7844, 979.840  # for original kinect coordinate system
        cx, cy = 1018.952, 779.486
        self.focal = np.array([fx, fy])
        self.principal_point = np.array([cx, cy])
        factor = 2
        self.K_half = np.array([[fx/factor, 0, cx/factor],
                                [0, fy/factor, cy/factor],
                                [0, 0, 1.]])
        factor = 1
        self.K_full_old = np.array([[fx / factor, 0, cx / factor],
                                [0, fy / factor, cy / factor],
                                [0, 0, 1.]]) # this should never be used
        self.input_resize = cfg['input_resize']
        self.start_inds = [0]
        self.seqs = seqs
        if self.split == 'train':
            self.augm_img = build_video_input_augmenter(depth_corruption_probability=depth_corruption_probability)
