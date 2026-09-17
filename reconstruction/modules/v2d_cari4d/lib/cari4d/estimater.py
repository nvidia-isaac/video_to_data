# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import tempfile
import uuid

import torch

import Utils
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
from scipy.spatial.transform import Rotation as R


def geodesic_distance(R1: R, R2: R) -> float:
    """
    Parameters
    ----------
    R1, R2 : scipy.spatial.transform.Rotation
        Rotations whose distance you want.  They can be built from quaternions,
        matrices, Euler angles, rotation vectors … anything that `Rotation.from_*`
        accepts.

    Returns
    -------
    float
        The geodesic distance in **radians** (0 ≤ θ ≤ π).
    """
    relative = R1.inv() * R2          # R_rel = R1⁻¹ · R2
    return relative.magnitude()       # ||log(R_rel)|| = angle


def matrix_distance(R1, R2):
  """
  R1, R2 : (3,3) real orthonormal matrices with det = +1
  """
  R_rel = R1.T @ R2
  trace = np.trace(R_rel)
  cos_theta = (trace - 1) / 2
  cos_theta = np.clip(cos_theta, -1.0, 1.0)
  return np.arccos(cos_theta)

def cluster_poses(angle_diff, dist_diff, poses_in, symmetry_tfs, return_indices=False):
  """
  Cluster poses based on translation and rotation difference, considering symmetries.

  Args:
      angle_diff (float): Angle threshold in degrees.
      dist_diff (float): Distance threshold.
      poses_in (list of np.ndarray): List of 4x4 pose matrices.
      symmetry_tfs (list of np.ndarray): List of 4x4 symmetry transformation matrices.

  Returns:
      list of np.ndarray: Clustered pose matrices.
  """
  poses_out = [poses_in[0]]
  indices_out = [0]

  radian_thres = angle_diff / 180.0 * np.pi

  for i in range(1, len(poses_in)):
    isnew = True
    cur_pose = poses_in[i]
    for cluster in poses_out:
      t0 = cluster[0:3, 3]
      t1 = cur_pose[0:3, 3]

      if np.linalg.norm(t0 - t1) >= dist_diff:
        continue

      for tf in symmetry_tfs:
        cur_pose_tmp = np.dot(cur_pose, tf)
        rot_diff = matrix_distance(cur_pose_tmp[0:3, 0:3], cluster[0:3, 0:3]) # this is correct geodesic distance
        if rot_diff < radian_thres:
          isnew = False
          break

      if not isnew:
        break

    if isnew:
      poses_out.append(poses_in[i])
      indices_out.append(i)

  return (poses_out, indices_out) if return_indices else poses_out


class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None,
               refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/',
               cfg=None):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(min_n_views=40, inplane_step=60)

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    self.init_refiner(refiner, cfg)

    self.pose_last = None   # Used for tracking; per the centered mesh

    # init pose buffers
    self.poses = None
    self.scores = None
    self.pose_source_ids = None
    self.score_pose_data = None

  def init_refiner(self, refiner, cfg):
    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000, random_seed=FOUNDATIONPOSE_MESH_DIAMETER_RANDOM_SEED)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = os.path.join(tempfile.gettempdir(), f'{uuid.uuid4()}.obj')
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")

  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center

  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)

  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    rot_grid = cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)

  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.001)
    if np.sum(valid) < 4:
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)

  def prepare_registration_frame(self, K, rgb, depth, ob_mask, tensor_residency=False):
    depth_filtered = erode_depth(depth, radius=2, device='cuda')
    depth_filtered = bilateral_filter_depth(depth_filtered, radius=2, device='cuda')
    xyz_map = depth2xyzmap(depth_filtered, K)
    prepared = {'depth': depth_filtered, 'xyz_map': xyz_map}
    if tensor_residency:
      prepared['rgb_tensor'] = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
      prepared['depth_tensor'] = torch.as_tensor(depth_filtered, device='cuda', dtype=torch.float)
      prepared['xyz_map_tensor'] = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
    return prepared

  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, obj_vis=None, vis_score_path=None, vis_refine_path=None, rgb_only=False, seed=0, both_depth_and_rgb=False, prepared_frame=None, tensor_residency=False, return_score_pose_data=False):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(seed)

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
      else:
        self.glctx = glctx

    if prepared_frame is None:
      depth = erode_depth(depth, radius=2, device='cuda')
      depth = bilateral_filter_depth(depth, radius=2, device='cuda')
      xyz_map = depth2xyzmap(depth, K)
      rgb_for_model = rgb
      depth_for_model = depth
      xyz_map_for_model = xyz_map
    else:
      depth = prepared_frame['depth']
      xyz_map = prepared_frame['xyz_map']
      rgb_for_model = prepared_frame.get('rgb_tensor', rgb) if tensor_residency else rgb
      depth_for_model = prepared_frame.get('depth_tensor', depth) if tensor_residency else depth
      xyz_map_for_model = prepared_frame.get('xyz_map_tensor', xyz_map) if tensor_residency else xyz_map

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.001) & (ob_mask>0)

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    if prepared_frame is None and not tensor_residency:
      poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
      poses = poses.data.cpu().numpy()
      center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      if np.allclose(center, np.zeros(3)):
        logging.info(f'center is all zero, use previous center')
        center = self.pose_last[..., :3, 3].clone().cpu().numpy()
      poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
      poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')
    else:
      center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      if np.allclose(center, np.zeros(3)):
        logging.info(f'center is all zero, use previous center')
        center = self.pose_last[..., :3, 3].clone().cpu().numpy()
      poses = self.rot_grid.clone()
      poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    if both_depth_and_rgb:
      # refiner to predict using both depth and rgb mode 
      refiner_poses = poses if tensor_residency else poses.data.cpu().numpy()
      poses_depth, vis_depth = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb_for_model, depth=depth_for_model, K=K, ob_in_cams=refiner_poses,
                                      normal_map=normal_map, xyz_map=xyz_map_for_model, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                      get_vis=self.debug>=2, rgb_only=False)
      poses_rgb, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb_for_model, depth=depth_for_model, K=K, ob_in_cams=refiner_poses,
                                      normal_map=normal_map, xyz_map=xyz_map_for_model, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                      get_vis=self.debug>=2, rgb_only=True)                                
      if vis_depth is not None:
        vis = np.concatenate([vis_depth, vis], axis=0)
      poses = torch.cat([poses_depth, poses_rgb], dim=0)
    else:
      refiner_poses = poses if tensor_residency else poses.data.cpu().numpy()
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb_for_model, depth=depth_for_model, K=K, ob_in_cams=refiner_poses,
                                      normal_map=normal_map, xyz_map=xyz_map_for_model, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                      get_vis=self.debug>=2, rgb_only=rgb_only)
    if vis is not None:
      outfile = f'{self.debug_dir}/vis_refiner.png' if vis_refine_path is None else vis_refine_path
      imageio.imwrite(outfile, vis)

    scorer_poses = poses if tensor_residency else poses.data.cpu().numpy()
    score_result = self.scorer.predict(mesh=self.mesh, rgb=rgb_for_model, depth=depth_for_model, K=K, ob_in_cams=scorer_poses, normal_map=normal_map,
                                      mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2,
                                      no_text=False, rgb_only=rgb_only, return_pose_data=return_score_pose_data)
    if return_score_pose_data:
      scores, vis, score_pose_data = score_result
    else:
      scores, vis = score_result
      score_pose_data = None
    if vis is not None:
      outfile = f'{self.debug_dir}/vis_score.png' if vis_score_path is None else vis_score_path
      imageio.imwrite(outfile, vis)

    ids = torch.as_tensor(scores).argsort(descending=True)
    scores = scores[ids]
    poses = poses[ids] # already ranked
    if score_pose_data is not None:
      score_pose_data = score_pose_data.select_by_indices(ids)

    # filter out based on consistency with prev pose
    if obj_vis is not None and self.pose_last is not None:
      pose_prev = self.pose_last[None].expand(len(poses), -1, -1)
      dist = Utils.geodesic_distance_batch(pose_prev[:, :3, :3], poses[:, :3, :3])
      mask = dist < torch.pi * 0.25
      if torch.sum(mask) > 1:
        poses = poses[mask]
        ids = ids[mask]
      else:
        print('all pose candidates are filtered out, not doing anything!')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]

    self.poses = poses
    self.scores = scores
    self.pose_source_ids = ids
    self.score_pose_data = score_pose_data

    return best_pose.data.cpu().numpy()


  def track_one(self, rgb, depth, K, iteration, extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                     ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None,
                                     xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration,
                                     get_vis=self.debug>=2)
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
