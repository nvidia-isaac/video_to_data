# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import kornia
import torch

from Utils import depth2xyzmap_batch


class FoundationPoseBatchData:
  def __init__(self, rgbAs=None, rgbBs=None, depthAs=None, depthBs=None, normalAs=None, normalBs=None, maskAs=None, maskBs=None, poseA=None, poseB=None, xyz_mapAs=None, xyz_mapBs=None, tf_to_crops=None, Ks=None, crop_masks=None, model_pts=None, mesh_diameters=None, labels=None):
    self.rgbAs = rgbAs
    self.rgbBs = rgbBs
    self.depthAs = depthAs
    self.depthBs = depthBs
    self.normalAs = normalAs
    self.normalBs = normalBs
    self.poseA = poseA
    self.poseB = poseB
    self.maskAs = maskAs
    self.maskBs = maskBs
    self.xyz_mapAs = xyz_mapAs
    self.xyz_mapBs = xyz_mapBs
    self.tf_to_crops = tf_to_crops
    self.crop_masks = crop_masks
    self.Ks = Ks
    self.model_pts = model_pts
    self.mesh_diameters = mesh_diameters
    self.labels = labels

  def select_by_indices(self, ids):
    out = FoundationPoseBatchData()
    for key, value in self.__dict__.items():
      if value is not None:
        out.__dict__[key] = value[ids.to(value.device)]
    return out


class FoundationPoseRefineDataProcessor:
  def __init__(self, cfg):
    self.cfg = cfg

  def transform_depth_to_xyzmap(self, batch, H_ori, W_ori, subtract_trans=True, crop_xyz_3d=False):
    bs = len(batch.rgbAs)
    H, W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()
    if batch.xyz_mapAs is None:
      depthAs_ori = kornia.geometry.transform.warp_perspective(batch.depthAs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)
      batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapAs[:,2:3]<0.001
    if batch.xyz_mapAs.shape[1]==3:
      batch.xyz_mapAs = batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1) if subtract_trans else batch.xyz_mapAs
      if self.cfg['normalize_xyz']:
        batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
        invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
        batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0
    else:
      assert batch.xyz_mapAs.shape[1] == 5, f'invalid xyz_mapAs shape {batch.xyz_mapAs.shape}'
      mask_ho = batch.xyz_mapAs[:,3:]
      batch.xyz_mapAs = batch.xyz_mapAs[:,:3]-batch.poseA[:,:3,3].reshape(bs,3,1,1) if subtract_trans else batch.xyz_mapAs[:,:3]
      if self.cfg['normalize_xyz']:
        batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
        invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
        batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0
      batch.xyz_mapAs = torch.cat([batch.xyz_mapAs, mask_ho], dim=1)
    if batch.xyz_mapBs is None:
      depthBs_ori = kornia.geometry.transform.warp_perspective(batch.depthBs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)
      batch.xyz_mapBs = kornia.geometry.transform.warp_perspective(batch.xyz_mapBs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapBs[:,2:3]<0.001
    if batch.xyz_mapBs.shape[1]==3:
      batch.xyz_mapBs = batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1) if subtract_trans else batch.xyz_mapBs
      if self.cfg['normalize_xyz']:
        batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
        invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
        batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0
    else:
      assert batch.xyz_mapBs.shape[1] == 5, f'invalid xyz_mapBs shape {batch.xyz_mapBs.shape}'
      mask_ho = batch.xyz_mapBs[:,3:]
      batch.xyz_mapBs = batch.xyz_mapBs[:,:3]-batch.poseA[:,:3,3].reshape(bs,3,1,1) if subtract_trans else batch.xyz_mapBs[:,:3]
      if self.cfg['normalize_xyz']:
        batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
        invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
        batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0
      batch.xyz_mapBs = torch.cat([batch.xyz_mapBs, mask_ho], dim=1)
    if crop_xyz_3d:
      assert subtract_trans
      dmap_xyz = batch.xyz_mapBs[:,:3]
      mask = ((dmap_xyz[:,0]<1) & (dmap_xyz[:,0]>-1) & (dmap_xyz[:,1]<1) & (dmap_xyz[:,1]>-1) & (dmap_xyz[:,1]<1) & (dmap_xyz[:,1]>-1))
      dmap_xyz[~mask[:,None].repeat(1,3,1,1)] = 0
      batch.xyz_mapBs[:,:3] = dmap_xyz
    return batch

  def transform_batch(self, batch, H_ori, W_ori, bound=1, subtract_trans=True, crop_xyz_3d=False):
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0
    return self.transform_depth_to_xyzmap(batch, H_ori, W_ori, subtract_trans=subtract_trans, crop_xyz_3d=crop_xyz_3d)


class FoundationPoseScoreDataProcessor:
  def __init__(self, cfg):
    self.cfg = cfg

  def transform_depth_to_xyzmap(self, batch, H_ori, W_ori):
    bs = len(batch.rgbAs)
    H, W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()
    if batch.xyz_mapAs is None:
      depthAs_ori = kornia.geometry.transform.warp_perspective(batch.depthAs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)
      batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    invalid = batch.xyz_mapAs[:,2:3]<0.1
    batch.xyz_mapAs = batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
    if self.cfg['normalize_xyz']:
      batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
      batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0
    if batch.xyz_mapBs is None:
      xyz_mapBs_list = []
      for i in range(0, bs, 128):
        depthBs_ori = kornia.geometry.transform.warp_perspective(batch.depthBs.expand(bs,-1,-1,-1)[i:i+128], crop_to_oris[i:i+128], dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
        xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:,0], batch.Ks[i:i+128], zfar=np.inf).permute(0,3,1,2)
        xyz_mapBs_list.append(kornia.geometry.transform.warp_perspective(xyz_mapBs, tf_to_crops[i:i+128], dsize=(H,W), mode='nearest', align_corners=False))
      batch.xyz_mapBs = torch.cat(xyz_mapBs_list, 0)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    invalid = batch.xyz_mapBs[:,2:3]<0.1
    batch.xyz_mapBs = batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1)
    if self.cfg['normalize_xyz']:
      batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
      batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0
    return batch

  def transform_batch(self, batch, H_ori, W_ori, bound=1):
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0
    return self.transform_depth_to_xyzmap(batch, H_ori, W_ori)
