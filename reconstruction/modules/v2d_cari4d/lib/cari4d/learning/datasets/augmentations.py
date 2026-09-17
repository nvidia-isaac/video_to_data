# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.




import io
import random,glob
from copy import deepcopy
from pathlib import Path
import imageio.v2 as imageio
import imgaug.augmenters as iaa
import cv2,os,sys
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../../')
sys.path.append(f'{code_dir}/../../')

from Utils import *
import scipy.signal
import scipy.stats
import torch,skimage


class HSVNoise:
  def __init__(self, noise_mag, prob=0.5):
    self.prob = prob
    self.noise_mag = noise_mag


  def __call__(self, data):
    '''
    @data: if data is a list (e.g. in triplet), we apply the same noise
    '''
    if np.random.uniform(0, 1) <= self.prob:
      if isinstance(data, list):
        noise = np.random.uniform(-self.noise_mag, self.noise_mag, size=data[0].rgbB.shape)
      else:
        noise = np.random.uniform(-self.noise_mag, self.noise_mag, size=data.rgbB.shape)

      def transform(data):
        hsv = cv2.cvtColor(data.rgbB, cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv = hsv.astype(np.float32)+noise
        hsv = np.clip(hsv,0,255)
        data.rgbB = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return data

      if isinstance(data, list):
        for i in range(len(data)):
          data[i] = transform(data[i])
      else:
        data = transform(data)
    return data


class ChangeBright:
  def __init__(self,prob=0.5,mag=[0.5,2], augment_imgA=False):
    self.mag = mag
    self.prob = prob
    self.augment_imgA = augment_imgA

  def __call__(self,data):
    if np.random.uniform()>=self.prob:
      return data

    augA = iaa.MultiplyBrightness(np.random.uniform(self.mag[0],self.mag[1]))  #!NOTE iaa has bug about deterministic, we sample ourselves
    augB = iaa.MultiplyBrightness(np.random.uniform(self.mag[0],self.mag[1]))

    if isinstance(data, dict):
      data['rgb'] = augB(images=[data['rgb'].astype(np.uint8)])[0]
      return data

    def transform(data):
      if self.augment_imgA:
        data.rgbA = augA(images=[data.rgbA.clip(0,255).astype(np.uint8)])[0]
      data.rgbB = augB(images=[data.rgbB.clip(0,255).astype(np.uint8)])[0]
      return data

    if isinstance(data, list):
      if self.augment_imgA:
        rgbAs = np.asarray([d.rgbA for d in data]).clip(0,255).astype(np.uint8)
        rgbAs = augA(images=rgbAs)
        for i in range(len(data)):
          data[i].rgbA = rgbAs[i]

      rgbBs = np.asarray([d.rgbB for d in data]).clip(0,255).astype(np.uint8)
      rgbBs = augB(images=rgbBs)
      for i in range(len(data)):
        data[i].rgbB = rgbBs[i]

    else:
      if self.augment_imgA:
        data.rgbA = augA(images=[data.rgbA.clip(0,255).astype(np.uint8)])[0]
      data.rgbB = augB(images=[data.rgbB.clip(0,255).astype(np.uint8)])[0]
    return data



class ChangeContrast:
  def __init__(self,prob=0.5,mag=[0.5,2]):
    self.mag = mag
    self.prob = prob

  def __call__(self,data):
    if np.random.uniform()>=self.prob:
      return data

    augB = iaa.GammaContrast(np.random.uniform(self.mag[0],self.mag[1]))

    if isinstance(data, dict):
      data['rgb'] = augB(images=[data['rgb'].astype(np.uint8)])[0]
      return data

    def transform(data):
      data.rgbB = augB(images=[data.rgbB.clip(0,255).astype(np.uint8)])[0]
      return data

    if isinstance(data, list):
      rgbBs = np.asarray([d.rgbB for d in data]).clip(0,255).astype(np.uint8)
      rgbBs = augB(images=rgbBs)
      for i in range(len(data)):
        data[i].rgbB = rgbBs[i]
    else:
      data.rgbB = augB(images=[data.rgbB.clip(0,255).astype(np.uint8)])[0]
    return data



class SaltAndPepper:
  def __init__(self, prob, ratio=0.1, per_channel=True):
    self.prob = prob
    self.ratio = ratio
    self.per_channel = per_channel

  def __call__(self, data):
    if np.random.uniform(0,1)>=self.prob:
      return data

    if isinstance(data, dict):
      aug = iaa.SaltAndPepper(self.ratio, per_channel=self.per_channel).to_deterministic()
      data['rgb'] = aug(images=[data['rgb'].astype(np.uint8)])[0]
      return data

    def transform(data, aug):
      data.rgbB = aug(images=[data.rgbB])[0]
      data.rgbB[data.crop_mask==0] = 0
      return data

    aug = iaa.SaltAndPepper(self.ratio, per_channel=self.per_channel).to_deterministic()

    if isinstance(data, list):
      for i in range(len(data)):
        data[i] = transform(data[i], aug)
    else:
      data = transform(data, aug)
    return data


class RGBGaussianNoise:
  def __init__(self, max_noise, prob=0.5):
    self.max_noise = max_noise
    self.prob = prob

  def __call__(self, data):
    if np.random.uniform(0, 1) >= self.prob:
      return data

    if isinstance(data, dict):
      shape = data['rgb'].shape
      noise = np.random.normal(0, self.max_noise, size=shape).clip(-self.max_noise, self.max_noise)
      data['rgb'] = (data['rgb'].astype(float) + noise).clip(0,255).astype(np.uint8)
      return data

    def transform(data, noise):
      data.rgbB = (data.rgbB.astype(np.float32) + noise).clip(0, 255)
      data.rgbB[data.crop_mask==0] = 0
      return data

    if isinstance(data, list):
      shape = data[0].rgbB.shape
    else:
      shape = data.rgbB.shape

    noise = np.random.normal(0, self.max_noise, size=shape).clip(-self.max_noise, self.max_noise)
    if isinstance(data, list):
      for i in range(len(data)):
        data[i] = transform(data[i], noise)
    else:
      data = transform(data, noise)
    return data


def _compress_jpeg_in_memory(image, compression):
  quality = int(np.clip(np.round(1 + 99 * (1.0 - compression / 101)), 1, 100))
  encoded = io.BytesIO()
  Image.fromarray(image).save(encoded, format='JPEG', quality=quality)
  encoded.seek(0)
  return imageio.imread(encoded, pilmode='RGB', format='jpeg')


class JpegAug:
  def __init__(self, prob, compression_range=[0,20]):
    self.prob = prob
    self.compression_range = compression_range   # Higher the more compression

  def __call__(self, data):
    if np.random.uniform(0,1)>=self.prob:
      return data

    def transform(data, aug):
      data.rgbB = _compress_jpeg_in_memory(data.rgbB.clip(0,255).astype(np.uint8), aug)
      data.rgbB[data.crop_mask==0] = 0
      return data

    compression = np.random.randint(self.compression_range[0], self.compression_range[1]+1)

    if isinstance(data, dict):
      data['rgb'] = _compress_jpeg_in_memory(data['rgb'].astype(np.uint8), compression)
      return data

    if isinstance(data, list):
      rgbBs = np.asarray([d.rgbB for d in data]).clip(0,255).astype(np.uint8)
      rgbBs = np.asarray([_compress_jpeg_in_memory(rgbB, compression) for rgbB in rgbBs])
      crop_masks = np.asarray([d.crop_mask for d in data])
      rgbBs[crop_masks==0] = 0
      for i in range(len(data)):
        data[i].rgbB = rgbBs[i]
    else:
      data.rgbB = _compress_jpeg_in_memory(data.rgbB.clip(0,255).astype(np.uint8), compression)
      data.rgbB[data.crop_mask==0] = 0
    return data


class GaussianBlur:
  def __init__(self, max_kernel_size, min_kernel_size=3, sigma_range=(0, 3.0), prob=0.4):
    self.prob = prob
    self.max_kernel_size = max_kernel_size
    self.min_kernel_size = min_kernel_size
    self.sigma_range = sigma_range

  def __call__(self, data):
    if random.uniform(0, 1) >= self.prob:
      return data

    sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
    ksize = np.random.randint(self.min_kernel_size, self.max_kernel_size+1)
    ksize = np.clip(ksize//2*2 + 1, self.min_kernel_size, self.max_kernel_size)  # Make it odd number

    if isinstance(data, dict):
      aug = iaa.GaussianBlur(sigma=sigma)
      data['rgb'] = aug(images=[data['rgb']])[0]
      return data

    def transform(data, ksize, sigma):
      data.rgbB = cv2.GaussianBlur(data.rgbB,(ksize,ksize),0)
      data.rgbB[data.crop_mask==0] = 0
      return data

    if isinstance(data, list):
      aug = iaa.GaussianBlur(sigma=sigma)
      rgbBs = np.asarray([d.rgbB for d in data])
      rgbBs = aug(images=rgbBs)
      crop_masks = np.asarray([d.crop_mask for d in data])
      rgbBs[crop_masks==0] = 0
      for i in range(len(data)):
        data[i].rgbB = rgbBs[i]
    else:
      data.rgbB = cv2.GaussianBlur(data.rgbB,(ksize,ksize), sigma)
      data.rgbB[data.crop_mask==0] = 0

    return data


class Rotate:
  def __init__(self):
    pass

  def __call__(self, data):

    def transform(data, mode):
      if mode!=-1:
        data.rgbA = cv2.rotate(data.rgbA, mode)
        data.rgbB = cv2.rotate(data.rgbB, mode)
        data.depthA = cv2.rotate(data.depthA, mode)
        data.depthB = cv2.rotate(data.depthB, mode)
        data.maskA = cv2.rotate(data.maskA, mode)
        data.maskB = cv2.rotate(data.maskB, mode)
      return data

    mode = np.random.choice([-1, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])

    if isinstance(data, list):
      for i in range(len(data)):
        data[i] = transform(data[i], mode)
    else:
      data = transform(data, mode)

    return data


class Flip:
  def __init__(self):
    pass

  def __call__(self, data):

    def transform(data, flip_code):
      data.rgbA = cv2.flip(data.rgbA, flip_code)
      data.rgbB = cv2.flip(data.rgbB, flip_code)
      data.depthA = cv2.flip(data.depthA, flip_code)
      data.depthB = cv2.flip(data.depthB, flip_code)
      data.maskA = cv2.flip(data.maskA, flip_code)
      data.maskB = cv2.flip(data.maskB, flip_code)
      return data

    mode = np.random.choice([-1,0,1,2])
    if mode>=0:
      flip_code = mode-1
      if isinstance(data, list):
        for i in range(len(data)):
          data[i] = transform(data, flip_code)
      else:
        data = transform(data, flip_code)
    return data


class DepthDropout:
  def __init__(self, prob):
    self.prob = prob

  def __call__(self, data):
    '''This should be called after scaling depth
    '''
    def transform(data):
      data.depthB = np.zeros(data.depthB.shape)
      if data.normalB is not None:
        if data.normalB.dtype==np.uint8:
          data.normalB = np.ones_like(data.normalB)*127
        else:
          data.normalB = np.zeros_like(data.normalB.shape)
      return data

    if np.random.uniform(0,1)<self.prob:
      if isinstance(data, list):
        for i in range(len(data)):
          data[i] = transform(data[i])
      else:
        data = transform(data)

    return data


class PseudoOriginalHelper:
  '''A list of cropped data when converting to ori resolution, they are taking a small fraction. This class is to mimic this sparse full resolution while aiming to be efficient
  '''
  def __init__(self, data:list, H_ori, W_ori):
    self.data = data
    self.H_ori = H_ori
    self.W_ori = W_ori
    self.H_crop, self.W_crop = self.data[0].depthB.shape[:2]
    corners_crop = np.array([0,0,self.W_crop-1, self.H_crop-1]).reshape(2,2)
    self.tf_to_crops = np.asarray([d.tf_to_crop for d in data]).astype(float)  #(N,3,3)
    corners_ori = transform_pts(corners_crop, np.linalg.inv(self.tf_to_crops[:,None])).reshape(len(data),4)
    self.corners_ori = corners_ori.round().astype(int)
    self.umin = self.corners_ori[:,0].min()
    self.vmin = self.corners_ori[:,1].min()
    self.umax = self.corners_ori[:,2].max()
    self.vmax = self.corners_ori[:,3].max()
    self.H = self.vmax-self.vmin+1
    self.W = self.umax-self.umin+1
    self.tf_to_pseudo = np.eye(3)
    self.tf_to_pseudo[0,2] = -self.umin
    self.tf_to_pseudo[1,2] = -self.vmin
    self.pseudo_to_crops = self.tf_to_crops@np.linalg.inv(self.tf_to_pseudo)[None]

  def make_ori_rgb(self):
    rgb_ori = np.zeros((self.H, self.W, 3), dtype=np.uint8)
    for i,d in enumerate(self.data):
      rgb = cv2.warpPerspective(d.rgbB, np.linalg.inv(self.pseudo_to_crops[i]), dsize=(self.W, self.H))
      rgb_ori = np.maximum(rgb_ori, rgb)

    return rgb_ori


  def make_ori_depth(self):
    depth_ori = np.zeros((self.H, self.W), dtype=float)
    for i,d in enumerate(self.data):
      depth = cv2.warpPerspective(d.depthB, np.linalg.inv(self.pseudo_to_crops[i]), flags=cv2.INTER_NEAREST)
      depth_ori = np.maximum(depth_ori, depth)

    return depth_ori


  def make_ori_mask(self):
    mask_ori = np.zeros((self.H, self.W), dtype=np.uint8)
    for i,d in enumerate(self.data):
      mask = cv2.warpPerspective(d.maskB.astype(np.uint8), np.linalg.inv(self.pseudo_to_crops[i]), dsize=(self.W, self.H), flags=cv2.INTER_NEAREST)
      mask_ori = np.maximum(mask_ori, mask)

    return mask_ori



class DepthMissingBase:
  def __init__(self, prob, H_ori, W_ori, edge_grad_thres=0.02, missing_ratio_range=[0.5, 1.0], dilate_kernel_range=[0, 5]):
    self.prob = prob
    self.H_ori = H_ori
    self.W_ori = W_ori
    self.edge_grad_thres = edge_grad_thres
    self.missing_ratio_range = missing_ratio_range
    self.dilate_kernel_range = dilate_kernel_range


  def make_ori_depth(self, data:list):
    '''The list of data are different crops from the original depth. Here we will make the original depth
    '''
    depth_ori = np.zeros((self.H_ori, self.W_ori), dtype=np.float32)
    for d in data:
      depth = cv2.warpPerspective(d.depthB, np.linalg.inv(d.tf_to_crop), dsize=(self.W_ori, self.H_ori), flags=cv2.INTER_NEAREST)
      depth_ori = np.maximum(depth_ori, depth)
    return depth_ori


  def make_ori_mask(self, data:list):
    '''The list of data are different crops from the original depth. Here we will make the original depth
    '''
    mask_ori = np.zeros((self.H_ori, self.W_ori), dtype=np.uint8)
    for d in data:
      mask = cv2.warpPerspective(d.maskB.astype(np.uint8), np.linalg.inv(d.tf_to_crop), dsize=(self.W_ori, self.H_ori), flags=cv2.INTER_NEAREST)
      mask_ori = np.maximum(mask_ori, mask)

    return mask_ori


  def make_ori_rgb(self, data:list):
    '''The list of data are different crops from the original depth. Here we will make the original depth
    '''
    rgb_ori = np.zeros((self.H_ori, self.W_ori, 3), dtype=np.uint8)
    for d in data:
      rgb = cv2.warpPerspective(d.rgbB, np.linalg.inv(d.tf_to_crop), dsize=(self.W_ori, self.H_ori), flags=cv2.INTER_LINEAR)
      update_mask = (rgb>0).any(axis=-1)
      rgb_ori[update_mask] = rgb[update_mask]

    return rgb_ori


  def make_drop_mask(self, depth):
    raise NotImplementedError


  def __call__(self, data):
    if np.random.uniform(0,1)>=self.prob:
      return data

    def transform(data, drop_mask):
      H,W = data.depthB.shape
      if drop_mask.shape[:2]==(self.H_ori, self.W_ori):
        drop_mask = cv2.warpPerspective(drop_mask.astype(np.uint8), data.tf_to_crop, dsize=(W, H), flags=cv2.INTER_NEAREST)
      data.depthB[drop_mask>0] = 0
      return data

    if isinstance(data, dict):
      drop_mask = self.make_drop_mask(data['depth'])
      data['depth'][drop_mask>0] = 0
      return data

    if isinstance(data, list):
      depth_ori = self.make_ori_depth(data)
      drop_mask = self.make_drop_mask(depth_ori)
      for i in range(len(data)):
        data[i] = transform(data[i], drop_mask)
    else:
      drop_mask = self.make_drop_mask(data.depthB)
      data = transform(data, drop_mask)

    return data


class DepthEllipseMissing(DepthMissingBase):
  def __init__(self, prob, H_ori, W_ori, max_num_ellipse=5, max_radius=30):
    self.prob = prob
    self.H_ori = H_ori
    self.W_ori = W_ori
    self.max_num_ellipse = max_num_ellipse
    self.max_radius = max_radius   # Ratio per crop size


  def generate_random_ellipses(self, depth=None, max_radius=None):
    num_ellipses_to_dropout = np.random.randint(1, self.max_num_ellipse+1)
    num_ellipses_to_dropout = int(num_ellipses_to_dropout)

    vs,us = np.where(depth>=0.1)
    num_ellipses_to_dropout = min(len(us), int(num_ellipses_to_dropout))
    indices = np.random.choice(len(us), size=num_ellipses_to_dropout, replace=False)
    if len(indices)==0:
      return None,None,None,None
    dropout_centers = np.stack([us,vs], axis=-1).reshape(-1,2)[indices]

    min_radius = 1
    x_radii = np.random.uniform(min_radius, max_radius, size=dropout_centers.shape[0])
    y_radii = np.random.uniform(min_radius, max_radius, size=dropout_centers.shape[0])
    angles = np.random.uniform(0, 360, size=dropout_centers.shape[0])

    return x_radii, y_radii, angles, dropout_centers


  def make_drop_mask(self, depth, max_radius):
    H,W = depth.shape
    drop_mask = np.zeros((H,W), dtype=np.uint8)
    x_radii, y_radii, angles, dropout_centers = self.generate_random_ellipses(depth, max_radius)
    if dropout_centers is None:
      return drop_mask
    num_ellipses_to_dropout = x_radii.shape[0]
    for i in range(num_ellipses_to_dropout):
      center = dropout_centers[i]
      x_radius = np.round(x_radii[i]).astype(int)
      y_radius = np.round(y_radii[i]).astype(int)
      angle = angles[i]
      drop_mask = cv2.ellipse(drop_mask, tuple(center), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)

    return drop_mask


  def __call__(self, data):
    if np.random.uniform(0,1)>=self.prob:
      return data

    def transform(data, drop_mask):
      H,W = data.depthB.shape[:2]
      if drop_mask.shape!=(H,W):
        drop_mask = cv2.warpPerspective(drop_mask, data.tf_to_crop, dsize=(W,H), flags=cv2.INTER_NEAREST)
      data.depthB[drop_mask>0] = 0
      if data.normalB is not None:
        invalid = data.depthB<0.1
        if data.normalB.dtype==np.uint8:
          data.normalB[invalid] = 127
        else:
          data.normalB[invalid] = 0
      return data

    if isinstance(data, dict):
      H,W = data['depth'].shape
      drop_mask = self.make_drop_mask(depth=(data['seg']>0).astype(float), max_radius=self.max_radius)
      data['depth'][drop_mask>0] = 0
      return data

    if isinstance(data, list):
      helper = PseudoOriginalHelper(data, H_ori=self.H_ori, W_ori=self.W_ori)
      H,W = data[0].depthB.shape[:2]
      drop_mask = self.make_drop_mask(depth=np.ones((helper.H, helper.W)), max_radius=self.max_radius)
    else:
      H,W = data.depthB.shape[:2]
      drop_mask = self.make_drop_mask(depth=np.ones((H,W)), max_radius=self.max_radius)


    if isinstance(data, list):
      for i in range(len(data)):
        H,W = data[i].depthB.shape[:2]
        drop_mask_cur = cv2.warpPerspective(drop_mask, helper.pseudo_to_crops[i], dsize=(W,H), flags=cv2.INTER_NEAREST)
        data[i] = transform(data[i], drop_mask_cur)
    else:
      data = transform(data, drop_mask)
    return data




class DepthRoiMissing(DepthMissingBase):
  def __init__(self, prob, H_ori, W_ori, max_missing_ratio=1, downscale_range=[0.1, 1]):
    self.prob = prob
    self.H_ori = H_ori
    self.W_ori = W_ori
    self.max_missing_ratio = max_missing_ratio
    self.downscale_range = downscale_range


  def make_drop_mask(self, mask, downscale):
    H,W = mask.shape[:2]
    mask_down = cv2.resize(mask.astype(np.uint8), fx=downscale, fy=downscale, dsize=None, interpolation=cv2.INTER_NEAREST)
    vs,us = np.where(mask_down>0)
    missing_ratio = np.random.uniform(0, self.max_missing_ratio)
    ids = np.random.choice(len(us), size=int(missing_ratio*len(us)), replace=False)
    drop_mask_down = np.zeros(mask_down.shape, dtype=np.uint8)
    drop_mask_down[vs[ids], us[ids]] = 1
    drop_mask = cv2.resize(drop_mask_down, dsize=(W,H), interpolation=cv2.INTER_NEAREST)
    return drop_mask


  def __call__(self, data):
    if np.random.uniform()>=self.prob:
      return data

    downscale = np.random.uniform(self.downscale_range[0], self.downscale_range[1])

    if isinstance(data, dict):
      mask = data['seg']>=1
      drop_mask = self.make_drop_mask(mask, downscale=downscale)
      data['depth'][drop_mask>0] = 0
      return data

    def transform(data, drop_mask):
      H,W = data.depthB.shape
      if drop_mask.shape!=(H,W):
        drop_mask = cv2.warpPerspective(drop_mask, data.tf_to_crop, dsize=(W,H), flags=cv2.INTER_NEAREST)
      data.depthB[drop_mask>0] = 0
      return data

    if isinstance(data, list):
      helper = PseudoOriginalHelper(data, H_ori=self.H_ori, W_ori=self.W_ori)
      mask = helper.make_ori_mask()
    else:
      mask = data.maskB.copy()

    drop_mask = self.make_drop_mask(mask, downscale=downscale)

    if isinstance(data, list):
      for i in range(len(data)):
        H,W = data[i].depthB.shape[:2]
        drop_mask_cur = cv2.warpPerspective(drop_mask, helper.tf_to_crops[i], dsize=(W,H), flags=cv2.INTER_NEAREST)
        data[i] = transform(data[i], drop_mask_cur)
    else:
      data.depthB[drop_mask>0] = 0

    return data


class DepthCorrelatedGaussianNoise:
  def __init__(self, prob, W_ori, H_ori, noise_range=[0, 0.01], rescale_factor_min=2, rescale_factor_max=10):
    """Adds random Gaussian noise to the depth image."""
    self.prob = prob
    self.W_ori = W_ori
    self.H_ori = H_ori
    self.noise_range = noise_range
    self.rescale_factor_min = rescale_factor_min
    self.rescale_factor_max = rescale_factor_max


  def make_noise(self, H, W, mag):
    rescale_factor = np.random.uniform(low=self.rescale_factor_min, high=self.rescale_factor_max)
    small_H, small_W = (np.array([H,W]) / rescale_factor).round().astype(int)
    additive_noise = np.random.normal(loc=0.0, scale=mag, size=(small_H, small_W)).clip(-mag, mag).astype(np.float32)
    additive_noise = cv2.resize(additive_noise, (W,H), interpolation=cv2.INTER_CUBIC)
    return additive_noise


  def __call__(self, data):
    if np.random.uniform(0,1)>=self.prob:
      return data

    mag = np.random.uniform(self.noise_range[0], self.noise_range[1])

    if isinstance(data, dict):
      H,W = data['depth'].shape
      noise = self.make_noise(H=H, W=W, mag=mag)
      valid = data['depth']>=0.1
      data['depth'][valid] += noise[valid]
      return data


    def transform(data, noise):
      invalid = (data.depthB<0.1) | (data.crop_mask==0)
      H,W = data.depthB.shape[:2]
      noise_window = cv2.warpPerspective(noise, data.tf_to_crop, dsize=(W, H), flags=cv2.INTER_NEAREST)
      data.depthB[~invalid] += noise_window[~invalid]
      data.depthB = data.depthB.clip(0, None)
      return data

    if isinstance(data, list):
      noise = self.make_noise(H=self.H_ori, W=self.W_ori, mag=mag)
      for i in range(len(data)):
        data[i] = transform(data[i], noise)
    else:
      noise = self.make_noise(H=data.depthB.shape[0], W=data.depthB.shape[1], mag=mag)
      data = transform(data, noise)

    return data


class DepthMissing(DepthMissingBase):
  def __init__(self, prob, H_ori, W_ori, max_missing_percent=0.5, down_scale=1):
    self.prob = prob
    self.H_ori = H_ori
    self.W_ori = W_ori
    self.down_scale = down_scale
    self.max_missing_percent = max_missing_percent


  def make_drop_mask(self, depth_down, missing_percent):
    drop_mask = np.zeros(depth_down.shape, dtype=np.uint8)
    vs,us = np.where(depth_down>0.1)
    missing_ids = np.random.choice(len(us), size=int(missing_percent*len(us)), replace=False)
    drop_mask[vs[missing_ids], us[missing_ids]] = 1
    drop_mask = cv2.resize(drop_mask, fx=1/self.down_scale, fy=1/self.down_scale, dsize=None, interpolation=cv2.INTER_NEAREST)
    return drop_mask


  def __call__(self,data):
    if np.random.uniform(0,1)>=self.prob:
      return data

    def transform(data, drop_mask):
      H,W = data.depthB.shape
      if drop_mask.shape!=(H,W):
        drop_mask = cv2.warpPerspective(drop_mask, data.tf_to_crop, dsize=(W,H), flags=cv2.INTER_NEAREST)
      data.depthB[drop_mask>0] = 0
      invalid = data.depthB<0.1
      return data

    missing_percent = np.random.uniform(0,self.max_missing_percent)

    if isinstance(data, dict):
      H,W = data['depth'].shape
      depth_down = np.ones((int(H*self.down_scale), int(W*self. down_scale)), dtype=float)
      drop_mask = self.make_drop_mask(depth_down, missing_percent=missing_percent)
      data['depth'][drop_mask>0] = 0
      return data

    if isinstance(data, list):
      helper = PseudoOriginalHelper(data, H_ori=self.H_ori, W_ori=self.W_ori)
      depth_down = np.ones((int(helper.H*self.down_scale), int(helper.W*self. down_scale)), dtype=float)
    else:
      H,W = data.depthB.shape
      depth_down = np.ones((int(H*self.down_scale), int(W*self. down_scale)), dtype=float)
    drop_mask = self.make_drop_mask(depth_down, missing_percent=missing_percent)

    if isinstance(data, list):
      for i in range(len(data)):
        H,W = data[i].depthB.shape[:2]
        drop_mask_cur = cv2.warpPerspective(drop_mask, helper.pseudo_to_crops[i], dsize=(W,H), flags=cv2.INTER_NEAREST)
        data[i] = transform(data[i], drop_mask_cur)
    else:
      data = transform(data, drop_mask)
    return data



class NormalizeDepth:
  def __init__(self):
    pass

  def __call__(self, data):
    def transform(data):
      min_val = 0
      max_val = 3
      invalid = data.depthA<0.1
      data.depthA = (data.depthA-min_val)/(max_val-min_val)
      data.depthA = data.depthA.clip(0,1)
      data.depthA[invalid] = 1

      invalid = data.depthB<0.1
      data.depthB = (data.depthB-min_val)/(max_val-min_val)
      data.depthB = data.depthB.clip(0,1)
      data.depthB[invalid] = 1
      return data

    if isinstance(data, list):
      for i in range(len(data)):
        data[i] = transform(data[i])
    else:
      data = transform(data)

    return data



class DepthToXyzmapAndNormalize:
  def __init__(self, H_ori=540, W_ori=720):
    self.H_ori = H_ori
    self.W_ori = W_ori

  def __call__(self, data):

    def transform(data):
      crop_to_ori = np.linalg.inv(data.tf_to_crop)
      invalid = (data.depthA<0.1)
      depth = cv2.warpPerspective(data.depthA, crop_to_ori, dsize=(self.W_ori, self.H_ori), flags=cv2.INTER_NEAREST)
      vs,us = np.where(depth>=0.1)
      uvs = np.stack([us,vs], axis=-1)
      xyz_map = depth2xyzmap(depth, data.K, uvs=uvs)
      data.xyz_mapA = cv2.warpPerspective(xyz_map, data.tf_to_crop, dsize=data.depthA.shape, flags=cv2.INTER_NEAREST)
      data.xyz_mapA -= data.poseA[:3,3].reshape(1,1,3)
      data.xyz_mapA[invalid] = -1

      invalid = (data.depthB<0.1)
      depth = cv2.warpPerspective(data.depthB, crop_to_ori, dsize=(self.W_ori, self.H_ori), flags=cv2.INTER_NEAREST)
      vs,us = np.where(depth>=0.1)
      uvs = np.stack([us,vs], axis=-1)
      xyz_map = depth2xyzmap(depth, data.K, uvs=uvs)
      data.xyz_mapB = cv2.warpPerspective(xyz_map, data.tf_to_crop, dsize=data.depthB.shape, flags=cv2.INTER_NEAREST)
      data.xyz_mapB -= data.poseA[:3,3].reshape(1,1,3)
      data.xyz_mapB[invalid] = -1
      return data

    if isinstance(data, list):
      for i in range(len(data)):
        data[i] = transform(data[i])
    else:
      data = transform(data)

    return data


class ChannelShuffle:
  def __init__(self, prob):
    self.prob = prob

  def __call__(self, data):
    if np.random.uniform(0,1)>=self.prob:
      return data

    cs = np.array([0,1,2])
    np.random.shuffle(cs)

    def transform(data):
      data.rgbA = data.rgbA[...,cs]
      data.rgbB = data.rgbB[...,cs]
      return data

    if isinstance(data, list):
      for i in range(len(data)):
        data[i] = transform(data[i])
    else:
      data = transform(data)

    return data


class RotateImgA:
  def __init__(self, prob, rot_range=[-30,30]):
    self.prob = prob
    self.rot_range = rot_range


  def transform(self, data):
    rot = np.random.uniform(self.rot_range[0], self.rot_range[1])
    aug = iaa.Rotate(rotate=rot, order=1, cval=0, mode='constant', fit_output=False, backend='cv2')
    data.rgbA = aug(images=[data.rgbA.astype(np.uint8)])[0]
    return data


  def __call__(self, data):
    if np.random.uniform(0,1)<self.prob:
      if isinstance(data, list):
        for i in range(len(data)):
          data[i] = self.transform(data[i])
      else:
        data = self.transform(data)

    return data


class ReplaceBackground:
  def __init__(self, img_root, prob):
    self.prob = prob
    self.rgb_files = sorted(glob.glob(f'{img_root}/**/*.jpg', recursive=True))
    logging.info(f"self.rgb_files:{len(self.rgb_files)}")

  def __call__(self, data):
    if np.random.uniform(0,1)>=self.prob:
      return data

    def transform(data, rgb_bg):
      data.rgbB[data.maskB==0] = rgb_bg[data.maskB==0]
      data.rgbB[data.crop_mask==0] = 0
      return data

    rgb_file = np.random.choice(self.rgb_files)
    rgb_bg = cv2.imread(rgb_file)[...,::-1]

    if isinstance(data, dict):
      H,W = data['rgb'].shape[:2]
      rgb_bg = cv2.resize(rgb_bg, (W,H))
      invalid = data['depth']<0.1
      data['rgb'][invalid] = rgb_bg[invalid]
      return data

    if isinstance(data, list):
      H,W = data[0].rgbB.shape[:2]
    else:
      H,W = data.rgbB.shape[:2]

    aug = iaa.CropToAspectRatio(aspect_ratio=W/H)
    rgb_bg = aug(images=[rgb_bg])[0]
    rgb_bg = cv2.resize(rgb_bg, (W,H))

    if isinstance(data, list):
      for i in range(len(data)):
        data[i] = transform(data[i], rgb_bg)
    else:
      data = transform(data, rgb_bg)

    return data



class RandomApply:
  def __init__(self, augmenter, prob):
    if not 0 <= prob <= 1:
      raise ValueError(f'prob must be in [0, 1], got {prob}')
    self.augmenter = augmenter
    self.prob = prob

  def __call__(self, data):
    if np.random.uniform(0, 1) >= self.prob:
      return data
    return self.augmenter(data)


class ComposedAugmenter:
  def __init__(self, augs):
    self.augs = augs

  def __call__(self, data):
    for aug in self.augs:
      logging.debug(f'{type(aug)} start')
      data = aug(data)
      logging.debug(f'{type(aug)} done')
    return data
