# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys, torch, trimesh, logging, importlib, scipy
import nvdiffrast.torch as dr
import torch.nn.functional as F
import torchvision
from functools import lru_cache
import open3d as o3d
import cv2
import numpy as np
from collections import defaultdict
from transformations import *
from collections import OrderedDict
import ruamel.yaml
yaml = ruamel.yaml.YAML()
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
try:
  import kornia
except:
  kornia = None
try:
  import mycpp.build.mycpp as mycpp
except:
  mycpp = None
try:
  from bundlesdf.mycuda import common
except:
  common = None
try:
  import warp as wp
  wp.init()
except:
  wp = None
enable_timer = 0

def NestDict():
  return defaultdict(NestDict)

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

BAD_DEPTH = 99
BAD_COLOR = 0

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)

COLOR_MAP=np.array([[0, 0, 0], #Ignore
                    [128,0,0], #Background
                    [0,128,0], #Wall
                    [128,128,0], #Floor
                    [0,0,128], #Ceiling
                    [128,0,128], #Table
                    [0,128,128], #Chair
                    [128,128,128], #Window
                    [64,0,0], #Door
                    [192,0,0], #Monitor
                    [64, 128, 0],     # 11th
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    ])


def set_logging_format(level=logging.INFO):
  importlib.reload(logging)
  FORMAT = '[%(funcName)s()] %(message)s'
  logging.basicConfig(level=level, format=FORMAT)

set_logging_format()


def make_mesh_tensors(mesh, device='cuda', max_tex_size=None):
  mesh_tensors = {}
  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and mesh.visual.material.image is not None:
    img = np.array(mesh.visual.material.image.convert('RGB'))
    img = img[...,:3]
    if max_tex_size is not None:
      max_size = max(img.shape[0], img.shape[1])
      if max_size>max_tex_size:
        scale = 1/max_size * max_tex_size
        img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
    mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None]/255.0
    mesh_tensors['uv_idx']  = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
    uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
    uv[:,1] = 1 - uv[:,1]
    mesh_tensors['uv']  = uv
  else:
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
      # create a color visual with grey color 
      logging.info(f"WARN: mesh doesn't have texture, assigning a pure color")
      mesh.visual = trimesh.visual.color.ColorVisuals(vertex_colors=np.tile(np.array([128,128,128]).reshape(1,3), (len(mesh.vertices), 1)))
    if mesh.visual.vertex_colors is None:
      logging.info(f"WARN: mesh doesn't have vertex_colors, assigning a pure color")
      mesh.visual.vertex_colors = np.tile(np.array([128,128,128]).reshape(1,3), (len(mesh.vertices), 1))
    mesh_tensors['vertex_color'] = torch.as_tensor(mesh.visual.vertex_colors[...,:3], device=device, dtype=torch.float)/255.0
    print("vertex colors first 3:", mesh_tensors['vertex_color'][:3])

  mesh_tensors.update({
    'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
    'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
    'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
  })
  return mesh_tensors


def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
  '''Just plain rendering, not support any gradient
  @K: (3,3) np array
  @ob_in_cams: (N,4,4) torch tensor, openCV camera
  @projection_mat: np array (4,4)
  @output_size: (height, width)
  @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
  @light_dir: in cam space
  @light_pos: in cam space
  '''
  if glctx is None:
    if context == 'gl':
      glctx = dr.RasterizeGLContext()
    elif context=='cuda':
      glctx = dr.RasterizeCudaContext()
    else:
      raise NotImplementedError
    logging.info("created context")

  if mesh_tensors is None:
    mesh_tensors = make_mesh_tensors(mesh)
  pos = mesh_tensors['pos']
  vnormals = mesh_tensors['vnormals']
  pos_idx = mesh_tensors['faces']
  has_tex = 'tex' in mesh_tensors
  batched_positions = pos.ndim == 3
  if batched_positions and pos.shape[0] != ob_in_cams.shape[0]:
    raise ValueError(f'Batched mesh positions and camera transforms differ: {pos.shape[0]} != {ob_in_cams.shape[0]}')
  if pos.ndim not in (2, 3):
    raise ValueError(f'Mesh positions must have shape [V,3] or [B,V,3], got {tuple(pos.shape)}')

  ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None]@ob_in_cams
  if projection_mat is None:
    if len(K.shape) == 2:
      projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100)
    elif len(K.shape) == 3:
      # K_rois
      projs = [projection_matrix_from_intrinsics(x, height=H, width=W, znear=0.001, zfar=100) for x in K]
      projection_mat = np.stack(projs, axis=0) # (B, 4, 4)
    else:
      raise NotImplementedError

  projection_mat = torch.as_tensor(projection_mat.reshape(-1,4,4), device='cuda', dtype=torch.float)
  mtx = projection_mat@ob_in_glcams # (1, 4, 4)@(B, 4, 4) -> (B, 4, 4)
  if output_size is None:
    output_size = np.asarray([H,W])

  pts_cam = transform_pts(pos, ob_in_cams)
  pos_homo = to_homo_torch(pos)
  pos_clip = (mtx[:,None]@pos_homo[...,None])[...,0] if batched_positions else (mtx[:,None]@pos_homo[None,...,None])[...,0]
  if bbox2d is not None:
    l = bbox2d[:,0]
    t = H-bbox2d[:,1]
    r = bbox2d[:,2]
    b = H-bbox2d[:,3]
    tf = torch.eye(4, dtype=torch.float, device='cuda').reshape(1,4,4).expand(len(ob_in_cams),4,4).contiguous()
    tf[:,0,0] = W/(r-l)
    tf[:,1,1] = H/(t-b)
    tf[:,3,0] = (W-r-l)/(r-l)
    tf[:,3,1] = (H-t-b)/(t-b)
    pos_clip = pos_clip@tf
  # rast_out: (B, H, W, 4), where [..., -1] is the triangle id
  rast_out, _1 = dr.rasterize(glctx, pos_clip, pos_idx, resolution=np.asarray(output_size))
  xyz_map, _2 = dr.interpolate(pts_cam, rast_out, pos_idx)
  depth = xyz_map[...,2]

  if has_tex:
    uv = mesh_tensors['uv']
    if batched_positions and uv.ndim == 2:
      uv = uv[None].expand(pos.shape[0], -1, -1).contiguous()
    if uv.ndim not in (2, 3) or (uv.ndim == 3 and uv.shape[0] != pos.shape[0]):
      raise ValueError(f'Texture coordinates must have shape [U,2] or [B,U,2], got {tuple(uv.shape)}')
    texc, _ = dr.interpolate(uv.contiguous(), rast_out, mesh_tensors['uv_idx'].contiguous())
    color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
  else:
    color, _ = dr.interpolate(mesh_tensors['vertex_color'], rast_out, pos_idx)

  if use_light:
    get_normal = True
  if get_normal:
    vnormals_cam = transform_dirs(vnormals, ob_in_cams)
    normal_map, _ = dr.interpolate(vnormals_cam, rast_out, pos_idx)
    normal_map = F.normalize(normal_map, dim=-1)
    normal_map = torch.flip(normal_map, dims=[1])
  else:
    normal_map = None

  if use_light:
    if light_dir is not None:
      light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float, device='cuda')
    else:
      light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float, device='cuda').reshape(1,1,3) - pts_cam
    diffuse_intensity = (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[...,None]
    diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out, pos_idx)  # (N_pose, H, W, 1)
    if light_color is None:
      light_color = color
    else:
      light_color = torch.as_tensor(light_color, device='cuda', dtype=torch.float)
    color = color*w_ambient + diffuse_intensity_map*light_color*w_diffuse

  color = color.clip(0,1)
  color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background using alpha
  color = torch.flip(color, dims=[1])   # Flip Y coordinates
  depth = torch.flip(depth, dims=[1])
  extra['xyz_map'] = torch.flip(xyz_map, dims=[1])
  return color, depth, normal_map


def nvdiff_rasterize(glctx, mesh_tensors, pos_clip, render_size):
  "pos_clip: (B, N, 4), already projected points using projection matrix"
  pos_idx = mesh_tensors['faces'].contiguous()
  rast_out, _ = dr.rasterize(glctx, pos_clip.contiguous(), pos_idx, resolution=np.asarray(render_size))
  rast_out = rast_out.contiguous()
  has_tex = 'tex' in mesh_tensors
  if has_tex:
    texc, _ = dr.interpolate(mesh_tensors['uv'].contiguous(), rast_out, mesh_tensors['uv_idx'].contiguous())
    color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
  else:
    color, _ = dr.interpolate(mesh_tensors['vertex_color'].contiguous(), rast_out, pos_idx)
  color = color.clip(0, 1)
  color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background using alpha
  color = torch.flip(color, dims=[1])  # Flip Y coordinates
  return color


def nvdiff_color_depth_render(Krois, glctx, mesh_tensors, render_size, pts_cam, depth_only=False):
  "render both color and depth"
  ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None].repeat(len(Krois), 1, 1)
  projs = [projection_matrix_from_intrinsics(x, height=render_size[0], width=render_size[1], znear=0.001, zfar=100) for x in Krois]
  projection_mat = torch.as_tensor(np.stack(projs, axis=0).reshape(-1, 4, 4), device='cuda', dtype=torch.float)
  mtx = projection_mat @ ob_in_glcams  # (1, 4, 4)@(B, 4, 4) -> (B, 4, 4)
  pos_homo = to_homo_torch(pts_cam)
  pos_clip = (mtx[:, None] @ pos_homo[..., None])[..., 0] # (B, N, 4)
  
  # rasterization 
  pos_idx = mesh_tensors['faces']
  rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=np.asarray(render_size))

  # obtain depth
  # compute points in camera space (opencv format)
  xyz_map, _2 = dr.interpolate(pts_cam, rast_out, pos_idx)
  depth = xyz_map[...,2]
  depth = torch.flip(depth, dims=[1])  # Flip Y coordinates
  xyz_map = torch.flip(xyz_map, dims=[1])  # Flip Y coordinates
  if depth_only:
    return depth, xyz_map

  # obtain color 
  has_tex = 'tex' in mesh_tensors
  if has_tex:
    texc, _ = dr.interpolate(mesh_tensors['uv'], rast_out, mesh_tensors['uv_idx'])
    color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
  else:
    color, _ = dr.interpolate(mesh_tensors['vertex_color'], rast_out, pos_idx)
  color = color.clip(0, 1)
  color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background using alpha
  color = torch.flip(color, dims=[1])  # Flip Y coordinates

  return color, depth, xyz_map
  


def set_seed(random_seed):
  import torch,random
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def geodesic_distance_batch(R1,R2):
  '''
  @R1: torch tensor (...,3,3)
  '''
  shape = R1.shape
  ones = [1 for _ in shape[:-2]]
  trace = (R1.float()@R2.float().transpose(-2,-1))*torch.eye(3).reshape(*ones,3,3).float().to(R1.device)  #(...,3,3)
  shape = trace.shape
  trace = trace.reshape(*shape[:-2],9).sum(dim=-1)
  cos = (trace-1)/2
  cos = cos.clip(-1,1)
  return torch.acos(cos)


def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud



def make_grid_image(imgs, nrow, padding=5, pad_value=255):
  '''
  @imgs: (B,H,W,C) np array
  @nrow: num of images per row
  '''
  grid = torchvision.utils.make_grid(torch.as_tensor(np.asarray(imgs)).permute(0,3,1,2), nrow=nrow, padding=padding, pad_value=pad_value)
  grid = grid.permute(1,2,0).contiguous().data.cpu().numpy().astype(np.uint8)
  return grid


if wp is not None:
  @wp.kernel(enable_backward=False)
  def bilateral_filter_depth_kernel(depth:wp.array(dtype=float, ndim=2), out:wp.array(dtype=float, ndim=2), radius:int, zfar:float, sigmaD:float, sigmaR:float):
    h,w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w>=W or h>=H:
      return
    out[h,w] = 0.0
    mean_depth = float(0.0)
    num_valid = int(0)
    for u in range(w-radius, w+radius+1):
      if u<0 or u>=W:
        continue
      for v in range(h-radius, h+radius+1):
        if v<0 or v>=H:
          continue
        cur_depth = depth[v,u]
        if cur_depth>=0.001 and cur_depth<zfar:
          num_valid += 1
          mean_depth += cur_depth
    if num_valid==0:
      return
    mean_depth /= float(num_valid)

    depthCenter = depth[h,w]
    sum_weight = float(0.0)
    sum = float(0.0)
    for u in range(w-radius, w+radius+1):
      if u<0 or u>=W:
        continue
      for v in range(h-radius, h+radius+1):
        if v<0 or v>=H:
          continue
        cur_depth = depth[v,u]
        if cur_depth>=0.001 and cur_depth<zfar and abs(cur_depth-mean_depth)<0.01:
          weight = wp.exp( -float((u-w)*(u-w) + (h-v)*(h-v)) / (2.0*sigmaD*sigmaD) - (depthCenter-cur_depth)*(depthCenter-cur_depth)/(2.0*sigmaR*sigmaR) )
          sum_weight += weight
          sum += weight*cur_depth
    if sum_weight>0 and num_valid>0:
      out[h,w] = sum/sum_weight

  def bilateral_filter_depth(depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device='cuda'):
    if isinstance(depth, np.ndarray):
      depth_wp = wp.array(depth, dtype=float, device=device)
    else:
      depth_wp = wp.from_torch(depth)
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(kernel=bilateral_filter_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]], inputs=[depth_wp, out_wp, radius, zfar, sigmaD, sigmaR])
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
      depth_out = depth_out.data.cpu().numpy()
    return depth_out


  @wp.kernel(enable_backward=False)
  def erode_depth_kernel(depth:wp.array(dtype=float, ndim=2), out:wp.array(dtype=float, ndim=2), radius:int, depth_diff_thres:float, ratio_thres:float, zfar:float):
    h,w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w>=W or h>=H:
      return
    d_ori = depth[h,w]
    if d_ori<0.001 or d_ori>=zfar:
      out[h,w] = 0.0
    bad_cnt = float(0)
    total = float(0)
    for u in range(w-radius, w+radius+1):
      if u<0 or u>=W:
        continue
      for v in range(h-radius, h+radius+1):
        if v<0 or v>=H:
          continue
        cur_depth = depth[v,u]
        total += 1.0
        if cur_depth<0.001 or cur_depth>=zfar or abs(cur_depth-d_ori)>depth_diff_thres:
          bad_cnt += 1.0
    if bad_cnt/total>ratio_thres:
      out[h,w] = 0.0
    else:
      out[h,w] = d_ori


  def erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100, device='cuda'):
    depth_wp = wp.from_torch(torch.as_tensor(depth, dtype=torch.float, device=device))
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(kernel=erode_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]], inputs=[depth_wp, out_wp, radius, depth_diff_thres, ratio_thres, zfar],)
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
      depth_out = depth_out.data.cpu().numpy()
    return depth_out



@lru_cache(maxsize=16)
def _depth2xyzmap_grid(height, width):
  vs,us = np.meshgrid(np.arange(0,height),np.arange(0,width), sparse=False, indexing='ij')
  return vs.reshape(-1),us.reshape(-1)


def depth2xyzmap(depth, K, uvs=None):
  invalid_mask = (depth<0.001)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = _depth2xyzmap_grid(H,W)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  xyz_map[invalid_mask] = 0
  return xyz_map


def depth2xyzmap_batch(depths, Ks, zfar):
  '''
  @depths: torch tensor (B,H,W)
  @Ks: torch tensor (B,3,3)
  '''
  bs = depths.shape[0]
  invalid_mask = (depths<0.001) | (depths>zfar)
  H,W = depths.shape[-2:]
  vs,us = torch.meshgrid(torch.arange(0,H),torch.arange(0,W), indexing='ij')
  vs = vs.reshape(-1).float().cuda()[None].expand(bs,-1)
  us = us.reshape(-1).float().cuda()[None].expand(bs,-1)
  zs = depths.reshape(bs,-1)
  Ks = Ks[:,None].expand(bs,zs.shape[-1],3,3)
  xs = (us-Ks[...,0,2])*zs/Ks[...,0,0]  #(B,N)
  ys = (vs-Ks[...,1,2])*zs/Ks[...,1,1]
  pts = torch.stack([xs,ys,zs], dim=-1)  #(B,N,3)
  xyz_maps = pts.reshape(bs,H,W,3)
  xyz_maps[invalid_mask] = 0
  return xyz_maps



def depth_to_vis(depth, zmin=None, zmax=None, mode='rgb', inverse=True):
  if zmin is None:
    zmin = depth.min()
  if zmax is None:
    zmax = depth.max()

  if inverse:
    invalid = depth<0.001
    vis = zmin/(depth+1e-8)
    vis[invalid] = 0
  else:
    depth = depth.clip(zmin, zmax)
    invalid = (depth==zmin) | (depth==zmax)
    vis = (depth-zmin)/(zmax-zmin)
    vis[invalid] = 1

  if mode=='gray':
    vis = (vis*255).clip(0, 255).astype(np.uint8)
  elif mode=='rgb':
    vis = cv2.applyColorMap((vis*255).astype(np.uint8), cv2.COLORMAP_JET)[...,::-1]
  else:
    raise RuntimeError

  return vis



def sample_views_icosphere(n_views, subdivisions=None, radius=1):
  if subdivisions is not None:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
  else:
    subdivision = 1
    while 1:
      mesh = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
      if mesh.vertices.shape[0]>=n_views:
        break
      subdivision += 1
  cam_in_obs = np.tile(np.eye(4)[None], (len(mesh.vertices),1,1))
  cam_in_obs[:,:3,3] = mesh.vertices
  up = np.array([0,0,1])
  z_axis = -cam_in_obs[:,:3,3]  #(N,3)
  z_axis /= np.linalg.norm(z_axis, axis=-1).reshape(-1,1)
  x_axis = np.cross(up.reshape(1,3), z_axis)
  invalid = (x_axis==0).all(axis=-1)
  x_axis[invalid] = [1,0,0]
  x_axis /= np.linalg.norm(x_axis, axis=-1).reshape(-1,1)
  y_axis = np.cross(z_axis, x_axis)
  y_axis /= np.linalg.norm(y_axis, axis=-1).reshape(-1,1)
  cam_in_obs[:,:3,0] = x_axis
  cam_in_obs[:,:3,1] = y_axis
  cam_in_obs[:,:3,2] = z_axis
  return cam_in_obs



def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo


def to_homo_torch(pts):
  '''
  @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
  '''
  ones = torch.ones((*pts.shape[:-1],1), dtype=torch.float, device=pts.device)
  homo = torch.cat((pts, ones),dim=-1)
  return homo


def transform_pts(pts,tf):
  """Transform 2d or 3d points
  @pts: (...,N_pts,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=pts.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]


def transform_dirs(dirs,tf):
  """
  @dirs: (...,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=dirs.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:3,:3]@dirs[...,None])[...,0]



FOUNDATIONPOSE_MESH_DIAMETER_RANDOM_SEED = 0


def compute_mesh_diameter(model_pts=None, mesh=None, n_sample=1000, random_seed=None):
  if mesh is not None:
    u, s, vh = scipy.linalg.svd(mesh.vertices, full_matrices=False)
    pts = u@s
    diameter = np.linalg.norm(pts.max(axis=0)-pts.min(axis=0))
    return float(diameter)

  if n_sample is None:
    pts = model_pts
  else:
    rng = np.random if random_seed is None else np.random.RandomState(random_seed)
    ids = rng.choice(len(model_pts), size=min(n_sample, len(model_pts)), replace=False)
    pts = model_pts[ids]
  dists = np.linalg.norm(pts[None]-pts[:,None], axis=-1)
  diameter = dists.max()
  return diameter



def compute_tf_batch(left, right, top, bottom, out_size):
  "all has the shape (B,), torch tensor, out_size: the final crop size, tuple of 2"
  B = len(left)
  left = left.round()
  right = right.round()
  top = top.round()
  bottom = bottom.round()

  tf = torch.eye(3)[None].expand(B,-1,-1).contiguous()
  tf[:,0,2] = -left
  tf[:,1,2] = -top
  new_tf = torch.eye(3)[None].expand(B,-1,-1).contiguous()
  new_tf[:,0,0] = out_size[0]/(right-left)
  new_tf[:,1,1] = out_size[1]/(bottom-top)
  tf = new_tf@tf
  return tf

def compute_crop_window_tf_batch(pts=None, H=None, W=None, poses=None, K=None, crop_ratio=1.2, out_size=None,
                                 rgb=None, uvs=None, method='min_box', mesh_diameter=None, ret_Kroi=False):
  '''Project the points and find the cropping transform
  @pts: (N,3)
  @poses: (B,4,4) tensor
  @min_box: min_box/min_circle
  @scale: scale to apply to the tightly enclosing roi
  '''

  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  if method=='box_3d':
    bottom, left, right, top, crop_radius = compute_bbox_2d_batch(K, crop_ratio, mesh_diameter, poses)
    tfs = compute_tf_batch(left, right, top, bottom, out_size)

    # compute K_rois as well
    if ret_Kroi:
      bs = len(poses)
      focal = torch.tensor([[K[0, 0], K[1,1]]]).repeat(bs, 1)
      principal_point= torch.tensor([[K[0, 2], K[1,2]]]).repeat(bs, 1)
      scale = out_size[0] / (crop_radius*2) # (B, )
      focal_roi = focal * scale.reshape(bs, 1)
      top_left = torch.stack([left, top], -1)
      principal_roi = (principal_point - top_left) * scale.reshape(bs, 1)
      Krois = torch.eye(3)[None].repeat(bs, 1, 1).contiguous()
      Krois[:, 0, 0] = focal_roi[:, 0]
      Krois[:, 1, 1] = focal_roi[:, 1]
      Krois[:, 0, 2] = principal_roi[:, 0]
      Krois[:, 1, 2] = principal_roi[:, 1]
      Krois[:, 1, 2] = principal_roi[:, 1]
      return tfs, Krois

    return tfs

  else:
    raise RuntimeError


def compute_bbox_2d_batch(K, crop_ratio, mesh_diameter, poses):
  "input poses are batchfied, (B, 4, 4)"
  B = len(poses)
  radius = mesh_diameter * crop_ratio / 2
  offsets = torch.tensor([0, 0, 0,
                          radius, 0, 0,
                          -radius, 0, 0,
                          0, radius, 0,
                          0, -radius, 0]).reshape(-1, 3)
  pts = poses[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3).float()  # camera translation
  K = torch.as_tensor(K).float()
  projected = (K @ pts.reshape(-1, 3).T).T
  uvs = projected[:, :2] / projected[:, 2:3]
  uvs = uvs.reshape(B, -1, 2)
  center = uvs[:, 0]  # (B,2)
  radius = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(B, -1).max(axis=-1)[0].reshape(-1)  # (B)
  left = center[:, 0] - radius
  right = center[:, 0] + radius
  top = center[:, 1] - radius
  bottom = center[:, 1] + radius
  return bottom, left, right, top, radius


def cv_draw_text(img,text,uv_top_left,color=(255, 255, 255),fontScale=0.5,thickness=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,outline_color=None,line_spacing=1.5):
  H,W = img.shape[:2]
  uv_top_left = np.array(uv_top_left, dtype=float)
  assert uv_top_left.shape == (2,)

  for line in text.splitlines():
    (w, h), _ = cv2.getTextSize(text=line,fontFace=fontFace,fontScale=fontScale,thickness=thickness,)
    uv_bottom_left_i = uv_top_left + [0, h]

    ############# Ensure inside image
    while uv_bottom_left_i[0]<0:
      uv_bottom_left_i[0] += 1
    while uv_bottom_left_i[0]+w>=W:
      uv_bottom_left_i[0] -= 1
    while uv_bottom_left_i[1]>=H:
      uv_bottom_left_i[1] -= 1
    while uv_bottom_left_i[1]-h<0:
      uv_bottom_left_i[1] += 1

    org = tuple(uv_bottom_left_i.astype(int))

    if outline_color is not None:
      cv2.putText(img,text=line,org=org,fontFace=fontFace,fontScale=fontScale,color=outline_color,thickness=thickness,lineType=cv2.LINE_AA,)
    cv2.putText(img,text=line,org=org,fontFace=fontFace,fontScale=fontScale,color=color,thickness=thickness,lineType=cv2.LINE_AA,)
    uv_top_left[1] =  uv_bottom_left_i[1]-h+h*line_spacing
  return img


def project_3d_to_2d(pt,K,ob_in_cam):
  pt = pt.reshape(4,1)
  projected = K @ ((ob_in_cam@pt)[:3,:])
  projected = projected.reshape(-1)
  projected = projected/projected[2]
  return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(color, ob_in_cam, scale=0.1, K=np.eye(3), thickness=3, transparency=0,is_input_rgb=False):
  '''
  @color: BGR
  '''
  if is_input_rgb:
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
  xx = np.array([1,0,0,1]).astype(float)
  yy = np.array([0,1,0,1]).astype(float)
  zz = np.array([0,0,1,1]).astype(float)
  xx[:3] = xx[:3]*scale
  yy[:3] = yy[:3]*scale
  zz[:3] = zz[:3]*scale
  origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
  xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
  yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
  zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
  line_type = cv2.LINE_AA
  arrow_len = 0
  tmp = color.copy()
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp = tmp.astype(np.uint8)
  if is_input_rgb:
    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

  return tmp


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0,255,0), linewidth=2):
  '''Revised from 6pack dataset/inference_dataset_nocs.py::projection
  @bbox: (2,3) min/max
  @line_color: RGB
  '''
  min_xyz = bbox.min(axis=0)
  xmin, ymin, zmin = min_xyz
  max_xyz = bbox.max(axis=0)
  xmax, ymax, zmax = max_xyz

  def draw_line3d(start,end,img):
    pts = np.stack((start,end),axis=0).reshape(-1,3)
    pts = (ob_in_cam@to_homo(pts).T).T[:,:3]   #(2,3)
    projected = (K@pts.T).T
    uv = np.round(projected[:,:2]/projected[:,2].reshape(-1,1)).astype(int)   #(2,2)
    img = cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)
    return img

  for y in [ymin,ymax]:
    for z in [zmin,zmax]:
      start = np.array([xmin,y,z])
      end = start+np.array([xmax-xmin,0,0])
      img = draw_line3d(start,end,img)

  for x in [xmin,xmax]:
    for z in [zmin,zmax]:
      start = np.array([x,ymin,z])
      end = start+np.array([0,ymax-ymin,0])
      img = draw_line3d(start,end,img)

  for x in [xmin,xmax]:
    for y in [ymin,ymax]:
      start = np.array([x,y,zmin])
      end = start+np.array([0,0,zmax-zmin])
      img = draw_line3d(start,end,img)

  return img


def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords='y_down'):
  """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

  Ref:
  1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
  2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

  :param K: 3x3 ndarray with the intrinsic camera matrix.
  :param x0 The X coordinate of the camera image origin (typically 0).
  :param y0: The Y coordinate of the camera image origin (typically 0).
  :param w: Image width.
  :param h: Image height.
  :param nc: Near clipping plane.
  :param fc: Far clipping plane.
  :param window_coords: 'y_up' or 'y_down'.
  :return: 4x4 ndarray with the OpenGL projection matrix.
  """
  x0 = 0
  y0 = 0
  w = width
  h = height
  nc = znear
  fc = zfar

  depth = float(fc - nc)
  q = -(fc + nc) / depth
  qn = -2 * (fc * nc) / depth

  # Draw our images upside down, so that all the pixel-based coordinate
  # systems are the same.
  if window_coords == 'y_up':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])

  # Draw the images upright and modify the projection matrix so that OpenGL
  # will generate window coords that compensate for the flipped image coords.
  elif window_coords == 'y_down':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])
  else:
    raise NotImplementedError

  return proj



def symmetry_spec_from_info(info, rot_angle_discrete=5):
  from lib_mhr.object_symmetry import bop_object_symmetry

  return bop_object_symmetry(info, continuous_angle_degrees=rot_angle_discrete)


def egocentric_delta_pose_to_pose(A_in_cam, trans_delta, rot_mat_delta):
  '''Used for Pose Refinement. Given the object's two poses in camera, convert them to relative poses in camera's egocentric view
  @A_in_cam: (B,4,4) torch tensor
  '''
  B_in_cam = torch.eye(4, dtype=torch.float, device=A_in_cam.device)[None].expand(len(A_in_cam),-1,-1).contiguous()
  B_in_cam[:,:3,3] = A_in_cam[:,:3,3]+trans_delta
  B_in_cam[:,:3,:3] = rot_mat_delta@A_in_cam[:,:3,:3]
  return B_in_cam


def make_yaml_dumpable(D):
  if isinstance(D, np.ndarray):
    return D.tolist()
  for d in D:
    if isinstance(D[d], dict) or isinstance(D[d], OrderedDict) or isinstance(D[d], defaultdict):
      D[d] = dict(D[d])
      D[d] = make_yaml_dumpable(D[d])
      continue
    if isinstance(D[d], np.ndarray):
      D[d] = D[d].tolist()
      continue
    if np.issubdtype(type(D[d]), int):
      D[d] = int(D[d])
      continue
    if np.issubdtype(type(D[d]), float):
      D[d] = float(D[d])
      continue
    if np.issubdtype(type(D[d]), str):
      D[d] = str(D[d])
      continue
    if isinstance(D[d], list):
      for i in range(len(D[d])):
        D[d][i] = make_yaml_dumpable(D[d][i])
      continue
  return dict(D)
