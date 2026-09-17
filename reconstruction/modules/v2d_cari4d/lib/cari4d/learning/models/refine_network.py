# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')
from Utils import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from functools import partial
from network_modules import *
from Utils import *



class RefineNet(nn.Module):
  def __init__(self, cfg=None, c_in=4, n_view=1):
    super().__init__()
    self.cfg = cfg
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
      norm_layer1d = nn.BatchNorm1d
    else:
      norm_layer = None
      norm_layer1d = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    posi_len = 400 if 'posi_len' not in cfg else cfg['posi_len']
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=posi_len)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, 3),
    )

    if self.cfg['rot_rep']=='axis_angle':
      rot_out_dim = 3
    elif self.cfg['rot_rep']=='6d':
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, rot_out_dim),
    )
    self._init_others()

  def _init_others(self):
    return

  def forward(self, A, B, pose=None):
    """
    @A: (B,C,H,W)

    before encodeAB: torch.Size([1, 256, 40, 40])
    after encodeAB: torch.Size([1, 512, 20, 20])
    after pose_embed: torch.Size([1, 400, 512])
    Inside refinenet forward: A=torch.Size([1, 6, 160, 160]), B=torch.Size([1, 6, 160, 160]), ab=torch.Size([1, 400, 512])

    compression:
ConvBNReLU in torch.Size([192, 6, 224, 224]), out: torch.Size([192, 64, 112, 112])
ConvBNReLU in torch.Size([192, 64, 112, 112]), out: torch.Size([192, 128, 56, 56])
ConvBNReLU in torch.Size([96, 256, 56, 56]), out: torch.Size([96, 512, 28, 28])

    """
    bs = len(A)
    output = {}

    x = torch.cat([A,B], dim=0) # shared encoder
    x = self.encodeA(x)
    a = x[:bs]
    b = x[bs:] # B, F, H, W

    ab = torch.cat((a,b),1).contiguous() # [1, 256, 40, 40]
    ab = self.encodeAB(ab)  #(B,C,H,W), [1, 512, 20, 20]


    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
    # ab: [252, 400, 512] (initialization), or (1, 400, 512) (refinement)

    output['trans'] = self.trans_head(ab).mean(dim=1)
    output['rot'] = self.rot_head(ab).mean(dim=1)

    return output
