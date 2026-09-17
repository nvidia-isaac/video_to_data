# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch.nn as nn


class Encoder16x16(nn.Module):
    "takes dino Dx16x16 feature as input, and output D_out feature vector"
    def __init__(self, cin, cout, nf=256, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
        ]
        assert activation is None
        self.network = nn.Sequential(*network)

    def forward(self, input):
        # Usage example:
        # HW_down = int(math.sqrt(T - 1))
        # feat_out = feats[:, 1:, :].reshape(B, HW_down, HW_down, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        # cls_fuse = self.feat_fuser(feat_out)  # (B, D)
        return self.network(input).reshape(input.size(0), -1)