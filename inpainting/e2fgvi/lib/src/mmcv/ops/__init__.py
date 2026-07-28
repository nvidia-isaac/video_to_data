# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torchvision-backed subset of :mod:`mmcv.ops` used by E2FGVI."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


class ModulatedDeformConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value: Any, offset: Any, mask: Any) -> Any:
        return modulated_deform_conv2d(
            value,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )


def modulated_deform_conv2d(
    value: Any,
    offset: Any,
    mask: Any,
    weight: Any,
    bias: Any,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
    groups: int,
    deform_groups: int,
) -> Any:
    """Map MMCV's operator signature to torchvision's equivalent operator."""
    del groups, deform_groups  # torchvision infers these values from tensor shapes.
    from torchvision.ops import deform_conv2d

    return deform_conv2d(
        value,
        offset,
        weight,
        bias=bias,
        stride=_pair(stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        mask=mask,
    )
