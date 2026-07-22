# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small subset of :mod:`mmcv.cnn` used by E2FGVI inference."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.nn.init as init


def constant_init(module: Any, val: float, bias: float = 0.0) -> None:
    init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    """Small stand-in preserving MMCV's ``conv`` checkpoint key names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        norm_cfg: dict[str, Any] | None = None,
        act_cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if norm_cfg is not None:
            raise NotImplementedError("the E2FGVI compatibility ConvModule supports norm_cfg=None only")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported ConvModule argument(s): {unexpected}")

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        if act_cfg is None:
            self.activate = nn.Identity()
        elif act_cfg.get("type") == "ReLU":
            self.activate = nn.ReLU(inplace=act_cfg.get("inplace", True))
        else:
            raise NotImplementedError(f"unsupported activation config: {act_cfg}")

    def forward(self, value: Any) -> Any:
        return self.activate(self.conv(value))
