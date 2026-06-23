# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import numpy as np


class ImageProcessorBase(ABC):
    """Base class for image processors."""

    @abstractmethod
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def map_points(self, pts: np.ndarray) -> np.ndarray:
        """Map Nx2 pixel coordinates forward through this processing step."""
        raise NotImplementedError


class ImagePipeline:
    """Sequential composition of image processors."""

    def __init__(self):
        self.processors: list[ImageProcessorBase] = []

    def add_processor(self, processor: ImageProcessorBase):
        self.processors.append(processor)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for processor in self.processors:
            img = processor(img)
        return img

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        """Map Nx2 pixel coordinates forward through all processing steps."""
        for processor in self.processors:
            pts = processor.map_points(pts)
        return pts
