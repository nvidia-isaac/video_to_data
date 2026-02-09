from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class NlfResult:
    poses: np.ndarray  # (T, 72) or (T, 156)
    betas: np.ndarray  # (T, 10)
    transls: np.ndarray # (T, 3)
    gender: str
    frames: List[str]
    model_type: str  # "smpl" or "smplh"

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)


