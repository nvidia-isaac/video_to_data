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
