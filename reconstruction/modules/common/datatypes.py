from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class DepthImage:
    depth: np.ndarray

    def to_pil_image(self) -> Image.Image:
        # Encode depth (meters) as uint16 PNG using inverse depth:
        # pixel = 65535 * (1 / (depth_m + 1))
        # Closer objects → higher pixel values; zero depth → 65535; infinity → 0.
        inverse_depth = 1.0 / (self.depth + 1.0)
        inverse_depth = 65535.0 * inverse_depth
        inverse_depth = inverse_depth.clip(0, 65535).astype(np.uint16)
        return Image.fromarray(inverse_depth, mode="I;16")

    @staticmethod
    def from_pil_image(pil_image: Image.Image) -> 'DepthImage':
        # Decode inverse-depth uint16 PNG back to depth in meters.
        inverse_depth = np.array(pil_image).astype(np.float32)
        depth_m = 1.0 / (inverse_depth / 65535.0) - 1.0
        return DepthImage(depth=depth_m)
    
    def width(self) -> int:
        return self.depth.shape[1]
    
    def height(self) -> int:
        return self.depth.shape[0]
    
    @staticmethod
    def load(path: str) -> 'DepthImage':
        return DepthImage.from_pil_image(Image.open(path))
    
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    
    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height
        }
    @staticmethod
    def from_dict(d: dict) -> 'CameraIntrinsics':
        return CameraIntrinsics(fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"], width=d["width"], height=d["height"])

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

@dataclass
class Transform3d:
    rotation: list[float]
    translation: list[float]
    scale: list[float]

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation,
            "translation": self.translation,
            "scale": self.scale
        }

    @staticmethod
    def from_dict(d: dict) -> 'Transform3d':
        return Transform3d(
            rotation=d["rotation"],
            translation=d["translation"],
            scale=d["scale"]
        )


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    
    def to_dict(self) -> dict:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'BoundingBox':
        return BoundingBox(x0=d["x0"], y0=d["y0"], x1=d["x1"], y1=d["y1"])


@dataclass
class Point:
    x: float
    y: float
    
    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'Point':
        return Point(x=d["x"], y=d["y"])


@dataclass
class Mask:
    mask: np.ndarray

    def to_pil_image(self) -> Image.Image:
        return Image.fromarray(self.mask * 255, mode="L")
    
    @staticmethod
    def from_pil_image(pil_image: Image.Image) -> 'Mask':
        return Mask(mask=np.array(pil_image) / 255)
    
    def width(self) -> int:
        return self.mask.shape[1]
    
    def height(self) -> int:
        return self.mask.shape[0]

    @staticmethod
    def load(path: str) -> 'Mask':
        return Mask.from_pil_image(Image.open(path))

