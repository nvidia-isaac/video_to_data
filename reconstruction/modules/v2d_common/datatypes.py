from dataclasses import dataclass
import json
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image as PILImage


@dataclass
class DepthImage:
    depth: np.ndarray

    def to_pil_image(self) -> PILImage.Image:
        # Encode depth (meters) as uint16 PNG using inverse depth:
        # pixel = 65535 * (1 / (depth_m + 1))
        # Closer objects → higher pixel values; zero depth → 65535; infinity → 0.
        inverse_depth = 1.0 / (self.depth + 1.0)
        inverse_depth = 65535.0 * inverse_depth
        inverse_depth = inverse_depth.clip(0, 65535).astype(np.uint16)
        return PILImage.fromarray(inverse_depth, mode="I;16")

    @staticmethod
    def from_pil_image(pil_image: PILImage.Image) -> 'DepthImage':
        # Decode inverse-depth uint16 PNG back to depth in meters.
        inverse_depth = np.array(pil_image).astype(np.float32)
        depth_m = 1.0 / (inverse_depth / 65535.0) - 1.0
        return DepthImage(depth=depth_m)

    def width(self) -> int:
        return self.depth.shape[1]

    def height(self) -> int:
        return self.depth.shape[0]

    def save(self, path: str) -> None:
        self.to_pil_image().save(path)

    @staticmethod
    def load(path: str) -> 'DepthImage':
        return DepthImage.from_pil_image(PILImage.open(path))

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

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(path: str) -> 'CameraIntrinsics':
        with open(path) as f:
            return CameraIntrinsics.from_dict(json.load(f))

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

    def scale_only(self) -> 'Transform3d':
        return Transform3d(rotation=[1.0, 0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0], scale=self.scale)

    def to_matrix(self) -> np.ndarray:
        """Build a 4×4 homogeneous matrix (scale → rotate → translate)."""
        w, x, y, z = self.rotation
        sx, sy, sz = self.scale
        tx, ty, tz = self.translation
        R = np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y],
            [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
            [2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y],
        ], dtype=np.float64)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R @ np.diag([sx, sy, sz])
        M[:3, 3] = [tx, ty, tz]
        return M

    @staticmethod
    def from_matrix(M: np.ndarray, scale: list[float] = None) -> 'Transform3d':
        """Build a Transform3d from a 4×4 pose matrix.

        If scale is None the rotation block is assumed to be a pure rotation
        (no scale baked in) and scale defaults to [1, 1, 1].
        """
        if scale is None:
            scale = [1.0, 1.0, 1.0]
        q_xyzw = Rotation.from_matrix(M[:3, :3]).as_quat()
        q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
        return Transform3d(rotation=q_wxyz, translation=M[:3, 3].tolist(), scale=scale)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(path: str) -> 'Transform3d':
        with open(path) as f:
            return Transform3d.from_dict(json.load(f))


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

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(path: str) -> 'BoundingBox':
        with open(path) as f:
            return BoundingBox.from_dict(json.load(f))


@dataclass
class BoundingBox3d:
    """Axis-aligned bounding box in 3D space."""
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "z0": self.z0,
                "x1": self.x1, "y1": self.y1, "z1": self.z1}

    @staticmethod
    def from_dict(d: dict) -> 'BoundingBox3d':
        return BoundingBox3d(x0=d["x0"], y0=d["y0"], z0=d["z0"],
                             x1=d["x1"], y1=d["y1"], z1=d["z1"])

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(path: str) -> 'BoundingBox3d':
        with open(path) as f:
            return BoundingBox3d.from_dict(json.load(f))


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

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(path: str) -> 'Point':
        with open(path) as f:
            return Point.from_dict(json.load(f))


@dataclass
class Mask:
    mask: np.ndarray

    def to_pil_image(self) -> PILImage.Image:
        return PILImage.fromarray((self.mask * 255).astype(np.uint8), mode="L")

    @staticmethod
    def from_pil_image(pil_image: PILImage.Image) -> 'Mask':
        return Mask(mask=np.array(pil_image) / 255)

    def width(self) -> int:
        return self.mask.shape[1]

    def height(self) -> int:
        return self.mask.shape[0]

    def save(self, path: str) -> None:
        self.to_pil_image().save(path)

    @staticmethod
    def load(path: str) -> 'Mask':
        return Mask.from_pil_image(PILImage.open(path))


@dataclass
class Image:
    """RGB image as a uint8 array of shape (H, W, 3)."""
    data: np.ndarray

    def to_pil_image(self) -> PILImage.Image:
        return PILImage.fromarray(self.data, mode='RGB')

    @staticmethod
    def from_pil_image(img: PILImage.Image) -> 'Image':
        return Image(data=np.array(img.convert('RGB'), dtype=np.uint8))

    def save(self, path: str) -> None:
        self.to_pil_image().save(path)

    @staticmethod
    def load(path: str) -> 'Image':
        return Image.from_pil_image(PILImage.open(path))

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]
