from enum import Enum
from functools import partial
import json
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


def to_np_array(value: list[float | int], dtype: np.dtype = np.float32) -> np.ndarray:
    return np.array(value, dtype=dtype)


ArrayFloat = Annotated[
    np.ndarray, BeforeValidator(partial(to_np_array, dtype=np.float32))
]
ArrayInt = Annotated[np.ndarray, BeforeValidator(partial(to_np_array, dtype=np.int32))]


def all_close_or_none(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    return (a is None and b is None) or (
        a is not None and b is not None and np.allclose(a, b)
    )


class EDEXEncoder(json.JSONEncoder):
    """Custom JSON encoder for EDEX files that handles numpy arrays and Path objects."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class DistortionModel(str, Enum):
    """
    Supported camera distortion models in EDEX format.

    - PINHOLE: No distortion (0 parameters)
    - FISHEYE: Fisheye distortion model (4 parameters)
    - BROWN5K: Brown-Conrady distortion with 5 parameters
    - POLYNOMIAL: Rational polynomial distortion (8 parameters)
    """

    PINHOLE = "pinhole"
    FISHEYE = "fisheye"
    BROWN5K = "brown5k"
    POLYNOMIAL = "polynomial"


class Intrinsics(BaseModel):
    """
    Camera intrinsic parameters including pinhole model and distortion.

    Attributes:
        distortion_model: Type of distortion model used
        distortion_params: Distortion coefficients (length depends on model)
        focal: Focal lengths [fx, fy] in pixels
        principal: Principal point [cx, cy] in pixels
        resolution: Image resolution [width, height] in pixels (aliased as 'size')
        projection: Optional 3x4 projection matrix [K'|t] for a stereo camera pair
        rectification: Optional 3x3 rectification matrix [R] for a stereo camera pair

    See also: https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    distortion_model: DistortionModel
    distortion_params: ArrayFloat
    focal: ArrayFloat
    principal: ArrayFloat
    resolution: ArrayInt = Field(alias="size")
    projection: ArrayFloat | None = None
    rectification: ArrayFloat | None = None

    @model_validator(mode="after")
    def check_fields(self):
        # Check distortion model
        match self.distortion_model:
            case DistortionModel.PINHOLE:
                if self.distortion_params.shape != (0,):
                    raise ValueError(
                        f"Invalid distortion params: {self.distortion_params}"
                    )
            case DistortionModel.FISHEYE:
                if self.distortion_params.shape != (4,):
                    raise ValueError(
                        f"Invalid distortion params: {self.distortion_params}"
                    )
            case DistortionModel.BROWN5K:
                if self.distortion_params.shape != (5,):
                    raise ValueError(
                        f"Invalid distortion params: {self.distortion_params}"
                    )
            case DistortionModel.POLYNOMIAL:
                if self.distortion_params.shape != (8,):
                    raise ValueError(
                        f"Invalid distortion params: {self.distortion_params}"
                    )

        # Check pinhole intrinsics
        if self.focal.shape != (2,):
            raise ValueError(f"Invalid focal: {self.focal}")
        if self.principal.shape != (2,):
            raise ValueError(f"Invalid principal: {self.principal}")
        if (
            self.resolution.shape != (2,)
            or self.resolution[0] <= 0
            or self.resolution[1] <= 0
        ):
            raise ValueError(f"Invalid resolution: {self.resolution}")

        # Check stereo rectification matrices
        if self.projection is not None and self.projection.shape != (3, 4):
            raise ValueError(f"Invalid projection: {self.projection}")
        if self.rectification is not None and self.rectification.shape != (3, 3):
            raise ValueError(f"Invalid rectification: {self.rectification}")

        return self

    def __eq__(self, other: "Intrinsics") -> bool:
        return (
            isinstance(other, Intrinsics)
            and self.distortion_model == other.distortion_model
            and np.allclose(self.distortion_params, other.distortion_params)
            and np.allclose(self.focal, other.focal)
            and np.allclose(self.principal, other.principal)
            and np.array_equal(self.resolution, other.resolution)
            and all_close_or_none(self.projection, other.projection)
            and all_close_or_none(self.rectification, other.rectification)
        )


class Camera(BaseModel):
    """
    Camera specification including intrinsics and extrinsic transform.

    Attributes:
        intrinsics: Camera intrinsic parameters
        transform: Optional 3x4 extrinsic transformation matrix [R|t] from camera to rig/world frame
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    intrinsics: Intrinsics
    transform: ArrayFloat | None = None

    @model_validator(mode="after")
    def check_fields(self):
        if self.transform is not None and self.transform.shape != (3, 4):
            raise ValueError(f"Invalid transform: {self.transform}")
        return self

    def __eq__(self, other: "Camera") -> bool:
        return (
            isinstance(other, Camera)
            and self.intrinsics == other.intrinsics
            and all_close_or_none(self.transform, other.transform)
        )


class IMU(BaseModel):
    """
    IMU (Inertial Measurement Unit) specification and data location.

    Attributes:
        g: Gravity vector [gx, gy, gz] in m/s^2
        measurements: Path to IMU measurements file (JSONL format)
        transform: 3x4 extrinsic transform matrix [R|t] from rig to IMU frame
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    g: ArrayFloat
    measurements: Path
    transform: ArrayFloat

    @model_validator(mode="after")
    def check_fields(self):
        if self.g.shape != (3,):
            raise ValueError(f"Invalid g: {self.g}")
        if self.transform.shape != (3, 4):
            raise ValueError(f"Invalid transform: {self.transform}")
        return self

    def __eq__(self, other: "IMU") -> bool:
        return (
            isinstance(other, IMU)
            and np.allclose(self.g, other.g)
            and self.measurements == other.measurements
            and np.allclose(self.transform, other.transform)
        )


class EDEXHeader(BaseModel):
    """
    Header section of EDEX file containing dataset metadata.

    Attributes:
        version: EDEX format version
        frame_start: Starting frame index (inclusive)
        frame_end: Ending frame index (exclusive)
        cameras: List of camera specifications in the rig
        imu: Optional IMU specification
    """

    version: str = "0.9"
    frame_start: int
    frame_end: int
    cameras: list[Camera]
    imu: IMU | None = None


class EDEXBody(BaseModel):
    """
    Body section of EDEX file containing data file references.

    Attributes:
        frame_metadata: Optional path to per-frame metadata file (JSONL format)
        sequence: List of paths to first frame images for each camera
    """

    frame_metadata: Path | None = None
    sequence: list[Path]


class EDEXMetadata:
    """
    EDEX metadata file reader/writer.

    EDEX is a format for storing multi-camera dataset metadata including
    camera intrinsics, extrinsics, IMU data, and frame sequences.

    Attributes:
        header: Dataset metadata (cameras, IMU, frame range)
        body: Data file references (images, frame metadata)
    """

    def __init__(self, header: EDEXHeader, body: EDEXBody | None = None):
        self.header = header
        self.body = body

    @classmethod
    def read(cls, filename: Path) -> "EDEXMetadata":
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            header = EDEXHeader.model_validate(data[0])
            body = EDEXBody.model_validate(data[1])
            return cls(header, body)
        except Exception as e:
            print(f"Error reading EDEX file {filename}")
            raise e

    def write(self, filename: Path):
        try:
            # Validate the header and body before writing
            new_header = EDEXHeader.model_validate(self.header.model_dump())
            new_body = EDEXBody.model_validate(self.body.model_dump())
            data = [
                new_header.model_dump(exclude_none=True),
                new_body.model_dump(exclude_none=True),
            ]
            with open(filename, "w") as f:
                json.dump(data, f, indent=2, cls=EDEXEncoder)
        except Exception as e:
            print(f"Error writing EDEX file {filename}")
            raise e

    def __str__(self) -> str:
        return f"EDEXMetadata(header={self.header.model_dump()}, body={self.body.model_dump()})"

    def __eq__(self, other: "EDEXMetadata") -> bool:
        return (
            isinstance(other, EDEXMetadata)
            and self.header == other.header
            and self.body == other.body
        )
