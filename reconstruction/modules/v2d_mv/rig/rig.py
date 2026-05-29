# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from .params import (
    CameraParam,
    EDEXMetadata,
    edex_camera_to_param,
    param_overwrite_in_edex,
)

_RIG_DIR = Path(__file__).parent / "rigs"


@dataclass
class CameraEntry:
    cam_id: int
    name: str
    image_path: str | None = None
    param: CameraParam | None = None


@dataclass
class StereoPair:
    name: str
    left: CameraEntry
    right: CameraEntry


class RigConfig:
    """Multi-camera rig configuration.

    Loads camera structure from a registered rig YAML (in modules/common/rig/)
    and optionally attaches CameraParam objects from a calibration file.
    """

    def __init__(
        self,
        rig_name: str,
        camera_params_path: str | Path | None = None,
    ):
        rig_yaml_path = _RIG_DIR / f"{rig_name}.yaml"
        if not rig_yaml_path.exists():
            raise FileNotFoundError(
                f"Rig config '{rig_name}' not found at {rig_yaml_path}"
            )

        cfg = OmegaConf.load(rig_yaml_path)

        self.cameras: dict[int, CameraEntry] = {}
        for cam in cfg.cameras:
            entry = CameraEntry(
                cam_id=cam.cam_id,
                name=cam.name,
                image_path=cam.get("image_path"),
            )
            self.cameras[cam.cam_id] = entry

        self.stereo_pairs: list[StereoPair] = []
        for pair in cfg.get("stereo_pairs", []):
            self.stereo_pairs.append(StereoPair(
                name=pair.name,
                left=self.cameras[pair.left],
                right=self.cameras[pair.right],
            ))

        if camera_params_path is not None:
            self.load_camera_params(camera_params_path)

    def get_camera(self, cam_id: int) -> CameraEntry:
        return self.cameras[cam_id]

    def get_camera_by_name(self, name: str) -> CameraEntry:
        for entry in self.cameras.values():
            if entry.name == name:
                return entry
        raise KeyError(f"No camera named '{name}'")

    def get_all_cameras(self) -> list[CameraEntry]:
        return list(self.cameras.values())

    def get_stereo_pairs(self) -> list[StereoPair]:
        return list(self.stereo_pairs)

    def get_left_cameras(self) -> list[CameraEntry]:
        return [pair.left for pair in self.stereo_pairs]

    def get_right_cameras(self) -> list[CameraEntry]:
        return [pair.right for pair in self.stereo_pairs]

    # ------------------------------------------------------------------
    # Format-dispatched camera param I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(path: Path) -> str:
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name == "edex" or suffix == ".edex":
            return "edex"
        raise ValueError(f"Unsupported camera params format: {path}")

    def load_camera_params(self, camera_params_path: str | Path) -> None:
        """Load camera params from a calibration file into the rig.

        Dispatches to a format-specific handler that knows how each format
        identifies cameras (e.g. by index for EDEX, by name for others).
        """
        path = Path(camera_params_path)
        fmt = self._detect_format(path)

        if fmt == "edex":
            self._load_edex_params(path)

    def _load_edex_params(self, path: Path) -> None:
        """Load params from EDEX, mapping by cam_id (index-based)."""
        edex = EDEXMetadata.read(path)
        for cam_id, entry in self.cameras.items():
            if cam_id < len(edex.header.cameras):
                entry.param = edex_camera_to_param(edex.header.cameras[cam_id])

    def merge_extrinsics(self, extrinsics_path: str | Path) -> None:
        """Merge extrinsic transforms from another camera params file.

        Loads params from *extrinsics_path* using the same format dispatch
        and copies the ``.T`` field onto the corresponding ``CameraEntry.param``.
        """
        path = Path(extrinsics_path)
        fmt = self._detect_format(path)

        if fmt == "edex":
            edex = EDEXMetadata.read(path)
            for cam_id, entry in self.cameras.items():
                if entry.param is not None and cam_id < len(edex.header.cameras):
                    ext_param = edex_camera_to_param(edex.header.cameras[cam_id])
                    entry.param.T = ext_param.T

    def save_camera_params(
        self,
        source_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """Merge current CameraParams back into a calibration file and write.

        Loads *source_path* to preserve non-camera metadata, overwrites
        camera parameters from this rig's entries, and writes to *output_path*.
        """
        source_path = Path(source_path)
        output_path = Path(output_path)
        fmt = self._detect_format(source_path)

        if fmt == "edex":
            edex = EDEXMetadata.read(source_path)
            for cam_id, entry in self.cameras.items():
                if entry.param is not None:
                    param_overwrite_in_edex(edex, cam_id, entry.param)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            edex.write(output_path)
