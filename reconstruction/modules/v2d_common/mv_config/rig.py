from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from .params import CameraParam, load_camera_params

_RIG_DIR = Path(__file__).parent / "rigs"


@dataclass
class CameraEntry:
    cam_id: int
    name: str
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

        params: list[CameraParam] | None = None
        if camera_params_path is not None:
            params = load_camera_params(Path(camera_params_path))

        self.cameras: dict[int, CameraEntry] = {}
        for cam in cfg.cameras:
            param = params[cam.cam_id] if params is not None else None
            entry = CameraEntry(cam_id=cam.cam_id, name=cam.name, param=param)
            self.cameras[cam.cam_id] = entry

        self.stereo_pairs: list[StereoPair] = []
        for pair in cfg.get("stereo_pairs", []):
            self.stereo_pairs.append(StereoPair(
                name=pair.name,
                left=self.cameras[pair.left],
                right=self.cameras[pair.right],
            ))

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
