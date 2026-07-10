import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))


class FakeFrameSource:
    def __init__(self, path):
        self.path = Path(path)
        self.image_paths = sorted(self.path.glob("*.png"))
        self.stems = [p.stem for p in self.image_paths]
        self.n_frames = len(self.image_paths)

    @classmethod
    def from_path(cls, path):
        return cls(path)

    def __getitem__(self, idx):
        return np.array(Image.open(self.image_paths[idx]).convert("L"))

    def close(self):
        pass


v2d = types.ModuleType("v2d")
v2d_common = types.ModuleType("v2d.common")
v2d_common_video = types.ModuleType("v2d.common.video")
v2d_common_video.FrameSource = FakeFrameSource
_STUB_MODULES = {
    "v2d": v2d,
    "v2d.common": v2d_common,
    "v2d.common.video": v2d_common_video,
}
_PREVIOUS_MODULES = {name: sys.modules.get(name) for name in _STUB_MODULES}
sys.modules.update(_STUB_MODULES)

import check_object_mask as cm

for name, module in _PREVIOUS_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


def _write_mask(path: Path, bbox: tuple[int, int, int, int]) -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask, mode="L").save(path)


def test_check_object_mask_uses_labeled_frame_key(tmp_path):
    bbox_dir = tmp_path / "bboxes"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "out"
    bbox_dir.mkdir()
    cam_mask_dir = mask_dir / "front_stereo_camera_left" / "0"
    cam_mask_dir.mkdir(parents=True)

    (bbox_dir / "front_stereo_camera_left.json").write_text(
        json.dumps(
            {
                "000112": [
                    {
                        "label": "blue_trash_can",
                        "confidence": 1.0,
                        "box": {"x0": 2.0, "y0": 2.0, "x1": 6.0, "y1": 6.0},
                    }
                ]
            }
        )
    )

    _write_mask(cam_mask_dir / "000000.png", (0, 0, 2, 2))
    _write_mask(cam_mask_dir / "000112.png", (2, 2, 6, 6))

    decision = cm.check_object_mask(
        mask_dir=str(mask_dir),
        labeled_bbox_dir=str(bbox_dir),
        output_dir=str(output_dir),
        min_containment=1.0,
        bbox_padding=0.0,
    )

    assert decision["status"] == "PASS"
    assert decision["per_camera_containment"] == {"front_stereo_camera_left": 1.0}
    assert decision["skipped_cameras"] == {}
