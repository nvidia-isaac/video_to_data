import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


LIB_DIR = Path(__file__).resolve().parents[1]


@dataclass
class _BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class _Sam2Prompt:
    frame_index: int
    object_id: int
    box: _BoundingBox | None = None


class _Sam2Prompts:
    def __init__(self, prompts):
        self.prompts = prompts


class _FakeFrameSource:
    frames = [np.full((20, 30, 3), 80, dtype=np.uint8)]
    last_instance = None

    def __init__(self):
        self.n_frames = len(self.frames)
        self.closed = False
        type(self).last_instance = self

    @classmethod
    def from_path(cls, _path):
        return cls()

    def __getitem__(self, idx):
        return self.frames[idx]

    def close(self):
        self.closed = True


def _load_module():
    target_name = "v2d.sam2.lib.mv_videos_to_masks"
    modules = {
        "v2d": types.ModuleType("v2d"),
        "v2d.common": types.ModuleType("v2d.common"),
        "v2d.common.datatypes": types.ModuleType("v2d.common.datatypes"),
        "v2d.common.video": types.ModuleType("v2d.common.video"),
        "v2d.mv": types.ModuleType("v2d.mv"),
        "v2d.mv.rig": types.ModuleType("v2d.mv.rig"),
        "v2d.sam2": types.ModuleType("v2d.sam2"),
        "v2d.sam2.lib": types.ModuleType("v2d.sam2.lib"),
        "v2d.sam2.lib.datatypes": types.ModuleType("v2d.sam2.lib.datatypes"),
        "v2d.sam2.lib.video_to_masks": types.ModuleType(
            "v2d.sam2.lib.video_to_masks"
        ),
    }
    modules["v2d.common.datatypes"].BoundingBox = _BoundingBox
    modules["v2d.common.video"].FrameSource = _FakeFrameSource
    modules["v2d.mv.rig"].RigConfig = object
    modules["v2d.sam2.lib.datatypes"].Sam2Prompt = _Sam2Prompt
    modules["v2d.sam2.lib.datatypes"].Sam2Prompts = _Sam2Prompts
    modules["v2d.sam2.lib.video_to_masks"].video_to_masks = lambda *_args, **_kwargs: None

    previous = {name: sys.modules.get(name) for name in modules}
    previous_target = sys.modules.get(target_name)
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            target_name,
            LIB_DIR / "mv_videos_to_masks.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[target_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
        if previous_target is None:
            sys.modules.pop(target_name, None)
        else:
            sys.modules[target_name] = previous_target


def test_save_bbox_prompt_visualization_draws_box_and_closes_source(tmp_path: Path):
    module = _load_module()
    output = tmp_path / "prompts" / "back_stereo_camera_left_bbox.png"
    prompt = _Sam2Prompt(
        frame_index=0,
        object_id=0,
        box=_BoundingBox(x0=5, y0=10, x1=24, y1=18),
    )

    module.save_bbox_prompt_visualization(
        "camera.h5",
        prompt,
        output,
        "blue_trash_can",
    )

    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert rendered[18, 24].tolist() == [0, 255, 0]
    label_region = rendered[:10]
    green_text = (
        (label_region[..., 1] > label_region[..., 0] + 20)
        & (label_region[..., 1] > label_region[..., 2] + 20)
        & (label_region[..., 1] > 100)
    )
    assert np.any(green_text)
    assert np.any(np.all(label_region < 20, axis=-1))
    assert rendered[15, 15].tolist() == [80, 80, 80]
    assert _FakeFrameSource.last_instance.closed is True


def test_load_bbox_prompt_label_uses_best_json_detection(tmp_path: Path):
    module = _load_module()
    bbox_path = tmp_path / "bboxes.json"
    bbox_path.write_text(
        json.dumps(
            {
                "000000": [
                    {"label": "mug", "confidence": 0.4, "box": {}},
                    {"label": "bottle", "confidence": 0.9, "box": {}},
                ]
            }
        )
    )

    assert module.load_bbox_prompt_label(bbox_path) == "bottle"
