"""NGC FoundationPose predictors backed by TensorRT engines."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_FP_DIR = Path(__file__).resolve().parent / "FoundationPose"
if str(_FP_DIR) not in sys.path:
    sys.path.insert(0, str(_FP_DIR))

from learning.datasets.h5_dataset import (  # noqa: E402
    PoseRefinePairH5Dataset,
    ScoreMultiPairH5Dataset,
)
from learning.training.predict_pose_refine import (  # noqa: E402
    make_crop_data_batch as make_refine_crop_data,
)
from learning.training.predict_score import (  # noqa: E402
    make_crop_data_batch as make_score_crop_data,
    vis_batch_data_scores,
)
from Utils import egocentric_delta_pose_to_pose, make_mesh_tensors  # noqa: E402
from pytorch3d.transforms import so3_exp_map  # noqa: E402

from v2d.foundation_pose.lib.trt_inference import TensorRTModule


INPUT_SIZE = (160, 160)
REFINER_MAX_BATCH = 42
SCORER_MAX_BATCH = 252


def _base_config():
    return OmegaConf.create(
        {
            "input_resize": list(INPUT_SIZE),
            "use_normal": False,
            "use_mask": False,
            "use_BN": False,
            "c_in": 4,
            "crop_ratio": 1.2,
            "n_view": 1,
            "trans_rep": "tracknet",
            "rot_rep": "axis_angle",
            "zfar": 3.0,
            # The NGC ONNX checkpoints use XYZ centered on the current pose and
            # normalized by half the mesh diameter, matching Isaac ROS.
            "normalize_xyz": True,
            "normal_uint8": False,
            "trans_normalizer": [
                0.019999999552965164,
                0.019999999552965164,
                0.05000000074505806,
            ],
            "rot_normalizer": 0.349065850398865,
        }
    )


def _model_inputs(pose_data) -> tuple[torch.Tensor, torch.Tensor]:
    input1 = torch.cat([pose_data.rgbAs, pose_data.xyz_mapAs], dim=1).float()
    input2 = torch.cat([pose_data.rgbBs, pose_data.xyz_mapBs], dim=1).float()
    # The TAO ONNX models retain the PyTorch NCHW channel layout.
    return input1.contiguous(), input2.contiguous()


def _run_refiner_chunks(
    runner: TensorRTModule,
    input1: torch.Tensor,
    input2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the bounded refiner while retaining the input hypothesis order."""
    chunks: dict[str, list[torch.Tensor]] = {"trans": [], "rot": []}
    for start in range(0, len(input1), REFINER_MAX_BATCH):
        stop = min(start + REFINER_MAX_BATCH, len(input1))
        outputs = runner(input1[start:stop], input2[start:stop])
        if set(outputs) != set(chunks):
            raise RuntimeError(
                f"Unexpected FoundationPose refiner outputs: {tuple(outputs)}"
            )
        for name in chunks:
            chunks[name].append(outputs[name])
    return {name: torch.cat(values, dim=0) for name, values in chunks.items()}


class TensorRTPoseRefinePredictor:
    """Drop-in replacement for NVLabs ``PoseRefinePredictor``."""

    def __init__(self, engine_path: str | os.PathLike[str]):
        self.cfg = _base_config()
        self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file="", mode="test")
        self.runner = TensorRTModule(engine_path)
        self.last_trans_update = None
        self.last_rot_update = None

    @torch.inference_mode()
    def predict(
        self,
        rgb,
        depth,
        K,
        ob_in_cams,
        xyz_map,
        normal_map=None,
        get_vis=False,
        mesh=None,
        mesh_tensors=None,
        glctx=None,
        mesh_diameter=None,
        iteration=5,
    ):
        del normal_map  # The commercial checkpoint does not consume normals.
        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh)
        poses = torch.as_tensor(ob_in_cams, device="cuda", dtype=torch.float32)
        depth_tensor = torch.as_tensor(depth, device="cuda", dtype=torch.float32)
        xyz_tensor = torch.as_tensor(xyz_map, device="cuda", dtype=torch.float32)

        for _ in range(iteration):
            pose_data = make_refine_crop_data(
                self.cfg.input_resize,
                poses,
                mesh,
                torch.as_tensor(rgb, device="cuda", dtype=torch.float32),
                depth_tensor,
                K,
                crop_ratio=self.cfg.crop_ratio,
                xyz_map=xyz_tensor,
                normal_map=None,
                cfg=self.cfg,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                dataset=self.dataset,
                mesh_diameter=mesh_diameter,
            )
            input1, input2 = _model_inputs(pose_data)
            outputs = _run_refiner_chunks(self.runner, input1, input2)
            # The commercial checkpoint predicts translation in units of
            # half the mesh diameter. This is the normalize_xyz branch of
            # the existing NVLabs pose decoder.
            trans_delta = outputs["trans"].float() * (mesh_diameter / 2.0)
            rot_delta = torch.tanh(outputs["rot"].float()) * self.cfg.rot_normalizer
            rot_matrix_delta = so3_exp_map(rot_delta).permute(0, 2, 1)
            poses = egocentric_delta_pose_to_pose(
                pose_data.poseA,
                trans_delta=trans_delta,
                rot_mat_delta=rot_matrix_delta,
            ).reshape(len(ob_in_cams), 4, 4)
            self.last_trans_update = trans_delta
            self.last_rot_update = rot_matrix_delta

        # Existing callers only persist refiner visualization when non-None.
        return poses, None


class TensorRTScorePredictor:
    """Drop-in replacement for NVLabs ``ScorePredictor``."""

    def __init__(self, engine_path: str | os.PathLike[str]):
        self.cfg = _base_config()
        self.dataset = ScoreMultiPairH5Dataset(
            cfg=self.cfg, mode="test", h5_file=None, max_num_key=1
        )
        self.runner = TensorRTModule(engine_path)

    @torch.inference_mode()
    def predict(
        self,
        rgb,
        depth,
        K,
        ob_in_cams,
        normal_map=None,
        get_vis=False,
        mesh=None,
        mesh_tensors=None,
        glctx=None,
        mesh_diameter=None,
    ):
        del normal_map
        if len(ob_in_cams) > SCORER_MAX_BATCH:
            raise ValueError(
                f"FoundationPose scorer supports at most {SCORER_MAX_BATCH} "
                f"hypotheses, got {len(ob_in_cams)}"
            )
        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh)
        pose_data = make_score_crop_data(
            self.cfg.input_resize,
            torch.as_tensor(ob_in_cams, device="cuda", dtype=torch.float32),
            mesh,
            torch.as_tensor(rgb, device="cuda", dtype=torch.float32),
            torch.as_tensor(depth, device="cuda", dtype=torch.float32),
            K,
            crop_ratio=self.cfg.crop_ratio,
            normal_map=None,
            mesh_diameter=mesh_diameter,
            glctx=glctx,
            mesh_tensors=mesh_tensors,
            dataset=self.dataset,
            cfg=self.cfg,
        )
        input1, input2 = _model_inputs(pose_data)
        scores = self.runner(input1, input2)["score_logit"].float().reshape(-1)
        # Preserve the offset used by NVLabs ScorePredictor. It does not affect
        # ranking or particle softmax weights, but keeps diagnostics comparable.
        scores = scores + 100.0
        if get_vis:
            ids = scores.argsort(descending=True)
            return scores, vis_batch_data_scores(pose_data, ids=ids, scores=scores)
        return scores, None
