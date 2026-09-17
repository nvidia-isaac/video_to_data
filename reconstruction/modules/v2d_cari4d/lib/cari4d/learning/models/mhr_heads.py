from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MHRHeadSpec:
    output_key: str
    dim: int
    optional: bool = False


BASE_MHR_HEADS = (
    MHRHeadSpec("delta_mhr_global_rot6d", 6),
    MHRHeadSpec("delta_mhr_trans", 3),
    MHRHeadSpec("delta_mhr_body_pose_cont", 260),
    MHRHeadSpec("delta_mhr_hand", 108),
)

OPTIONAL_MHR_HEADS = {
    "shape": MHRHeadSpec("delta_mhr_shape", 45, optional=True),
    "scale": MHRHeadSpec("delta_mhr_scale", 28, optional=True),
    "face": MHRHeadSpec("delta_mhr_face", 72, optional=True),
}


def enabled_mhr_head_specs(*, pred_shape: bool = False, pred_scale: bool = False,
                           pred_face: bool = False) -> tuple[MHRHeadSpec, ...]:
    specs = list(BASE_MHR_HEADS)
    if pred_shape:
        specs.append(OPTIONAL_MHR_HEADS["shape"])
    if pred_scale:
        specs.append(OPTIONAL_MHR_HEADS["scale"])
    if pred_face:
        specs.append(OPTIONAL_MHR_HEADS["face"])
    return tuple(specs)


def mhr_output_dim(*, pred_shape: bool = False, pred_scale: bool = False,
                   pred_face: bool = False) -> int:
    return sum(spec.dim for spec in enabled_mhr_head_specs(
        pred_shape=pred_shape,
        pred_scale=pred_scale,
        pred_face=pred_face,
    ))

