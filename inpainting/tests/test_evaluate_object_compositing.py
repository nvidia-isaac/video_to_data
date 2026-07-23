from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from inpainting.contracts import ContractError
from inpainting import evaluate_object_compositing
from inpainting.evaluate_object_compositing import (
    DEPTH_GUARD_M,
    evaluate_arrays,
    evaluate_files,
)


def _depth(mask: np.ndarray, values: list[list[list[float]]]) -> np.ndarray:
    depth = np.asarray(values, dtype=np.float32)
    depth[~mask] = np.inf
    return depth


def _fixture() -> dict[str, np.ndarray]:
    robot_mask = np.asarray(
        [
            [[True, True, True, False]],
            [[True, True, True, True]],
            [[False, True, True, True]],
        ],
        dtype=np.bool_,
    )
    robot_depth = _depth(
        robot_mask,
        [
            [[1.0, 1.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0, 1.0]],
            [[0.0, 1.0, 1.0, 1.0]],
        ],
    )

    gt_occluder_mask = np.asarray(
        [
            [[True, True, False, False]],
            [[True, False, True, False]],
            [[False, False, False, False]],
        ],
        dtype=np.bool_,
    )
    gt_occluder_depth = _depth(
        gt_occluder_mask,
        [
            [[0.5, 0.997, 0.0, 0.0]],
            [[0.5, 0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.0, 0.0]],
        ],
    )

    candidate_occluder_mask = np.asarray(
        [
            [[False, True, False, False]],
            [[False, False, True, True]],
            [[False, False, True, False]],
        ],
        dtype=np.bool_,
    )
    candidate_occluder_depth = _depth(
        candidate_occluder_mask,
        [
            [[0.0, 0.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.5, 0.5]],
            [[0.0, 0.0, 0.5, 0.0]],
        ],
    )
    return {
        "robot_mask": robot_mask,
        "robot_depth": robot_depth,
        "gt_occluder_mask": gt_occluder_mask,
        "gt_occluder_depth": gt_occluder_depth,
        "candidate_occluder_mask": candidate_occluder_mask,
        "candidate_occluder_depth": candidate_occluder_depth,
    }


def _evaluate() -> dict:
    return evaluate_arrays(
        **_fixture(),
        sequence_id="synthetic-105",
        candidate_name="estimated-depth",
        worst_frame_count=2,
    )


def _write_fixture(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, array in _fixture().items():
        path = root / f"{name}.npy"
        np.save(path, array)
        paths[f"{name}_path"] = path
    return paths


def test_micro_metrics_guard_rates_and_object_quality() -> None:
    result = _evaluate()
    assert result["definitions"]["depth_guard_m"] == DEPTH_GUARD_M == 0.003

    visibility = result["metrics"]["visibility"]["micro"]
    assert visibility["robot_pixel_count"] == 10
    assert visibility["gt_visible_pixel_count"] == 7
    assert visibility["gt_occluded_pixel_count"] == 3
    assert visibility["candidate_visible_pixel_count"] == 6
    assert visibility["both_visible_pixel_count"] == 4
    assert visibility["both_occluded_pixel_count"] == 1
    assert visibility["visible_union_pixel_count"] == 9
    assert visibility["visible_iou"] == pytest.approx(4 / 9)

    false_visible = visibility["false_visible"]
    assert false_visible["count"] == 2
    assert false_visible["decision_denominator"] == {
        "name": "gt_occluded_robot_pixels",
        "pixel_count": 3,
    }
    assert false_visible["rate_decision_denominator"] == pytest.approx(2 / 3)
    assert false_visible["rate_total_robot_pixels"] == pytest.approx(0.2)

    false_occluded = visibility["false_occluded"]
    assert false_occluded["count"] == 3
    assert false_occluded["decision_denominator"] == {
        "name": "gt_visible_robot_pixels",
        "pixel_count": 7,
    }
    assert false_occluded["rate_decision_denominator"] == pytest.approx(3 / 7)
    assert false_occluded["rate_total_robot_pixels"] == pytest.approx(0.3)
    assert visibility["decision_disagreement"] == {
        "count": 5,
        "rate_total_robot_pixels": 0.5,
    }

    object_mask = result["metrics"]["occluder_mask"]["micro"]
    assert object_mask == {
        "gt_pixel_count": 4,
        "candidate_pixel_count": 4,
        "intersection_pixel_count": 2,
        "union_pixel_count": 6,
        "iou": pytest.approx(1 / 3),
    }
    overlap_depth = result["metrics"]["overlap_depth"]["micro"]
    assert overlap_depth["overlap_pixel_count"] == 2
    assert overlap_depth["absolute_error_sum_m"] == pytest.approx(0.497)
    assert overlap_depth["mae_m"] == pytest.approx(0.2485)

    # 0.997 m is retained against the 1.0 m robot by the fixed 3 mm guard.
    first = result["per_frame"][0]["visibility"]
    assert first["gt_visible_pixel_count"] == 2
    assert first["visible_iou"] == pytest.approx(1 / 3)
    assert result["worst_frames"]["highest_decision_disagreement_count"] == [
        {"frame_index": 0, "value": 2},
        {"frame_index": 1, "value": 2},
    ]
    assert result["self_checks"]["all_passed"] is True
    assert all(check["passed"] for check in result["self_checks"]["checks"].values())


def test_persistent_temporal_transitions_exclude_robot_silhouette_motion() -> None:
    result = _evaluate()
    temporal = result["metrics"]["persistent_pixel_temporal_transition"]
    assert temporal["adjacent_frame_pair_count"] == 2
    assert temporal["persistent_pixel_pair_count"] == 6
    assert temporal["gt_transition_count"] == 2
    assert temporal["candidate_transition_count"] == 3
    assert temporal["both_transition_count"] == 1
    assert temporal["spurious_candidate_transition_count"] == 2
    assert temporal["missed_candidate_transition_count"] == 1
    assert temporal["transition_disagreement_count"] == 3
    assert temporal["transition_disagreement_rate"] == pytest.approx(0.5)
    assert temporal["per_frame_pair"] == [
        {
            "from_frame": 0,
            "to_frame": 1,
            "persistent_pixel_count": 3,
            "gt_transition_count": 1,
            "candidate_transition_count": 2,
            "both_transition_count": 1,
            "neither_transition_count": 1,
            "spurious_candidate_transition_count": 1,
            "missed_candidate_transition_count": 0,
            "transition_disagreement_count": 1,
            "transition_disagreement_rate": pytest.approx(1 / 3),
        },
        {
            "from_frame": 1,
            "to_frame": 2,
            "persistent_pixel_count": 3,
            "gt_transition_count": 1,
            "candidate_transition_count": 1,
            "both_transition_count": 0,
            "neither_transition_count": 1,
            "spurious_candidate_transition_count": 1,
            "missed_candidate_transition_count": 1,
            "transition_disagreement_count": 2,
            "transition_disagreement_rate": pytest.approx(2 / 3),
        },
    ]


def test_empty_decision_denominators_are_explicit_json_nulls() -> None:
    mask = np.zeros((1, 1, 2), dtype=np.bool_)
    depth = np.full(mask.shape, np.inf, dtype=np.float32)
    result = evaluate_arrays(
        robot_mask=mask,
        robot_depth=depth,
        gt_occluder_mask=mask,
        gt_occluder_depth=depth,
        candidate_occluder_mask=mask,
        candidate_occluder_depth=depth,
        sequence_id="empty",
        candidate_name="empty",
    )
    visibility = result["metrics"]["visibility"]["micro"]
    assert visibility["visible_iou"] == 1.0
    assert visibility["false_visible"]["rate_decision_denominator"] is None
    assert visibility["false_visible"]["rate_total_robot_pixels"] is None
    assert visibility["false_occluded"]["rate_decision_denominator"] is None
    assert result["metrics"]["occluder_mask"]["micro"]["iou"] == 1.0
    temporal = result["metrics"]["persistent_pixel_temporal_transition"]
    assert temporal["transition_disagreement_rate"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda arrays: arrays.__setitem__(
                "candidate_occluder_mask",
                arrays["candidate_occluder_mask"].astype(np.uint8),
            ),
            "boolean dtype",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "candidate_occluder_depth",
                arrays["candidate_occluder_depth"][:, :, :3],
            ),
            "must have shape",
        ),
        (
            lambda arrays: arrays["candidate_occluder_depth"].__setitem__(
                (0, 0, 0), 0.25
            ),
            "outside its mask",
        ),
    ],
)
def test_invalid_bundle_arrays_fail_closed(mutation, message: str) -> None:
    arrays = _fixture()
    mutation(arrays)
    with pytest.raises(ContractError, match=message):
        evaluate_arrays(
            **arrays,
            sequence_id="invalid",
            candidate_name="invalid",
        )


def test_file_evaluator_memmaps_fingerprints_and_writes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "evaluation.json"
    opened_as_memmaps: list[bool] = []
    original_open = evaluate_object_compositing._open_npy_memmap

    def recording_open(path: Path, *, label: str) -> np.ndarray:
        array = original_open(path, label=label)
        opened_as_memmaps.append(isinstance(array, np.memmap))
        return array

    monkeypatch.setattr(evaluate_object_compositing, "_open_npy_memmap", recording_open)
    evaluate_files(
        **paths,
        output_path=output,
        sequence_id="synthetic-105",
        candidate_name="estimated-depth",
        worst_frame_count=2,
    )
    first = output.read_bytes()
    assert opened_as_memmaps == [True] * 6
    parsed = json.loads(first)
    assert parsed["inputs"]["robot_mask"]["sha256"]
    assert parsed["inputs"]["candidate_occluder_depth"]["size_bytes"] > 0
    assert not list(tmp_path.glob(".*.partial"))

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        evaluate_files(
            **paths,
            output_path=output,
            sequence_id="synthetic-105",
            candidate_name="estimated-depth",
        )
    evaluate_files(
        **paths,
        output_path=output,
        sequence_id="synthetic-105",
        candidate_name="estimated-depth",
        worst_frame_count=2,
        overwrite=True,
    )
    assert output.read_bytes() == first


def test_command_line_entrypoint_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "cli.json"
    arguments = [
        "evaluate_object_compositing",
        "--robot-mask",
        str(paths["robot_mask_path"]),
        "--robot-depth",
        str(paths["robot_depth_path"]),
        "--gt-occluder-mask",
        str(paths["gt_occluder_mask_path"]),
        "--gt-occluder-depth",
        str(paths["gt_occluder_depth_path"]),
        "--candidate-occluder-mask",
        str(paths["candidate_occluder_mask_path"]),
        "--candidate-occluder-depth",
        str(paths["candidate_occluder_depth_path"]),
        "--sequence-id",
        "synthetic-105",
        "--candidate-name",
        "video2data-object",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    evaluate_object_compositing.main()
    assert output.is_file()
    assert json.loads(output.read_text())["candidate_name"] == "video2data-object"
    assert "visible IoU=" in capsys.readouterr().out
