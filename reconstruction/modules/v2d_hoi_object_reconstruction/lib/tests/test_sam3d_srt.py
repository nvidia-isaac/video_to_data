# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from v2d_hoi_object_reconstruction.lib import sam3d_srt as _sam3d_srt


def _write_mesh(job_dir, frame_id):
    frame_dir = job_dir / "sam3d" / frame_id
    frame_dir.mkdir(parents=True)
    (frame_dir / "mesh.glb").write_bytes(b"mesh")
    return frame_dir


def _write_completed(job_dir, frame_id, scale=1.0):
    frame_dir = _write_mesh(job_dir, frame_id)
    srt_dir = frame_dir / "srt"
    srt_dir.mkdir()
    (srt_dir / "srt_result.json").write_text(json.dumps({"scale": scale}))
    (srt_dir / "output_scaled.glb").write_bytes(b"scaled")


def test_osmo_fast_path_defaults():
    assert _sam3d_srt.SRTConfig() == _sam3d_srt.SRTConfig(
        max_views=25,
        maxiter=60,
        top_k=1,
        parallel=8,
    )


def test_completed_result_requires_valid_result_and_nonempty_scaled_mesh(tmp_path):
    _write_completed(tmp_path, "000001", scale=[1.0, 2.0, 3.0])
    result_path = tmp_path / "sam3d" / "000001" / "srt" / "srt_result.json"
    mesh_path = tmp_path / "sam3d" / "000001" / "srt" / "output_scaled.glb"
    assert _sam3d_srt._completed_result(tmp_path, "000001")["scale"] == [
        1.0,
        2.0,
        3.0,
    ]

    result_path.write_text("not json")
    assert _sam3d_srt._completed_result(tmp_path, "000001") is None

    result_path.write_text(json.dumps({"scale": 1.0}))
    mesh_path.write_bytes(b"")
    assert _sam3d_srt._completed_result(tmp_path, "000001") is None


def test_scheduler_reuses_complete_candidates_and_restores_environment(
    tmp_path, monkeypatch
):
    _write_completed(tmp_path, "000001", scale=1.25)
    _write_mesh(tmp_path, "000002")
    config = _sam3d_srt.SRTConfig(parallel=8)
    calls = []

    def fake_run(job_dir, frame_id, use_depth, stage1_end_frame, worker_config):
        calls.append((frame_id, use_depth, stage1_end_frame, worker_config))
        return _sam3d_srt.SRTOutcome(frame_id, {"scale": 2.5}, 3.0)

    class FakeFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class FakePool:
        def __init__(self, max_workers, mp_context):
            assert max_workers == 1
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            return FakeFuture(function(*args))

    monkeypatch.setattr(_sam3d_srt, "_run_candidate", fake_run)
    monkeypatch.setattr(_sam3d_srt, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(_sam3d_srt, "as_completed", list)
    monkeypatch.setenv("OMP_NUM_THREADS", "original")

    outcomes = _sam3d_srt.run_srt_candidates(
        tmp_path,
        ["000001", "000002"],
        use_depth=True,
        stage1_end_frame=925,
        config=config,
    )

    assert [outcome.frame_id for outcome in outcomes] == ["000001", "000002"]
    assert [outcome.resumed for outcome in outcomes] == [True, False]
    assert calls == [("000002", True, 925, config)]
    assert _sam3d_srt.os.environ["OMP_NUM_THREADS"] == "original"
