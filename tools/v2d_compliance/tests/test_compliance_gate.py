# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the V2D Challenge compliance, quota, and anti-reward-hacking gates."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from tools.v2d_compliance import agent_submission_guard as guard
from tools.v2d_compliance import compliance_gate as cg


def _encode_mesh_b64(vertices: np.ndarray, faces: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez(
        buf,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
    )
    return base64.b64encode(zlib.compress(buf.getvalue(), level=9)).decode("ascii")


def _make_octahedron_mesh(scale: float = 0.08, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Create a valid non-planar triangle mesh with >= 32 active vertices."""
    rng = np.random.default_rng(42)
    angles_u = np.linspace(0.1, np.pi - 0.1, 8)
    angles_v = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    pts = []
    for u in angles_u:
        for v in angles_v:
            r = scale * (1.0 + 0.25 * np.sin(3.0 * u) * np.cos(2.0 * v))
            pts.append([r * np.sin(u) * np.cos(v) + offset, r * np.sin(u) * np.sin(v), r * np.cos(u)])
    vertices = np.asarray(pts, dtype=np.float64) + rng.normal(0.0, 1e-3, size=(len(pts), 3))
    faces = []
    for i in range(len(vertices) - 2):
        faces.append([0, i + 1, i + 2])
    return vertices, np.asarray(faces, dtype=np.int64)


class TestWeeklyQuotaAndScopeGates(unittest.TestCase):
    def test_weekly_quota_blocks_at_five_despite_kaggle_daily_allowance(self) -> None:
        now = dt.datetime(2026, 10, 24, 18, 0, 0, tzinfo=dt.timezone.utc)
        slugs = list(cg.CANONICAL_TRACK_COMPETITIONS["track_1"].values())
        # Simulate 5 submissions across the past 4 days (only 1 today, so Kaggle's 5/day limit says 4 left)
        kaggle_subs = {
            slug: [
                {"ref": 100 + i, "date": (now - dt.timedelta(days=i)).isoformat()}
                for i in range(5)
            ]
            for slug in slugs
        }
        status = cg.check_weekly_quota(
            "track_1",
            kaggle_submissions_by_slug=kaggle_subs,
            now_utc=now,
        )
        self.assertFalse(status.ok)
        self.assertEqual(status.weekly_used_max, 5)
        self.assertEqual(status.weekly_remaining, 0)
        self.assertTrue(any("WEEKLY_QUOTA_EXCEEDED" in r for r in status.blocking_reasons))
        self.assertIsNotNone(status.next_slot_opens_utc)

    def test_weekly_quota_allows_when_under_five_and_rolls_off_after_seven_days(self) -> None:
        now = dt.datetime(2026, 10, 24, 18, 0, 0, tzinfo=dt.timezone.utc)
        slugs = list(cg.CANONICAL_TRACK_COMPETITIONS["track_2_tier1"].values())
        # 4 submissions within 7 days + 2 submissions 8 days ago
        kaggle_subs = {
            slug: [
                {"ref": 1, "date": (now - dt.timedelta(days=1)).isoformat()},
                {"ref": 2, "date": (now - dt.timedelta(days=2)).isoformat()},
                {"ref": 3, "date": (now - dt.timedelta(days=4)).isoformat()},
                {"ref": 4, "date": (now - dt.timedelta(days=6)).isoformat()},
                {"ref": 5, "date": (now - dt.timedelta(days=8)).isoformat()},
                {"ref": 6, "date": (now - dt.timedelta(days=9)).isoformat()},
            ]
            for slug in slugs
        }
        status = cg.check_weekly_quota(
            "track_2_tier1",
            kaggle_submissions_by_slug=kaggle_subs,
            now_utc=now,
        )
        self.assertTrue(status.ok)
        self.assertEqual(status.weekly_used_max, 4)
        self.assertEqual(status.weekly_remaining, 1)

    def test_unlimited_final_three_days_and_post_freeze_block(self) -> None:
        slugs = list(cg.CANONICAL_TRACK_COMPETITIONS["track_3"].values())
        in_final_3d = dt.datetime(2026, 11, 3, 12, 0, 0, tzinfo=dt.timezone.utc)
        kaggle_subs = {
            slug: [
                {"ref": i, "date": (in_final_3d - dt.timedelta(hours=i * 3)).isoformat()}
                for i in range(9)
            ]
            for slug in slugs
        }
        status_final = cg.check_weekly_quota(
            "track_3",
            kaggle_submissions_by_slug=kaggle_subs,
            now_utc=in_final_3d,
        )
        self.assertTrue(status_final.ok)
        self.assertTrue(status_final.in_unlimited_window)

        after_freeze = dt.datetime(2026, 11, 4, 22, 0, 1, tzinfo=dt.timezone.utc)
        status_frozen = cg.check_weekly_quota(
            "track_3",
            kaggle_submissions_by_slug={},
            now_utc=after_freeze,
        )
        self.assertFalse(status_frozen.ok)
        self.assertTrue(any("COMPETITION_FROZEN" in r for r in status_frozen.blocking_reasons))

    def test_cross_board_count_divergence_is_blocked(self) -> None:
        now = dt.datetime(2026, 10, 24, 18, 0, 0, tzinfo=dt.timezone.utc)
        slugs = list(cg.CANONICAL_TRACK_COMPETITIONS["track_3"].values())
        kaggle_subs = {
            slug: [{"ref": 1, "date": (now - dt.timedelta(days=1)).isoformat()}]
            for slug in slugs
        }
        # Simulate an agent probing RPE with 2 extra single-board uploads
        kaggle_subs["v2d-challenge-track3-rpe"].append(
            {"ref": 2, "date": (now - dt.timedelta(hours=6)).isoformat()}
        )
        status = cg.check_weekly_quota(
            "track_3",
            kaggle_submissions_by_slug=kaggle_subs,
            now_utc=now,
        )
        self.assertFalse(status.ok)
        self.assertTrue(status.cross_board_divergence)
        self.assertTrue(any("CROSS_BOARD_COUNT_DIVERGENCE" in r for r in status.blocking_reasons))

    def test_track_scope_and_standalone_track3_mesh_blocked(self) -> None:
        policy = {"allowed_tracks": ["track_1", "track_3"]}
        _, errs_ok = cg.check_track_scope("track_1", policy=policy)
        self.assertEqual(errs_ok, [])

        _, errs_unauthorized = cg.check_track_scope("track_2_tier2", policy=policy)
        self.assertTrue(any("TRACK_SCOPE_VIOLATION" in e for e in errs_unauthorized))

        _, errs_mesh = cg.check_track_scope("track_3_mesh", policy=policy)
        self.assertTrue(any("STANDALONE_TRACK3_MESH_BLOCKED" in e for e in errs_mesh))

    def test_bundle_atomicity_blocks_only_probing_without_matching_failed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sub_file = Path(tmp) / "track3.parquet"
            sub_file.write_bytes(b"parquet-payload-v1")
            file_sha = cg.sha256_file(sub_file)

            # 1. --only without --retry-receipt is blocked
            errs1 = cg.check_bundle_atomicity("track_3", sub_file, only_metrics=["mppe_cm"])
            self.assertTrue(any("SPLIT_LEADERBOARD_PROBING_BLOCKED" in e for e in errs1))

            # 2. --only with a receipt from a different file SHA-256 is blocked
            receipt_path = Path(tmp) / "receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "track": "track_3",
                        "file_sha256": "a" * 64,
                        "submissions": {
                            "add_auc": {"slug": "v2d-challenge-track3-auc", "state": "PENDING"},
                            "mppe_cm": {"slug": "v2d-challenge-track3-mppe", "state": "UPLOAD_FAILED"},
                        },
                    }
                )
            )
            errs2 = cg.check_bundle_atomicity(
                "track_3",
                sub_file,
                only_metrics=["mppe_cm"],
                retry_receipt_path=receipt_path,
            )
            self.assertTrue(any("RETRY_RECEIPT_HASH_MISMATCH" in e for e in errs2))

            # 3. Valid retry receipt with matching SHA-256 and UPLOAD_FAILED state succeeds
            receipt_path.write_text(
                json.dumps(
                    {
                        "track": "track_3",
                        "file_sha256": file_sha,
                        "submissions": {
                            "add_auc": {"slug": "v2d-challenge-track3-auc", "state": "PENDING"},
                            "mppe_cm": {"slug": "v2d-challenge-track3-mppe", "state": "UPLOAD_FAILED"},
                        },
                    }
                )
            )
            errs3 = cg.check_bundle_atomicity(
                "track_3",
                sub_file,
                only_metrics=["mppe_cm"],
                retry_receipt_path=receipt_path,
            )
            self.assertEqual(errs3, [])


class TestAgentUploadAuthorizationAndHookGuard(unittest.TestCase):
    def test_agent_environment_blocks_live_upload_but_allows_dry_run(self) -> None:
        env = {"CLAUDECODE": "1"}
        self.assertEqual(
            cg.check_upload_authorization(dry_run=True, env=env, stdin_isatty=True),
            [],
        )
        errs = cg.check_upload_authorization(
            dry_run=False,
            confirm_human_upload=True,
            env=env,
            stdin_isatty=True,
        )
        self.assertTrue(any("AGENT_LIVE_UPLOAD_PROHIBITED" in e for e in errs))

    def test_ci_push_blocks_live_upload_while_approved_workflow_dispatch_allows(self) -> None:
        push_env = {"CI": "true", "GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "push"}
        errs_push = cg.check_upload_authorization(
            dry_run=False,
            confirm_human_upload=True,
            env=push_env,
            stdin_isatty=False,
        )
        self.assertTrue(any("CI_UPLOAD_LOCKED" in e for e in errs_push))

        cd_env = {
            "CI": "true",
            "GITHUB_ACTIONS": "true",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "V2D_CI_APPROVED_UPLOAD": "1",
        }
        self.assertEqual(
            cg.check_upload_authorization(
                dry_run=False,
                confirm_human_upload=True,
                env=cd_env,
                stdin_isatty=False,
            ),
            [],
        )

    def test_pretooluse_hook_blocks_live_v2d_submit_and_allows_dry_run(self) -> None:
        bad_payload = {
            "tool_name": "Bash",
            "tool_input": {"command": "python3 submit/v2d_submit.py sub.parquet --track track_3 -m test"},
        }
        self.assertEqual(
            guard.inspect_hook_payload(bad_payload),
            "prohibited_upload_command:v2d_submit_without_dry_run",
        )

        good_payload = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "python3 submit/v2d_submit.py sub.parquet --track track_3 -m test --dry-run"
            },
        }
        self.assertIsNone(guard.inspect_hook_payload(good_payload))


class TestAntiRewardHackingArtifactAudits(unittest.TestCase):
    def _build_track3_df(
        self,
        *,
        commit_url: str = "https://github.com/team-robotics/v2d-entry/commit/7f4c9a12b8e30194d5a67c8e90123456789abcde",
        ckpt_ep0: str | None = None,
        ckpt_ep1: str | None = None,
        mesh0: tuple[np.ndarray, np.ndarray] | None = None,
        mesh1: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        v0, f0 = mesh0 or _make_octahedron_mesh(scale=0.08, offset=0.0)
        v1, f1 = mesh1 or _make_octahedron_mesh(scale=0.11, offset=0.02)
        h_eval_script = next(iter(cg.RELEASED_EVAL_SCRIPT_SHA256))
        h_eval_code = hashlib.sha256(b"real_eval_code_v1").hexdigest()
        h_packer = hashlib.sha256(b"real_packer_v1").hexdigest()
        h_uploader = hashlib.sha256(b"real_uploader_v1").hexdigest()
        c0 = ckpt_ep0 or hashlib.sha256(b"real_safetensors_weights_ep1").hexdigest()
        c1 = ckpt_ep1 or hashlib.sha256(b"real_safetensors_weights_ep3").hexdigest()

        rows = []
        for ep, ckpt, (mv, mf) in ((1, c0, (v0, f0)), (3, c1, (v1, f1))):
            for f in range(4):
                rows.append(
                    {
                        "row_id": f"track_3/e{ep:06d}/w000/f{f:06d}/o0",
                        "pos_x": 0.1 * f,
                        "pos_y": 0.2,
                        "pos_z": 0.5,
                        "quat_x": 0.0,
                        "quat_y": 0.0,
                        "quat_z": 0.0,
                        "quat_w": 1.0,
                        "eval_script_sha256": h_eval_script,
                        "eval_code_sha256": h_eval_code,
                        "checkpoint_sha256": ckpt,
                        "packer_script_sha256": h_packer,
                        "uploader_script_sha256": h_uploader,
                        "code_commit_url": commit_url,
                        "object_mesh_b64": _encode_mesh_b64(mv, mf) if f == 0 else "",
                    }
                )
        return pd.DataFrame(rows)

    def test_valid_track3_submission_passes_all_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "track3_valid.parquet"
            self._build_track3_df().to_parquet(path, index=False)
            report = cg.audit_submission_artifact(path, "track_3")
            self.assertTrue(report.ok, f"Unexpected failures: {report.failures}")

    def test_detects_microscopic_mesh_noise_evasion_and_reference_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            v0, f0 = _make_octahedron_mesh(scale=0.08)
            # Add 1e-7 m Gaussian noise to v0 for episode 3 to evade exact SHA-256 check
            rng = np.random.default_rng(99)
            v1_noisy = v0 + rng.normal(0.0, 1e-7, size=v0.shape)

            path = Path(tmp) / "track3_noisy.parquet"
            self._build_track3_df(mesh0=(v0, f0), mesh1=(v1_noisy, f0)).to_parquet(path, index=False)
            report = cg.audit_submission_artifact(
                path,
                "track_3",
                reference_meshes={"public_scan.glb": v0},
            )
            self.assertFalse(report.ok)
            self.assertTrue(any("MICROSCOPIC_MESH_NOISE_EVASION" in f for f in report.failures))
            self.assertTrue(any("PROHIBITED_REFERENCE_MESH_COPY" in f for f in report.failures))

    def test_detects_synthetic_checkpoint_hashes_and_prohibited_commits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            synth_ckpt = hashlib.sha256(b"episode_000001").hexdigest()
            bad_commit = "https://github.com/nvidia-isaac/video_to_data/commit/8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef"
            path = Path(tmp) / "track3_bad_prov.parquet"
            self._build_track3_df(commit_url=bad_commit, ckpt_ep0=synth_ckpt).to_parquet(
                path, index=False
            )
            report = cg.audit_submission_artifact(path, "track_3")
            self.assertFalse(report.ok)
            self.assertTrue(any("PROHIBITED_UPSTREAM_COMMIT_URL" in f for f in report.failures))
            self.assertTrue(any("SYNTHETIC_CHECKPOINT_HASH" in f for f in report.failures))

    def test_detects_track2_cross_tier_ground_truth_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tier1_dir = tmp_path / "tier1_ref"
            tier1_dir.mkdir()
            # LeRobot Tier 1 parquet stores [x, y, z, qw, qx, qy, qz]
            ref_poses = [
                np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.2, 0.3, 0.4, 0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64),
            ]
            pd.DataFrame({"observation.object.pose": ref_poses}).to_parquet(
                tier1_dir / "episode_000000.parquet", index=False
            )

            # Build a Track 2 Tier 2 CSV that copies Tier 1 orientations
            h_eval_script = next(iter(cg.RELEASED_EVAL_SCRIPT_SHA256))
            h_eval_code = hashlib.sha256(b"eval_code").hexdigest()
            h_ckpt = hashlib.sha256(b"real_ckpt_weights").hexdigest()
            h_pack = hashlib.sha256(b"pack").hexdigest()
            h_up = hashlib.sha256(b"up").hexdigest()
            commit = "https://github.com/team-robotics/v2d-entry/commit/7f4c9a12b8e30194d5a67c8e90123456789abcde"
            sub_rows = [
                {
                    "row_id": "track_2/e000000/w000/f000000/o0",
                    "pos_x": 0.1,
                    "pos_y": 0.2,
                    "pos_z": 0.3,
                    "quat_x": 0.0,
                    "quat_y": 0.0,
                    "quat_z": 0.0,
                    "quat_w": 1.0,
                    "eval_script_sha256": h_eval_script,
                    "eval_code_sha256": h_eval_code,
                    "checkpoint_sha256": h_ckpt,
                    "packer_script_sha256": h_pack,
                    "uploader_script_sha256": h_up,
                    "code_commit_url": commit,
                },
                {
                    "row_id": "track_2/e000000/w000/f000001/o0",
                    "pos_x": 0.2,
                    "pos_y": 0.3,
                    "pos_z": 0.4,
                    "quat_x": 0.70710678,
                    "quat_y": 0.0,
                    "quat_z": 0.0,
                    "quat_w": 0.70710678,
                    "eval_script_sha256": h_eval_script,
                    "eval_code_sha256": h_eval_code,
                    "checkpoint_sha256": h_ckpt,
                    "packer_script_sha256": h_pack,
                    "uploader_script_sha256": h_up,
                    "code_commit_url": commit,
                },
            ]
            csv_path = tmp_path / "track2_tier2_leaked.csv"
            pd.DataFrame(sub_rows).to_csv(csv_path, index=False)

            report = cg.audit_submission_artifact(
                csv_path,
                "track_2_tier2",
                tier1_reference_dir=tier1_dir,
            )
            self.assertFalse(report.ok)
            self.assertTrue(any("CROSS_TIER_GROUND_TRUTH_LEAKAGE" in f for f in report.failures))

    def test_detects_track1_frozen_rotation_metric_gaming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            v0, f0 = _make_octahedron_mesh(scale=0.08)
            commit = "https://github.com/team-robotics/v2d-entry/commit/7f4c9a12b8e30194d5a67c8e90123456789abcde"
            rows = []
            # Episode 0 constants at frame 999999: array 5 (vertices), array 6 (faces)
            for idx, v in enumerate(v0):
                rows.append(
                    {
                        "row_id": f"t1_0_999999_5_{idx}",
                        "x": float(v[0]),
                        "y": float(v[1]),
                        "z": float(v[2]),
                        "code_commit_url": commit,
                    }
                )
            for idx, face in enumerate(f0):
                rows.append(
                    {
                        "row_id": f"t1_0_999999_6_{idx}",
                        "x": float(face[0]),
                        "y": float(face[1]),
                        "z": float(face[2]),
                        "code_commit_url": commit,
                    }
                )
            # Dynamic frames 0..3 with frozen root rotation (pt=1) and frozen object rotation (arr=3)
            for frame in range(4):
                rows.append(
                    {
                        "row_id": f"t1_0_{frame}_0_1",
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "code_commit_url": commit,
                    }
                )
                rows.append(
                    {
                        "row_id": f"t1_0_{frame}_0_2",
                        "x": 0.1 * frame,
                        "y": 0.2 * frame,
                        "z": 0.3 * frame,
                        "code_commit_url": commit,
                    }
                )
                for r_pt, diag in enumerate(([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])):
                    rows.append(
                        {
                            "row_id": f"t1_0_{frame}_3_{r_pt}",
                            "x": diag[0],
                            "y": diag[1],
                            "z": diag[2],
                            "code_commit_url": commit,
                        }
                    )
            pq_path = Path(tmp) / "track1_frozen.parquet"
            pd.DataFrame(rows).to_parquet(pq_path, index=False)

            report = cg.audit_submission_artifact(pq_path, "track_1")
            self.assertFalse(report.ok)
            self.assertTrue(any("DEGENERATE_FROZEN_ROOT_ROTATION" in f for f in report.failures))
            self.assertTrue(any("DEGENERATE_FROZEN_OBJECT_ROTATION" in f for f in report.failures))

    def test_submission_kit_zip_manifest_and_sync(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        kit_zip = repo_root / "docs" / "v2d_challenge" / "assets" / "v2d_submission_kit.zip"
        self.assertTrue(kit_zip.exists(), f"Missing {kit_zip}")

        with zipfile.ZipFile(kit_zip, "r") as zf:
            names = set(zf.namelist())
            required_entries = {
                "v2d_submission_kit/KIT_MANIFEST.json",
                "v2d_submission_kit/tools/compliance_gate.py",
                "v2d_submission_kit/tools/agent_submission_guard.py",
                "v2d_submission_kit/config/v2d_policy.example.json",
                "v2d_submission_kit/.github/workflows/v2d_submission_ci.yml",
                "v2d_submission_kit/submit/v2d_submit.py",
                "v2d_submission_kit/tools/preflight.py",
                "v2d_submission_kit/skills/v2d-submit/SKILL.md",
            }
            self.assertTrue(required_entries.issubset(names))

            manifest = json.loads(zf.read("v2d_submission_kit/KIT_MANIFEST.json").decode("utf-8"))
            for rel_path, expected_sha in manifest["files"].items():
                arc_path = f"v2d_submission_kit/{rel_path}"
                if arc_path in names:
                    actual_sha = hashlib.sha256(zf.read(arc_path)).hexdigest()
                    self.assertEqual(
                        actual_sha,
                        expected_sha,
                        f"SHA-256 mismatch in KIT_MANIFEST.json for {rel_path}",
                    )

            # Verify byte-for-byte sync between repo tools/v2d_compliance and kit bundle
            self.assertEqual(
                zf.read("v2d_submission_kit/tools/compliance_gate.py"),
                (repo_root / "tools" / "v2d_compliance" / "compliance_gate.py").read_bytes(),
            )
            self.assertEqual(
                zf.read("v2d_submission_kit/tools/agent_submission_guard.py"),
                (repo_root / "tools" / "v2d_compliance" / "agent_submission_guard.py").read_bytes(),
            )
            self.assertEqual(
                zf.read("v2d_submission_kit/.github/workflows/v2d_submission_ci.yml"),
                (repo_root / "tools" / "v2d_compliance" / "participant_ci_template.yml").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
