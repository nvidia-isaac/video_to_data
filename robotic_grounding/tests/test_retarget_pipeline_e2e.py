#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""End-to-end gate for the retarget pipeline.

Runs every stage in ``workflow/retarget.yaml`` except ``load`` against a
committed ``{dataset}_loaded/`` fixture. The ``load`` stage needs raw
dataset data we can't commit (license + size), so the test starts from
the loaded parquet.

Stages exercised (in ``workflow/retarget.yaml`` order):
  1.5 URDFs      -> ``scripts/generate_rigid_urdfs.py``
  2   process    -> ``scripts/retarget/run_retarget.py``
  3   reconstruct-> ``scripts/reconstruct_support_surfaces.py``
  4   visualize  -> ``scripts/retarget/vis_retargeted.py``
  5   video      -> ``scripts/rsl_rl/dummy_agent.py``

Coverage today: **taco only**. An ``arctic`` case will be added when an
``arctic_loaded/`` fixture lands in the repo.

Requires Isaac Lab (``ISAACLAB_PATH``), GPU, and ``pxr``.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import torch


class TestRetargetPipelineE2E(unittest.TestCase):
    """End-to-end tests for retarget pipeline stages 1.5 through 5."""

    project_root: Path
    scripts_dir: Path
    assets_dir: Path
    isaaclab_path: str
    isaaclab_script: str
    # Lib dir to prepend to LD_LIBRARY_PATH so pinocchio imports; None when the
    # Isaac Lab python already imports pinocchio cleanly (see
    # ``_ensure_pinocchio_ld_path``).
    _pinocchio_lib_dir: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        """Resolve paths and enforce Isaac Lab + GPU preconditions."""
        cls.project_root = Path(__file__).parent.parent.absolute()
        cls.scripts_dir = cls.project_root / "scripts"
        cls.assets_dir = (
            cls.project_root
            / "source"
            / "robotic_grounding"
            / "robotic_grounding"
            / "assets"
            / "human_motion_data"
        )

        isaaclab_path = os.environ.get("ISAACLAB_PATH")
        if not isaaclab_path:
            raise unittest.SkipTest("ISAACLAB_PATH environment variable is not set")
        isaaclab_script = os.path.join(isaaclab_path, "isaaclab.sh")
        if not os.path.exists(isaaclab_script):
            raise unittest.SkipTest(f"IsaacLab script not found at {isaaclab_script}")
        cls.isaaclab_path = isaaclab_path
        cls.isaaclab_script = isaaclab_script

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available — retarget E2E requires GPU")

        print(f"GPU available: {torch.cuda.get_device_name(0)}")

        cls._ensure_pinocchio_ld_path()

    @classmethod
    def _ensure_pinocchio_ld_path(cls) -> None:
        """Make ``import pinocchio`` work inside ``isaaclab.sh -p`` subprocesses.

        The Isaac Lab container ships pinocchio (pin 3.7.0) compiled against
        urdfdom v4 / tinyxml2 v10, but the system cmeel wheels are v6 / v11, so
        ``import pinocchio`` dies with::

            ImportError: liburdfdom_sensor.so.4.0: cannot open shared object file

        ``scripts/retarget/process_soma_sequence.sh::setup_pinocchio_ld_path``
        fixes this for the OSMO workflow by installing the matching cmeel wheels
        into a side prefix and prepending its ``lib/`` to ``LD_LIBRARY_PATH``.
        This test invokes each stage directly via ``isaaclab.sh -p`` (not
        through that script), so we replicate the fix here and stash the lib dir
        on the class for ``_run`` to inject into every stage's environment.

        No-op (leaves ``_pinocchio_lib_dir`` None) when the Isaac Lab python
        already imports pinocchio cleanly — e.g. on an image where the cmeel
        soversion mismatch has been fixed at build time.
        """
        cls._pinocchio_lib_dir = None

        def _pinocchio_imports() -> bool:
            return (
                subprocess.run(
                    [cls.isaaclab_script, "-p", "-c", "import pinocchio"],
                    capture_output=True,
                    check=False,
                ).returncode
                == 0
            )

        if _pinocchio_imports():
            print("pinocchio import OK (no LD_LIBRARY_PATH fix needed)")
            return

        cache_dir = Path(
            os.environ.get(
                "PINOCCHIO_DEPS_PREFIX",
                str(Path.home() / ".cache" / "robotic_grounding" / "pinocchio_deps"),
            )
        )
        lib_dir = cache_dir / "cmeel.prefix" / "lib"
        have_libs = (lib_dir / "liburdfdom_sensor.so.4.0").exists() and (
            lib_dir / "libtinyxml2.so.10"
        ).exists()
        if not have_libs:
            print(f"Installing pinocchio v4/v10 cmeel deps to {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            # ``--no-deps`` so pip doesn't pull cmeel-tinyxml2 v11 (urdfdom
            # 4.0.1's metadata range otherwise resolves to v11; we need v10).
            subprocess.run(
                [
                    cls.isaaclab_script,
                    "-p",
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(cache_dir),
                    "--no-deps",
                    "cmeel-urdfdom==4.0.1",
                    "cmeel-tinyxml2==10.0.0",
                ],
                capture_output=True,
                check=False,
            )

        cls._pinocchio_lib_dir = str(lib_dir)
        print(f"pinocchio v4 deps: {lib_dir} (prepended to LD_LIBRARY_PATH)")

        check = subprocess.run(
            [cls.isaaclab_script, "-p", "-c", "import pinocchio"],
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "LD_LIBRARY_PATH": f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}",
            },
        )
        if check.returncode != 0:
            print(
                "WARNING: pinocchio still fails to import after the "
                f"LD_LIBRARY_PATH fix; stages will hit the ImportError.\n"
                f"{check.stderr[-1000:]}"
            )

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------
    def _run(
        self,
        argv: list[str],
        *,
        stage: str,
        timeout: int,
    ) -> subprocess.CompletedProcess:
        """Invoke a stage via ``isaaclab.sh -p``; fail the test on non-zero exit.

        Every retarget stage that ships in this repo is invoked in CI as
        ``${ISAACLAB_PATH}/isaaclab.sh -p <script> <args>`` so that
        ``pxr``/Isaac Sim imports resolve the same way they do in the
        OSMO workflow. We reuse that here for parity.
        """
        cmd = [self.isaaclab_script, "-p", *argv]
        env = dict(os.environ)
        env.setdefault("OMNI_HEADLESS", "1")
        env.setdefault("WANDB_MODE", "disabled")
        # Make ``import pinocchio`` resolvable for every stage that pulls in the
        # retarget pipeline (1.5 URDFs, 2 process, 4 visualize). See
        # ``_ensure_pinocchio_ld_path``.
        if self._pinocchio_lib_dir:
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{self._pinocchio_lib_dir}:{prev}" if prev else self._pinocchio_lib_dir
            )

        print(f"\n--- stage: {stage} ---")
        print(f"cmd: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            # With text=True, stdout/stderr on TimeoutExpired come back as str,
            # but mypy doesn't know that — normalize via a local decode helper.
            def _tail(buf: bytes | str | None) -> str:
                if buf is None:
                    return ""
                s = buf if isinstance(buf, str) else buf.decode(errors="replace")
                return s[-2000:]

            self.fail(
                f"[{stage}] timed out after {timeout}s\n"
                f"stdout tail:\n{_tail(e.stdout)}\n"
                f"stderr tail:\n{_tail(e.stderr)}"
            )
        if result.returncode != 0:
            self.fail(
                f"[{stage}] exit code {result.returncode}\n"
                f"stderr tail:\n{result.stderr[-2000:] if result.stderr else ''}"
            )
        return result

    # ------------------------------------------------------------------
    # Per-stage helpers
    # ------------------------------------------------------------------
    def _stage_urdfs(self, dataset: str) -> None:
        """Stage 1.5: regenerate rigid URDFs for the dataset's objects."""
        self._run(
            [
                str(self.scripts_dir / "generate_rigid_urdfs.py"),
                "--dataset",
                dataset,
            ],
            stage="1.5 URDFs",
            timeout=120,
        )
        urdf_dir = (
            self.project_root
            / "source/robotic_grounding/robotic_grounding/assets/urdfs"
            / dataset
        )
        self.assertTrue(
            urdf_dir.is_dir() and any(urdf_dir.glob("*.urdf")),
            f"No URDFs under {urdf_dir} after generate_rigid_urdfs.py --dataset {dataset}",
        )

    def _stage_process(
        self,
        dataset: str,
        sequence_id: str,
        loaded_dir: Path,
        processed_dir: Path,
    ) -> Path:
        """Stage 2: IK-retarget MANO to robot; assert schema + non-empty columns."""
        self._run(
            [
                str(self.scripts_dir / "retarget/run_retarget.py"),
                "--dataset",
                dataset,
                "--input_dir",
                str(loaded_dir),
                "--output_dir",
                str(processed_dir),
                "--device",
                "cuda:0",
                "--save",
                "--sequence_id",
                sequence_id,
            ],
            stage="2 process",
            timeout=180,
        )
        parquet_files = list(
            processed_dir.rglob(f"sequence_id={sequence_id}/robot_name=*/*.parquet")
        )
        self.assertTrue(
            parquet_files,
            f"No processed parquet under {processed_dir} for {sequence_id}",
        )
        table = pq.read_table(parquet_files[0]).to_pydict()
        # Required fields from BASE_FIELDS + MANO_FIELDS + SHARPA_FIELDS +
        # OBJECT_FIELDS (data_logger.py:680). We spot-check the ones most
        # likely to break on a retarget regression — a missing column or an
        # empty trajectory is the signal we want.
        required = (
            "object_body_position",
            "robot_right_wrist_position",
            "robot_left_wrist_position",
            "robot_right_finger_joints",
            "robot_left_finger_joints",
            "mano_right_finger_pose",
        )
        for col in required:
            self.assertIn(col, table, f"Column {col!r} missing in {parquet_files[0]}")
            self.assertTrue(
                table[col][0],
                f"Column {col!r} has empty trajectory in {parquet_files[0]}",
            )
        return parquet_files[0]

    def _stage_reconstruct(
        self,
        dataset: str,
        sequence_id: str,
        loaded_dir: Path,
    ) -> None:
        """Stage 3: build support-surface USDs; assert USD is parseable with ≥1 mesh."""
        self._run(
            [
                str(self.scripts_dir / "reconstruct_support_surfaces.py"),
                "--dataset",
                dataset,
                "--input_dir",
                str(loaded_dir),
                "--sequence_id",
                sequence_id,
            ],
            stage="3 reconstruct",
            timeout=180,
        )
        # reconstruct_support_surfaces.py writes to
        # ``input_dir.parent / 'reconstructed_stage' / f'{seq}_support.usda'``
        recon_dir = loaded_dir.parent / "reconstructed_stage"
        usd_path = recon_dir / f"{sequence_id}_support.usda"
        self.assertTrue(
            usd_path.exists(),
            f"Support-surface USD not found at {usd_path}",
        )
        from pxr import Usd  # noqa: PLC0415 — pxr is Docker-only, must be deferred

        stage = Usd.Stage.Open(str(usd_path))
        self.assertIsNotNone(stage, f"Usd.Stage.Open returned None for {usd_path}")
        # Support surfaces are written as ``UsdGeom.Cylinder`` disks
        # (see ``scripts/reconstruct_support_surfaces.py::create_disk``).
        cylinder_prims = [p for p in stage.Traverse() if p.GetTypeName() == "Cylinder"]
        self.assertTrue(
            cylinder_prims,
            f"No Cylinder (support-surface) prims in {usd_path}",
        )

    def _stage_visualize(
        self,
        sequence_id: str,
        processed_dir: Path,
        html_dir: Path,
    ) -> None:
        """Stage 4: viser recording + pyrender MP4; assert both artifacts land."""
        self._run(
            [
                str(self.scripts_dir / "retarget/vis_retargeted.py"),
                "--input_dir",
                str(processed_dir),
                "--save_html",
                "--save_mp4",
                "--html_dir",
                str(html_dir),
                "--sequence_id",
                sequence_id,
            ],
            stage="4 visualize",
            timeout=240,
        )
        recordings = html_dir / "recordings"
        self.assertTrue(
            (recordings / f"{sequence_id}.viser").exists(),
            f"No .viser for {sequence_id} in {recordings}",
        )
        self.assertTrue(
            (recordings / f"{sequence_id}.mp4").exists(),
            f"No .mp4 for {sequence_id} in {recordings}",
        )

    def _stage_video(
        self,
        sequence_id: str,
        processed_dir: Path,
        video_dir: Path,
    ) -> None:
        """Stage 5: Isaac Sim playback via dummy_agent; gates termination regressions.

        We record 100 steps: the TACO termination regression crashed at
        step ~100 (``retargeted_horizon ≈ 102`` for this sequence), so 100
        is the smallest value that ticks past the trajectory end when
        terminations are disabled again.
        """
        seq_dirs = list(processed_dir.rglob(f"sequence_id={sequence_id}/robot_name=*"))
        self.assertTrue(
            seq_dirs, f"Could not locate processed SEQ_DIR under {processed_dir}"
        )
        self._run(
            [
                str(self.scripts_dir / "rsl_rl/dummy_agent.py"),
                "--task",
                "Sharpa-V2P-v0-Play",
                "--motion_file",
                str(seq_dirs[0]),
                "--num_envs",
                "1",
                "--headless",
                "--record_video",
                "--output_dir",
                str(video_dir),
                "--video_length",
                "100",
            ],
            stage="5 video",
            # Isaac Sim startup alone takes ~85 s on CI's A10G, leaving
            # <155 s for env setup + 100-step loop + video encode at the
            # previous 240 s timeout (observed timing out SIGKILL in CI on
            # 2026-04-23). Bump headroom so we gate on success/failure,
            # not runner speed. Local RTX 5880 finishes in ~60 s.
            timeout=420,
        )
        mp4s = list(video_dir.glob("*.mp4"))
        self.assertTrue(mp4s, f"No MP4 written to {video_dir}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _run_pipeline(self, dataset: str, sequence_id: str) -> None:
        """Drive all stages for one dataset inside a tmp scratch workspace.

        We copy the committed ``{dataset}_loaded/`` fixture into a tmpdir so
        that reconstruct's default output location
        (``input_dir.parent/reconstructed_stage``) lands in the tmpdir
        instead of mutating the committed source tree. SceneConfig's
        ``_discover_support_surface`` walks up from the processed parquet to
        find ``reconstructed_stage`` alongside it, so the whole chain stays
        hermetic to the tmpdir.
        """
        committed_loaded = self.assets_dir / dataset / f"{dataset}_loaded"
        self.assertTrue(
            committed_loaded.is_dir(),
            f"Missing committed loaded fixture: {committed_loaded}",
        )

        with tempfile.TemporaryDirectory(prefix=f"retarget_e2e_{dataset}_") as tmp:
            workdir = Path(tmp) / dataset
            workdir.mkdir(parents=True)

            loaded_dir = workdir / f"{dataset}_loaded"
            shutil.copytree(committed_loaded, loaded_dir)

            processed_dir = workdir / f"{dataset}_processed"
            html_dir = workdir / f"{dataset}_html"
            video_dir = workdir / f"{dataset}_video"

            self._stage_urdfs(dataset)
            self._stage_process(dataset, sequence_id, loaded_dir, processed_dir)
            self._stage_reconstruct(dataset, sequence_id, loaded_dir)
            self._stage_visualize(sequence_id, processed_dir, html_dir)
            self._stage_video(sequence_id, processed_dir, video_dir)

    def test_taco_pipeline(self) -> None:
        """All retarget.yaml stages pass for the committed taco sequence."""
        self._run_pipeline("taco", "taco_empty__kettle__plate_20231031_060")


if __name__ == "__main__":
    unittest.main(verbosity=2)
