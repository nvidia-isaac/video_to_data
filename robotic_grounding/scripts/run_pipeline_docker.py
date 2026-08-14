# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-launched, full hand->robot retargeting pipeline (no manual container shell-in).

This is the **Pattern A** entry point: a plain host Python script that builds and
runs ``docker run`` commands itself — mirroring reconstruction's
``v2d.docker.container.run_in_container`` philosophy ("host orchestrates,
containers infer"). You run it from the host; each stage spins up the right
image, does its work, and exits.

It spans the same two images (loader + robotic-grounding):

  raw dataset --[LOAD]--> {ds}_loaded --[RETARGET...]--> {ds}_processed -> support/vis/quality
   (+ MANO)   v2d_task_library_loader            robotic-grounding:latest

Stages (select with --stages; default runs everything except dummy):
  load      raw -> {ds}_loaded            (image: v2d_task_library_loader, MANO FK)
  urdf      object STL + rigid URDF       (robotic-grounding)
  processed {ds}_loaded -> {ds}_processed (robotic-grounding, IK retarget)
  support   support-surface reconstruction(robotic-grounding)
  vis       viser HTML (+ MP4)            (robotic-grounding, pyrender)
  dummy     Isaac playback MP4            (robotic-grounding + Kit; heavy, opt-in)
  assess    data-quality / penetration filter (robotic-grounding)

A stage that produces nothing ABORTS the run (non-zero exit) rather than letting the
later stages fall through onto a previous run's artifacts: `load` must write output for
THIS invocation, and `processed`/`support` must have sequences to read. When a stage
does legitimately reuse artifacts an earlier run produced (e.g. `--stages processed,vis`
without `load`), it says so and prints when they were written.

The `assess` stage runs by default under `--stages auto` (report-only: writes
{ds}_quality.csv; a data_assessor failure aborts the run). Only the filtering is opt-in:
  --assess                      force assess when using a custom --stages that omits it
  --reject                      also write a rejected-IDs file (fail => rejected)
  --filter-penetration          run ONLY the hand_penetration check
  --penetration-threshold CM    penetration depth tolerance in cm (default check value)
  --disable-checks a,b          drop specific checks   --checks a,b  run only these

The in-container shell pipeline (run_retarget_local.sh) is unchanged and remains
the **Pattern B** path for when you prefer ``./workflow/run.sh start`` + a shell.

Examples:
  # Full taco pipeline from the host, HTML+MP4, no Isaac, no filtering:
  python scripts/run_pipeline_docker.py taco \
      --mano-dir /data/RELEASE_TEST/mano --hmd /data/RELEASE_TEST

  # Just (re)retarget + support on 5 seqs, then filter penetrations >2cm:
  python scripts/run_pipeline_docker.py taco --hmd /data/RELEASE_TEST \
      --stages processed,support --max-sequences 5 \
      --assess --reject --filter-penetration --penetration-threshold 2.0
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# --- repo / container path constants ----------------------------------------
# This file: robotic_grounding/scripts/run_pipeline_docker.py
RG_DIR = Path(__file__).resolve().parent.parent  # robotic_grounding/
MONO_DIR = RG_DIR.parent  # monorepo root (host)
RECON_DIR = MONO_DIR / "reconstruction"
ASSETS_DIR = RG_DIR / "source" / "robotic_grounding" / "robotic_grounding" / "assets"

# Where workflow/run.sh mounts the host HUMAN_MOTION_DATA_DIR inside the
# robotic-grounding container. We mount the same way so paths line up.
CONTAINER_MONO = "/workspace/video_to_data"
CONTAINER_RG = f"{CONTAINER_MONO}/robotic_grounding"
CONTAINER_ASSETS = f"{CONTAINER_RG}/source/robotic_grounding/robotic_grounding/assets"
CONTAINER_DATA_DIR = f"{CONTAINER_ASSETS}/human_motion_data"
RG_IMAGE_DEFAULT = "robotic-grounding:latest"

ALL_STAGES = [
    "load",
    "segment",
    "urdf",
    "processed",
    "support",
    "vis",
    "dummy",
    "assess",
]
DEFAULT_STAGES = ["load", "urdf", "processed", "support", "vis", "assess"]
# Datasets whose loaded sequences are long egocentric recordings that must be
# segmented into atomic interaction clips (Stage 1.6) before retargeting. For
# these, `processed`/`support` consume {ds}_loaded_segmented, not {ds}_loaded.
SEGMENT_DATASETS = {"hot3d"}
# Datasets whose object meshes ship INSIDE the dataset download (not as a separate
# object_assets/URDF set). The loader reads them from <HMD>/<ds>/dataset/... (paths
# baked at load time), so we OMIT --object_assets_dir (per run_loader.py) and, in the
# retarget stages, also mount the data root at /data/human_motion_data so those baked
# mesh paths resolve. URDFs (if needed) are generated from the downloaded meshes.
RUNTIME_MESH_DATASETS = {"grab", "h2o", "dexycb"}

# Recommended stages per dataset, used when --stages is "auto" (the default). The
# orchestrator prints these with a one-line rationale so a new user doesn't need to
# know each dataset's quirks. `segment` still auto-inserts for SEGMENT_DATASETS.
RECOMMENDED_STAGES = {
    "taco": ["load", "urdf", "processed", "support", "vis", "assess"],
    "hot3d": [
        "load",
        "urdf",
        "processed",
        "support",
        "vis",
        "assess",
    ],  # + segment (auto)
    "oakink2": ["load", "urdf", "processed", "support", "vis", "assess"],
    "arctic": [
        "load",
        "processed",
        "support",
        "vis",
        "assess",
    ],  # no urdf: articulated, shipped
    "grab": [
        "load",
        "urdf",
        "processed",
        "support",
        "vis",
        "assess",
    ],  # runtime-mesh, rigid
    "h2o": ["load", "urdf", "processed", "support", "vis", "assess"],
    "dexycb": ["load", "urdf", "processed", "support", "vis", "assess"],
}
# Object URDFs are generated by the `urdf` stage (generate_rigid_urdfs) from the object
# meshes and written to the data-root object_assets/urdfs/<ds>. We generate them by default
# for every RIGID dataset so each produced dataset is sim/RL-ready. The EXCEPTION is arctic:
# its objects are articulated — the URDFs can't be generated, are REQUIRED at LOAD (MuJoCo
# FK), and are supplied via the ArtiGrasp download (see ARCTIC_SETUP.md).
RECOMMEND_NOTES = {
    "hot3d": "'segment' (Stage 1.6) auto-inserts; processed/support read the segmented clips. "
    "'urdf' generates the object URDFs into object_assets",
    "arctic": "no 'urdf' GEN stage — objects are articulated; their URDFs are REQUIRED at LOAD "
    "(MuJoCo FK) and SHIPPED via ArtiGrasp (generate_rigid_urdfs can't build them)",
    "grab": "runtime-mesh, rigid; 'urdf' generates the object URDFs from the dataset's .ply",
    "h2o": "runtime-mesh, rigid, single-object takes; 'urdf' generates the object URDFs from "
    "the dataset meshes. Support reconstruction uses a hand-release gate (h2o-only) so an "
    "object held steady mid-interaction isn't mistaken for a resting surface",
    "dexycb": "runtime-mesh, rigid; 'urdf' generates the object URDFs from the dataset meshes",
    "taco": "rigid; 'urdf' generates the object URDFs into object_assets",
}


def _run(cmd: list[str], dry: bool, cwd: str | None = None) -> None:
    """Print and execute a command, aborting the pipeline on failure."""
    prefix = f"(cd {cwd}) " if cwd else ""
    print("\n$ " + prefix + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return
    rc = subprocess.run(cmd, cwd=cwd, check=False).returncode
    if rc != 0:
        sys.exit(f"\ncommand failed (exit {rc}); aborting.")


def build_images(args: argparse.Namespace, stages: list[str] | None, dry: bool) -> None:
    """Build the Docker image(s) needed. `stages=None` builds BOTH (used by --build-only).

    - loader image v2d_task_library_loader (reconstruction): install the host
      orchestration packages, then build.
    - robotic-grounding:latest via workflow/run.sh build.
    """
    do_loader = stages is None or "load" in stages
    do_rg = stages is None or any(
        s in stages
        for s in ("segment", "urdf", "processed", "support", "vis", "dummy", "assess")
    )
    if do_loader:
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-e",
                "modules/v2d_common",
                "-e",
                "modules/v2d_docker",
                "-e",
                "modules/v2d_task_library_loader/docker",
            ],
            dry,
            cwd=str(RECON_DIR),
        )
        _run(
            [sys.executable, "-m", "v2d.task_library_loader.docker.build"],
            dry,
            cwd=str(RECON_DIR),
        )
    if do_rg:
        _run(["bash", "workflow/run.sh", "build"], dry, cwd=str(RG_DIR))


def _filter_flags(args: argparse.Namespace) -> list[str]:
    f: list[str] = []
    if args.sequence_pattern:
        f += ["--sequence_pattern", args.sequence_pattern]
    if args.max_sequences is not None:
        f += ["--max_sequences", str(args.max_sequences)]
    return f


def _retarget_filter_flags(args: argparse.Namespace) -> list[str]:
    """Sequence filter for POST-segmentation stages (processed/support/vis/assess).

    For segmented datasets the --max-sequences/--sequence-pattern filter selects
    SOURCE sequences at load+segment; the segmented dir then holds exactly those
    sources' atomic clips. So downstream stages must process ALL of them — applying
    the same cap here would keep only the first N *clips* (e.g. 2 clips instead of
    all clips of the 2 selected sources). Non-segmented datasets keep the filter.
    """
    return [] if args.dataset in SEGMENT_DATASETS else _filter_flags(args)


def _obj_assets_host(args: argparse.Namespace) -> Path:
    """Host dir holding object meshes + generated STLs/URDFs for this dataset.

    Lives under the DATA root (``<hmd>/<ds>/object_assets``), NOT in the repo, so
    the codebase stays clean. It's bind-mounted both at ``/data/object_assets``
    (loader convention) and over the in-repo ``assets/{meshes,urdfs}/<ds>`` paths
    that MESHES_DIR/generate_rigid_urdfs hardcode — so generated STLs/URDFs land
    here and downstream stages read them here. Override with ``--obj-assets``.
    """
    if args.obj_assets:
        return Path(args.obj_assets).resolve()
    return Path(args.hmd).resolve() / args.dataset / "object_assets"


# --- LOAD (Pattern A: the loader's own host wrapper self-launches docker) -----
def stage_load(args: argparse.Namespace, ds: str, dry: bool) -> None:
    """LOAD stage: raw dataset -> {ds}_loaded parquet (MANO FK) via the loader image."""
    run_loader = (
        RECON_DIR / "modules" / "v2d_task_library_loader" / "docker" / "run_loader.py"
    )
    if not run_loader.is_file():
        sys.exit(f"loader not found at {run_loader}")
    if not args.mano_dir:
        sys.exit(
            "--mano-dir is required for the 'load' stage "
            "(dir with models/MANO_{LEFT,RIGHT}.pkl)."
        )
    out = Path(args.hmd) / ds / f"{ds}_loaded"
    cmd = [
        sys.executable,
        str(run_loader),
        "--dataset",
        ds,
        "--mano_model_dir",
        str(Path(args.mano_dir).resolve()),
        "--human_motion_data_dir",
        str(Path(args.hmd).resolve()),  # HMD ROOT (loader appends <ds>/dataset)
        "--output_dir",
        str(out),
        "--device",
        f"cuda:{args.gpu}",
        "--save",
    ]
    if ds not in RUNTIME_MESH_DATASETS:
        # object_assets datasets (taco/hot3d/arctic): meshes + URDFs live under
        # <hmd>/<ds>/object_assets. Runtime-mesh datasets read meshes from the
        # dataset itself, so we omit this (per run_loader.py).
        cmd += ["--object_assets_dir", str(_obj_assets_host(args))]
    cmd += _filter_flags(args)
    if args.dev:
        # Mount live reconstruction/modules over the loader image's baked lib so
        # local edits to <dataset>_loader.py take effect without rebuilding.
        cmd.append("--dev")
    started = time.time()
    _run(cmd, dry)
    # An exit code of 0 doesn't prove the loader wrote anything (older loader images
    # skip every failing sequence and still exit 0), so verify the output is ours.
    _require_fresh_load(args, ds, started, dry)


# --- robotic-grounding docker run helper (Pattern A for retarget stages) ------
def _rg_docker(
    args: argparse.Namespace, inner: str, *, with_kit: bool = False
) -> list[str]:
    """Build a `docker run` for the robotic-grounding image executing `inner`."""
    uid, gid = os.getuid(), os.getgid()
    cmd = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--gpus",
        f"device={args.gpu}",
        "--network",
        "host",
        "--user",
        f"{uid}:{gid}",
        "--group-add",
        "1234",
        "-v",
        f"{MONO_DIR}:{CONTAINER_MONO}",
        "-v",
        f"{Path(args.hmd).resolve()}:{CONTAINER_DATA_DIR}",
    ]
    # Object assets (meshes + URDFs) live under the DATA root, never the repo. Mount
    # at /data/object_assets (where the loader bakes object_mesh_paths) AND over the
    # in-repo assets/{meshes,urdfs}/<ds> that MESHES_DIR / generate_rigid_urdfs hardcode
    # — so the `urdf` stage writes generated STLs/URDFs to the data root, keeping the
    # codebase clean. Applies to every dataset (incl. runtime-mesh URDF generation).
    obj = _obj_assets_host(args)
    cmd += [
        "-v",
        f"{obj}:/data/object_assets",
        "-v",
        f"{obj}/meshes/{args.dataset}:{CONTAINER_ASSETS}/meshes/{args.dataset}",
        "-v",
        f"{obj}/urdfs/{args.dataset}:{CONTAINER_ASSETS}/urdfs/{args.dataset}",
    ]
    if args.dataset in RUNTIME_MESH_DATASETS:
        # Meshes ship inside the dataset, so the loader baked dataset-relative paths
        # (e.g. /data/human_motion_data/grab/dataset/tools/.../x.ply); mount them too.
        cmd += ["-v", f"{Path(args.hmd).resolve()}:/data/human_motion_data"]
    cmd += [
        "-e",
        "HOME=/tmp",
        "-e",
        "ACCEPT_EULA=Y",
        "-w",
        CONTAINER_RG,
    ]
    if with_kit:
        # Persist Kit's extension cache between dummy_agent runs (cold start
        # otherwise downloads ~780 extensions). Writable host dirs owned by us.
        kit = (
            Path(os.environ.get("HOME", "/tmp"))
            / ".cache"
            / "robotic-grounding"
            / "kit"
        )
        for sub in ("data", "cache", "logs"):
            (kit / sub).mkdir(parents=True, exist_ok=True)
            cmd += ["-v", f"{kit / sub}:/isaac-sim/kit/{sub}"]
    cmd += ["--entrypoint", "/bin/bash", args.image, "-lc", inner]
    return cmd


def _c_paths(ds: str) -> dict[str, str]:
    base = f"{CONTAINER_DATA_DIR}/{ds}"
    segmented = f"{base}/{ds}_loaded_segmented"
    return {
        "loaded": f"{base}/{ds}_loaded",
        "segmented": segmented,
        # what the retarget/support stages read: segmented for hot3d, else loaded.
        "retarget_input": (
            segmented if ds in SEGMENT_DATASETS else f"{base}/{ds}_loaded"
        ),
        "processed": f"{base}/{ds}_processed",
        "html": f"{base}/{ds}_html",
        "videos": f"{base}/{ds}_videos",
        "quality": f"{base}/{ds}_quality.csv",
        "reject": f"{base}/{ds}_rejected.txt",
    }


def _inner(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


# Clock/filesystem granularity slack when deciding whether an artifact was written
# by THIS run. Stale artifacts are hours-to-days old, so a couple of seconds of
# tolerance costs nothing.
FRESHNESS_SLACK_S = 2.0


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _newest_parquet_mtime(host_dir: Path) -> float | None:
    """Newest mtime among the parquet files under `host_dir` (None if there are none)."""
    if not host_dir.is_dir():
        return None
    mtimes = [p.stat().st_mtime for p in host_dir.rglob("*.parquet")]
    return max(mtimes) if mtimes else None


def _require_fresh_load(
    args: argparse.Namespace, ds: str, started: float, dry: bool
) -> None:
    """Abort unless the LOAD stage wrote output for THIS invocation.

    A load that matches sequences and then skips every one of them (missing raw
    data, missing object meshes/URDFs) is a failed run, but an exit code alone
    doesn't prove work happened. Without this check the later stages read whatever
    ``{ds}_loaded`` an earlier run left behind, and the pipeline reports a full
    quality pass over artifacts this invocation never produced.
    """
    if dry:
        return
    out = Path(args.hmd).resolve() / ds / f"{ds}_loaded"
    newest = _newest_parquet_mtime(out)
    if newest is not None and newest >= started - FRESHNESS_SLACK_S:
        return
    detail = (
        f"{out} contains no parquet output."
        if newest is None
        else f"{out} holds only PRE-EXISTING output, newest written {_fmt_ts(newest)} "
        "— before this stage started."
    )
    sys.exit(
        f"\nThe 'load' stage produced no output for this run.\n  {detail}\n"
        "  Every matched sequence was skipped — see the 'Skipping <id>: <reason>' lines "
        "above for the cause (usually missing raw data or object meshes/URDFs).\n"
        "  Stopping here so the later stages don't run against artifacts this run did "
        "not produce. Fix the data and re-run 'load', or pass an explicit "
        f"'--stages processed,support,vis,assess' to deliberately reuse the existing "
        f"{ds}_loaded."
    )


def _require_sequences(
    args: argparse.Namespace, ds: str, container_input: str, dry: bool
) -> None:
    """Abort with a clear message if the retarget input has no loaded sequences.

    Avoids the cryptic downstream pyarrow error ("No match for FieldRef ... sequence_id")
    when LOAD produced an empty dataset (usually: raw data missing/incomplete). When the
    input predates this run (a resumed / partial-stage invocation) say so, with the write
    date, so a reader can tell reused artifacts from freshly produced ones.
    """
    if dry:
        return
    host_in = Path(args.hmd).resolve() / ds / Path(container_input).name
    if host_in.is_dir() and (
        any(host_in.glob("sequence_id=*")) or next(host_in.rglob("*.parquet"), None)
    ):
        newest = _newest_parquet_mtime(host_in)
        started = getattr(args, "_run_started", None)
        if (
            newest is not None
            and started is not None
            and newest < started - FRESHNESS_SLACK_S
        ):
            print(
                f"  NOTE: consuming PRE-EXISTING artifacts in {host_in} "
                f"(newest written {_fmt_ts(newest)}); this run did not produce them."
            )
        return
    sys.exit(
        f"\nNo loaded sequences in {host_in} — LOAD produced nothing to retarget.\n"
        f"  Check the raw data for '{ds}' is present and complete (e.g. dexycb needs a "
        f"subject pose tarball under {ds}/dataset/; grab needs grab/sN/*.npz). Re-run the "
        f"'load' stage after the data is in place."
    )


def stage_segment(args: argparse.Namespace, ds: str, dry: bool) -> None:
    """SEGMENT stage: split {ds}_loaded into atomic interaction clips (hot3d)."""
    # Stage 1.6: split long egocentric {ds}_loaded sequences into atomic clips
    # ({ds}_loaded_segmented) that the retarget/support stages then consume.
    p = _c_paths(ds)
    inner = _inner(
        [
            "python",
            "scripts/segment_to_atomic.py",
            "--input_dir",
            p["loaded"],
            "--output_dir",
            p["segmented"],
        ]
        + _filter_flags(args)
    )
    _run(_rg_docker(args, inner), dry)


def stage_urdf(args: argparse.Namespace, ds: str, dry: bool) -> None:
    """URDF stage: generate rigid object URDFs (+ visual STLs) from the meshes."""
    inner = _inner(["python", "scripts/generate_rigid_urdfs.py", "--dataset", ds])
    _run(_rg_docker(args, inner), dry)


def stage_processed(args: argparse.Namespace, ds: str, dry: bool) -> None:
    """PROCESSED stage: IK-retarget {ds}_loaded -> {ds}_processed (robot joints)."""
    p = _c_paths(ds)
    _require_sequences(args, ds, p["retarget_input"], dry)
    inner = _inner(
        [
            "python",
            "scripts/retarget/run_retarget.py",
            "--dataset",
            ds,
            "--robot",
            args.robot,
            "--input_dir",
            p["retarget_input"],
            "--output_dir",
            p["processed"],
            "--device",
            "cuda:0",
            "--save",
        ]
        + _retarget_filter_flags(args)
    )
    _run(_rg_docker(args, inner), dry)


def stage_support(args: argparse.Namespace, ds: str, dry: bool) -> None:
    """SUPPORT stage: reconstruct support-surface USDs from object still-poses."""
    p = _c_paths(ds)
    _require_sequences(args, ds, p["retarget_input"], dry)
    inner = _inner(
        [
            "python",
            "scripts/reconstruct_support_surfaces.py",
            "--input_dir",
            p["retarget_input"],
            "--dataset",
            ds,
        ]
        + _retarget_filter_flags(args)
    )
    _run(_rg_docker(args, inner), dry)


def stage_vis(args: argparse.Namespace, ds: str, dry: bool) -> None:
    """VIS stage: render viser HTML (+ MP4) of the retargeted motion."""
    p = _c_paths(ds)
    parts = [
        "python",
        "scripts/retarget/vis_retargeted.py",
        "--input_dir",
        p["processed"],
        "--save_html",
        "--html_dir",
        p["html"],
    ]
    if not args.no_mp4:
        parts.append("--save_mp4")
    inner = _inner(parts + _retarget_filter_flags(args))
    _run(_rg_docker(args, inner), dry)


def stage_dummy(args: argparse.Namespace, ds: str, dry: bool) -> None:
    """DUMMY stage: Isaac dummy-agent playback -> per-sequence MP4 (opt-in)."""
    # Isaac playback -> per-sequence MP4. Self-contained hardened loop (same
    # lessons as run_retarget_local.sh Stage 5): detach from the TTY (</dev/null),
    # render to /tmp then `mv` (Kit's shutdown hangs under `timeout` otherwise),
    # and cap concurrency by RAM (~10GB/process). One GPU per container here, so
    # JOBS processes share device={gpu}.
    p = _c_paths(ds)
    base = f"{CONTAINER_DATA_DIR}/{ds}"
    inner = f"""
set -uo pipefail
PROCESSED={shlex.quote(p['processed'])}
VIDEO_DIR={shlex.quote(p['videos'])}
LOG_DIR={shlex.quote(base + '/dummy_agent_logs')}
mkdir -p "$LOG_DIR" "$VIDEO_DIR"
mapfile -t SEQ_DIRS < <(ls -d "$PROCESSED"/sequence_id=*/robot_name=* 2>/dev/null)
echo "dummy_agent over ${{#SEQ_DIRS[@]}} sequences (JOBS={args.jobs}, TIMEOUT={args.timeout}s)"
running=0
for seq_dir in "${{SEQ_DIRS[@]}}"; do
  (
    sid=$(basename "$(dirname "$seq_dir")"); sid=${{sid#sequence_id=}}
    [ -f "$seq_dir/.dummy_agent_ok" ] && exit 0
    tmp=/tmp/da_$sid; rm -rf "$tmp"
    timeout --signal=KILL {args.timeout} python scripts/rsl_rl/dummy_agent.py \
      --task Sharpa-V2D-v0-Play --motion_file "$seq_dir" --num_envs 1 --headless \
      --record_video --output_dir "$tmp" --video_length 300 \
      --success_marker "$seq_dir/.dummy_agent_ok" </dev/null >"$LOG_DIR/$sid.log" 2>&1 || true
    [ -f "$tmp/rl-video-step-0.mp4" ] && mv "$tmp/rl-video-step-0.mp4" "$VIDEO_DIR/$sid.mp4"
    rm -rf "$tmp"
  ) &
  running=$((running+1))
  if [ "$running" -ge {args.jobs} ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait
echo "dummy_agent done -> $VIDEO_DIR"
"""
    _run(_rg_docker(args, inner, with_kit=True), dry)


def stage_assess(args: argparse.Namespace, ds: str, dry: bool) -> None:
    """ASSESS stage: run data-quality checks -> {ds}_quality.csv (+ rejected list)."""
    p = _c_paths(ds)
    parts = ["python", "scripts/data_assessor.py", "--input_dir", p["processed"]]
    if args.filter_penetration:
        parts += ["--checks", "hand_penetration"]
    elif args.checks:
        parts += ["--checks", args.checks]
    # Disable any check whose producing stage did NOT run this invocation, else it
    # fails spuriously (missing artifact): dummy_agent_success needs the `dummy` stage's
    # sentinel. So a normal run (load..vis..assess) reports only the geometric checks
    # (hand_penetration, arctic_support_disks); adding `--stages ...,dummy` re-enables it.
    disable = {c.strip() for c in (args.disable_checks or "").split(",") if c.strip()}
    run_stages = set(getattr(args, "_resolved_stages", []) or [])
    if "dummy" not in run_stages:
        disable.add("dummy_agent_success")
    if disable:
        parts += ["--disable", ",".join(sorted(disable))]
    if args.penetration_threshold is not None:
        # --penetration-threshold is in CM; the check's param is `max_penetration` in METERS.
        parts += [
            "--set",
            f"hand_penetration.max_penetration={args.penetration_threshold / 100.0}",
        ]
    if args.reject:
        parts += ["--reject", "--output_reject", p["reject"]]
    parts += ["--output_csv", p["quality"]]
    inner = _inner(parts + _retarget_filter_flags(args))
    _run(_rg_docker(args, inner), dry)


STAGE_FNS = {
    "load": stage_load,
    "segment": stage_segment,
    "urdf": stage_urdf,
    "processed": stage_processed,
    "support": stage_support,
    "vis": stage_vis,
    "dummy": stage_dummy,
    "assess": stage_assess,
}


def parse_args() -> argparse.Namespace:
    """Parse the orchestrator's command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "dataset", nargs="?", default="taco", help="dataset name (default: taco)"
    )
    p.add_argument(
        "--hmd",
        help="HOST human-motion-data root (mounted into both images). "
        "Required for every command except --build-only.",
    )
    p.add_argument(
        "--mano-dir",
        default=os.environ.get("MANO_DIR"),
        help="HOST dir with models/MANO_{LEFT,RIGHT}.pkl (load stage).",
    )
    p.add_argument(
        "--obj-assets",
        default=None,
        help="HOST dir for object meshes/STLs/URDFs "
        "(default: <hmd>/<ds>/object_assets; kept out of the repo).",
    )
    p.add_argument("--robot", default="sharpa_wave")
    p.add_argument(
        "--image",
        default=RG_IMAGE_DEFAULT,
        help=f"robotic-grounding image (default: {RG_IMAGE_DEFAULT}).",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--stages",
        default="auto",
        help=f"comma list of {ALL_STAGES}, or 'auto' (default) for the "
        "recommended set for this dataset.",
    )
    p.add_argument(
        "--with-dummy",
        action="store_true",
        help="append the (heavy, Isaac) dummy stage.",
    )
    p.add_argument("--no-mp4", action="store_true", help="vis: skip MP4, HTML only.")
    p.add_argument(
        "--jobs", type=int, default=4, help="dummy: concurrent processes (RAM-bound)."
    )
    p.add_argument(
        "--timeout", type=int, default=600, help="dummy: per-clip timeout (s)."
    )
    # sampling
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--sequence-pattern", default=None)
    # filtering (assess stage)
    p.add_argument("--assess", action="store_true", help="run the assess stage.")
    p.add_argument(
        "--reject",
        action="store_true",
        help="assess: write rejected-IDs file (implies --assess).",
    )
    p.add_argument(
        "--filter-penetration",
        action="store_true",
        help="assess: run ONLY the hand_penetration check (implies --assess).",
    )
    p.add_argument(
        "--penetration-threshold",
        type=float,
        default=None,
        help="assess: hand_penetration depth tolerance in cm.",
    )
    p.add_argument(
        "--checks", default=None, help="assess: run only these checks (csv)."
    )
    p.add_argument(
        "--disable-checks", default=None, help="assess: drop these checks (csv)."
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="build the Docker image(s) the selected stages need before running "
        "(loader image for 'load'; robotic-grounding for retarget stages).",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="just build BOTH images and exit — no dataset, --hmd, or --mano-dir needed.",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="load: mount live reconstruction code over the loader image's "
        "baked lib (use after editing a *_loader.py without rebuilding).",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print commands, run nothing."
    )
    return p.parse_args()


def main() -> None:
    """Entry point: build images and/or run the requested stages for the dataset."""
    args = parse_args()
    if args.build_only:
        print("=== building both images (loader + robotic-grounding) ===")
        build_images(args, None, args.dry_run)
        print("\n=== done: images built ===")
        return
    if not args.hmd:
        sys.exit(
            "--hmd is required (the HOST data root). Use --build-only to just build images."
        )
    ds = args.dataset
    if args.stages == "auto":
        stages = list(RECOMMENDED_STAGES.get(ds, DEFAULT_STAGES))
        note = RECOMMEND_NOTES.get(ds)
        print(
            f"  recommended stages for '{ds}': {','.join(stages)}"
            + (f"\n    ({note})" if note else "")
            + "\n    (override with --stages a,b,c)"
        )
    else:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if (
        args.reject
        or args.filter_penetration
        or args.checks
        or args.penetration_threshold is not None
    ):
        args.assess = True
    if args.assess and "assess" not in stages:
        stages.append("assess")
    if args.with_dummy and "dummy" not in stages:
        stages.insert(
            stages.index("vis") + 1 if "vis" in stages else len(stages), "dummy"
        )
    # hot3d (and any SEGMENT_DATASETS) must segment {ds}_loaded before retargeting.
    if (
        ds in SEGMENT_DATASETS
        and "segment" not in stages
        and any(s in stages for s in ("processed", "support"))
    ):
        seg_exists = (Path(args.hmd).resolve() / ds / f"{ds}_loaded_segmented").is_dir()
        if "load" in stages or not seg_exists:
            pos = stages.index("load") + 1 if "load" in stages else 0
            stages.insert(pos, "segment")
            print(
                f"  (auto-inserted 'segment' stage — {ds} needs atomic-clip segmentation)"
            )
        else:
            print(f"  (reusing existing {ds}_loaded_segmented; skipping 'segment')")
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        sys.exit(f"unknown stage(s): {bad}; valid: {ALL_STAGES}")

    # Object assets live under the data root; pre-create the per-dataset dirs as
    # this user so Docker's bind mounts don't auto-create them root-owned.
    obj = _obj_assets_host(args)
    if not args.dry_run and any(
        s in stages for s in ("load", "urdf", "processed", "support", "vis")
    ):
        # data-root object_assets dirs (mounted into the container); created as this
        # user so Docker doesn't auto-create them root-owned. For runtime-mesh datasets
        # these hold only the generated STLs/URDFs (source meshes stay in the dataset).
        (obj / "meshes" / ds).mkdir(parents=True, exist_ok=True)
        (obj / "urdfs" / ds).mkdir(parents=True, exist_ok=True)

    print(f"=== pipeline for '{ds}' ===")
    print(f"  stages : {stages}")
    print(f"  hmd    : {args.hmd}")
    print(f"  objects: {obj}")
    print(f"  image  : {args.image}  gpu={args.gpu}  robot={args.robot}")
    if args.build:
        print("\n########## building image(s) ##########")
        build_images(args, stages, args.dry_run)
    args._resolved_stages = stages  # so stage_assess knows which stages ran
    args._run_started = time.time()  # so stages can flag pre-existing artifacts
    for s in stages:
        print(f"\n########## stage: {s} ##########")
        STAGE_FNS[s](args, ds, args.dry_run)
    print(f"\n=== done: {stages} ===")


if __name__ == "__main__":
    main()
