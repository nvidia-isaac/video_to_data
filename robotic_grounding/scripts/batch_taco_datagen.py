# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch taco data-gen test: run every taco checkpoint's policy, export LeRobot, report completion.

Runs INSIDE the container (has boto3/pyarrow/h5py). For each taco checkpoint sequence:
  1. ensure CSS data is local (processed parquet + reconstructed_stage support usda),
  2. run scripts/rsl_rl/record_dataset.py (NUM_ENVS envs, NUM_EPISODES eps) -> LeRobot dir,
  3. read the per-episode `full_completion` / `completion_ratio` from the LeRobot episodes
     parquet, and accumulate.
Serial (parallel Isaac runs collide on the recorder file lock); kills lingering kit
processes between runs. Resumable: completed tasks (LeRobot meta present) are re-scored,
not re-run. Prints a progress line per task and a final per-task + total summary that is
also compared to each checkpoint's metadata.json `full_completion_pct`.

Requires CSS_* env vars. Launch (from repo root, in container):
  python scripts/batch_taco_datagen.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time

import boto3
import pyarrow.parquet as pq
from botocore.config import Config

REPO = "/workspace/video_to_data/robotic_grounding"
CKPT_ROOT = (
    "/workspace/video_to_data/Datagen_Checkpoints/floating_sharpa_checkpoints/taco"
)
ASSETS = (
    f"{REPO}/source/robotic_grounding/robotic_grounding/assets/human_motion_data/taco"
)
OUT_ROOT = os.environ.get(
    "OUT_ROOT", "datasets/Taco_Datagen_Initial"
)  # relative to REPO (cwd)
# Sharding: each shard processes seqs[SHARD_INDEX::SHARD_COUNT]. Shards write disjoint seq
# subdirs under a shared OUT_ROOT + a per-shard SUMMARY. Run one shard per container (each
# with its own Kit cache) via scripts/launch_sharded_datagen.sh; then SCORE_ONLY=1 merges.
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "1"))
SCORE_ONLY = bool(
    int(os.environ.get("SCORE_ONLY", "0"))
)  # skip running; just score OUT_ROOT
NUM_ENVS = int(os.environ.get("NUM_ENVS", "10"))
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "20"))
RUN_TIMEOUT = int(
    os.environ.get("RUN_TIMEOUT", "1500")
)  # s per record run; hung runs killed
# VOC assist for the first VOC_DECAY_STEPS steps (object settled at frame 0), then VOC turns
# OFF (scale -> VOC_SCALE=0.0) and the policy is on its own. Same value is the
# completion-ratio warmup. NOTE: VOC_SCALE is the value VOC decays *to* (the floor), NOT the
# start — the per-env factor always resets to 1.0. So 0.0 = "assist then off"; 1.0 = "always on".
VOC_SCALE = 0.0
VOC_DECAY_STEPS = 20
# Termination terms to disable for this run (empty = keep all). Disabling
# hand_wrist_away_from_trajectory lets episodes survive transient early hand divergence
# (the policy often recovers); object_away + time_out still end the episode.
DISABLE_TERMINATIONS = ["hand_wrist_away_from_trajectory"]
# The taco floating-Sharpa checkpoints were TRAINED with primitive (capsule/cylinder) robot
# URDFs, so we MUST record with them too — recording on the full mesh-collision URDFs changes
# the finger contact model, grasps slip, and completion craters (verified: env032 0%->85%,
# env062 0%->100% just by flipping this). Default ON. See memory: v2d-primitive-urdf-requirement.
USE_PRIMITIVE_URDFS = bool(int(os.environ.get("USE_PRIMITIVE_URDFS", "1")))
# Visual domain randomization (materials/lighting/support textures, per-episode). OFF by
# default so completion-measurement runs stay clean; turn ON for the real training dataset.
DOMAIN_RANDOMIZATION = bool(int(os.environ.get("DOMAIN_RANDOMIZATION", "0")))

BUCKET = "datasets"
PROC_PREFIX = "v2d/human_motion_data/taco/taco_processed"
STAGE_PREFIX = "v2d/human_motion_data/taco/reconstructed_stage"


def log(msg: str) -> None:
    print(msg, flush=True)


_S3_CLIENT = None


def _s3():
    """Lazily create the CSS S3 client (so SCORE_ONLY / merge works without CSS env)."""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=os.environ["CSS_ENDPOINT_URL"],
            aws_access_key_id=os.environ["CSS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["CSS_SECRET_KEY"],
            region_name=os.environ.get("CSS_REGION", "us-east-1"),
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
    return _S3_CLIENT


def ensure_data(seq: str) -> tuple[bool, str]:
    """Make the processed parquet + support usda for `seq` present locally (pull if missing)."""
    pdir = f"{ASSETS}/taco_processed/sequence_id={seq}/robot_name=sharpa_wave"
    if not glob.glob(f"{pdir}/*.parquet"):
        os.makedirs(pdir, exist_ok=True)
        r = _s3().list_objects_v2(
            Bucket=BUCKET,
            Prefix=f"{PROC_PREFIX}/sequence_id={seq}/robot_name=sharpa_wave/",
        )
        keys = [
            o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".parquet")
        ]
        if not keys:
            return False, "no CSS processed parquet"
        obj = _s3().get_object(Bucket=BUCKET, Key=keys[0])
        with open(f"{pdir}/{os.path.basename(keys[0])}", "wb") as fh:
            fh.write(obj["Body"].read())
    sdir = f"{ASSETS}/reconstructed_stage"
    os.makedirs(sdir, exist_ok=True)
    spath = f"{sdir}/{seq}_support.usda"
    if not os.path.exists(spath):
        try:
            obj = _s3().get_object(
                Bucket=BUCKET, Key=f"{STAGE_PREFIX}/{seq}_support.usda"
            )
            with open(spath, "wb") as fh:
                fh.write(obj["Body"].read())
        except Exception as e:  # noqa: BLE001
            return False, f"no support usda ({type(e).__name__})"
    return True, "ok"


def lerobot_dir(seq: str) -> str:
    return f"{OUT_ROOT}/{seq}"


def score_lerobot(seq: str) -> tuple[float, float, int, int] | None:
    """Return (full_completion_pct, completion_ratio_mean, n_full, n_eps) from the LeRobot dir."""
    eps = glob.glob(f"{lerobot_dir(seq)}/meta/episodes/**/*.parquet", recursive=True)
    if not eps:
        return None
    t = pq.read_table(eps[0])
    cols = t.column_names
    if "full_completion" not in cols:
        return None
    full = t.column("full_completion").to_pylist()
    ratio = (
        t.column("completion_ratio").to_pylist() if "completion_ratio" in cols else []
    )
    n = len(full)
    n_full = sum(1 for x in full if x)
    mean_r = (
        (sum(r for r in ratio if r is not None) / len(ratio)) if ratio else float("nan")
    )
    return 100.0 * n_full / n, mean_r, n_full, n


def run_record(seq: str) -> None:
    ckpts = sorted(glob.glob(f"{CKPT_ROOT}/{seq}/model_*.pt"))
    if not ckpts:
        raise RuntimeError("no checkpoint .pt")
    os.makedirs(OUT_ROOT, exist_ok=True)
    cmd = [
        "python",
        "scripts/rsl_rl/record_dataset.py",
        "--headless",
        "--task",
        "Sharpa-V2D-Record-v0",
        "--checkpoint",
        ckpts[-1],
        "--motion_file",
        f"taco/taco_processed/{seq}/sharpa_wave",
        "--num_envs",
        str(NUM_ENVS),
        "--num_episodes",
        str(NUM_EPISODES),
        "--voc_scale",
        str(VOC_SCALE),
        "--voc_decay_steps",
        str(VOC_DECAY_STEPS),
        "--output_file",
        f"{OUT_ROOT}/{seq}.hdf5",
        "--task_name",
        seq,
    ]
    if DISABLE_TERMINATIONS:
        cmd += ["--disable_terminations", *DISABLE_TERMINATIONS]
    if USE_PRIMITIVE_URDFS:
        cmd += ["--use_primitive_urdfs"]
    if DOMAIN_RANDOMIZATION:
        cmd += ["--domain_randomization"]
    with open(f"{OUT_ROOT}/{seq}.log", "w") as lf:
        try:
            subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT, cwd=REPO
            )
        except subprocess.TimeoutExpired:
            lf.write("\n[batch] RUN_TIMEOUT exceeded — killed.\n")
    # Defensively clear any lingering kit process so the next run doesn't hit a file lock.
    subprocess.run(
        ["bash", "-lc", "pkill -9 -f record_dataset.py; sleep 2"], check=False
    )


def metadata_full_pct(seq: str) -> float | None:
    p = f"{CKPT_ROOT}/{seq}/metadata.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("full_completion_pct")


def main() -> None:
    seqs = sorted(
        os.path.basename(d) for d in glob.glob(f"{CKPT_ROOT}/*") if os.path.isdir(d)
    )
    # Optional smoke-test knobs: BATCH_SEQS=comma,list (exact) or BATCH_LIMIT=N (first N).
    if os.environ.get("BATCH_SEQS"):
        want = set(os.environ["BATCH_SEQS"].split(","))
        seqs = [s for s in seqs if s in want]
    elif os.environ.get("BATCH_LIMIT"):
        seqs = seqs[: int(os.environ["BATCH_LIMIT"])]
    if SHARD_COUNT > 1:
        seqs = seqs[SHARD_INDEX::SHARD_COUNT]
        log(
            f"[batch] shard {SHARD_INDEX}/{SHARD_COUNT}: {len(seqs)} of the checkpoints"
        )
    log(
        f"[batch] {len(seqs)} taco checkpoints | {NUM_ENVS} envs x {NUM_EPISODES} eps each -> {OUT_ROOT}"
    )
    results: list[tuple] = (
        []
    )  # (seq, full_pct, mean_ratio, n_full, n_eps, meta_full, status)
    t0 = time.time()
    for i, seq in enumerate(seqs, 1):
        elapsed = (time.time() - t0) / 60.0
        prefix = f"[{i}/{len(seqs)}] {elapsed:5.1f}min {seq}"
        meta = metadata_full_pct(seq)
        # Resume: if already exported, just re-score.
        scored = score_lerobot(seq)
        if scored is None:
            ok, msg = ensure_data(seq)
            if not ok:
                log(f"{prefix} -> SKIP (data: {msg})")
                results.append((seq, None, None, 0, 0, meta, f"nodata:{msg}"))
                continue
            try:
                run_record(seq)
            except Exception as e:  # noqa: BLE001
                log(f"{prefix} -> FAIL ({e})")
                results.append((seq, None, None, 0, 0, meta, f"runerr:{e}"))
                continue
            scored = score_lerobot(seq)
        if scored is None:
            log(f"{prefix} -> FAIL (no LeRobot episodes/completion)")
            results.append((seq, None, None, 0, 0, meta, "noscore"))
            continue
        full_pct, mean_r, n_full, n_eps = scored
        meta_str = f"{meta:.1f}" if meta is not None else "?"
        log(
            f"{prefix} -> full={full_pct:5.1f}% (meta {meta_str}) mean_ratio={mean_r:.3f} N={n_eps}"
        )
        results.append((seq, full_pct, mean_r, n_full, n_eps, meta, "ok"))

    summary_name = f"SUMMARY_shard{SHARD_INDEX}.md" if SHARD_COUNT > 1 else "SUMMARY.md"
    _write_summary(results, f"{OUT_ROOT}/{summary_name}")
    log(
        f"\n[batch] DONE in {(time.time() - t0)/60:.1f} min. Summary -> {OUT_ROOT}/{summary_name}"
    )


def _write_summary(results, out_path: str) -> None:
    ok = [r for r in results if r[6] == "ok"]
    tot_full = sum(r[3] for r in ok)
    tot_eps = sum(r[4] for r in ok)
    total_full_pct = 100.0 * tot_full / tot_eps if tot_eps else float("nan")
    mean_task_full = sum(r[1] for r in ok) / len(ok) if ok else float("nan")

    lines = [f"# {OUT_ROOT} — batch completion report", ""]
    lines.append(
        f"- tasks: {len(results)} | ok: {len(ok)} | failed/skipped: {len(results) - len(ok)}"
    )
    lines.append(
        f"- config: {NUM_ENVS} envs x {NUM_EPISODES} eps/task | VOC_SCALE={VOC_SCALE} "
        f"decay={VOC_DECAY_STEPS} | disabled_terms={DISABLE_TERMINATIONS} | "
        f"primitive_urdfs={USE_PRIMITIVE_URDFS} | domain_randomization={DOMAIN_RANDOMIZATION}"
    )
    lines.append(
        f"- **TOTAL full_completion_pct (episode-weighted): {total_full_pct:.2f}%** "
        f"({tot_full}/{tot_eps} eps)"
    )
    lines.append(
        f"- mean per-task full_completion_pct (unweighted): {mean_task_full:.2f}%"
    )
    metas = [r[5] for r in ok if r[5] is not None]
    if metas:
        lines.append(
            f"- checkpoint metadata mean full_completion_pct (same tasks): "
            f"{sum(metas)/len(metas):.2f}%"
        )
    lines += [
        "",
        "| task | ours full% | meta full% | mean_ratio | N | status |",
        "|---|---|---|---|---|---|",
    ]
    for seq, full_pct, mean_r, _nf, n_eps, meta, status in results:
        fp = f"{full_pct:.1f}" if full_pct is not None else "-"
        mp = f"{meta:.1f}" if meta is not None else "-"
        mr = f"{mean_r:.3f}" if mean_r is not None else "-"
        lines.append(f"| {seq} | {fp} | {mp} | {mr} | {n_eps} | {status} |")
    report = "\n".join(lines)
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(report + "\n")
    log("\n" + report)


def _score_only() -> None:
    """Skip running; score every checkpoint's already-exported LeRobot dir -> combined SUMMARY."""
    seqs = sorted(
        os.path.basename(d) for d in glob.glob(f"{CKPT_ROOT}/*") if os.path.isdir(d)
    )
    results = []
    for seq in seqs:
        meta = metadata_full_pct(seq)
        scored = score_lerobot(seq)
        if scored is None:
            continue
        full_pct, mean_r, n_full, n_eps = scored
        results.append((seq, full_pct, mean_r, n_full, n_eps, meta, "ok"))
    log(f"[batch] SCORE_ONLY: scored {len(results)} exported tasks under {OUT_ROOT}")
    _write_summary(results, f"{OUT_ROOT}/SUMMARY.md")


if __name__ == "__main__":
    if SCORE_ONLY:
        _score_only()
    else:
        main()
