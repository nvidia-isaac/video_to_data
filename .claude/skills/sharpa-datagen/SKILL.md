---
name: sharpa-datagen
description: Guide for generating robot-behaviour datasets from trained Sharpa (floating-hand) V2D policies — rolling out checkpoints in Isaac Lab with cameras and exporting to LeRobot. Use this skill whenever the user wants to generate/record a dataset from a trained policy, run `record_dataset.py` / the `Sharpa-V2D-Record-v0` task, batch-generate data across a dataset's checkpoints, visualize recorded rollouts, prepare local retargeted motion + support surfaces for a record run, or debug why recorded episodes fail (objects dropping, immediate divergence, 0% completion, missing assets, file locks). Also trigger on "data generation", "datagen", "record rollouts", "LeRobot export", "policy rollout dataset", or mentions of front_cam/ego_cam/VOC/completion_ratio.
---

# Sharpa V2D Data Generation

Roll out a trained per-sequence Sharpa policy in Isaac Lab, record camera + state/action
observations, and export a **LeRobot v3** dataset. All work runs **inside the container**,
from `/workspace/video_to_data/robotic_grounding`.

> This skill encodes lessons that are easy to get wrong. Read **Prerequisites** and
> **Data requirements** before running anything — most failures are missing data, not code.

## 0. TL;DR happy path (one sequence)

```bash
# inside the container, cwd = /workspace/video_to_data/robotic_grounding
python scripts/rsl_rl/record_dataset.py --headless \
  --task Sharpa-V2D-Record-v0 \
  --checkpoint ../Datagen_Checkpoints/floating_sharpa_checkpoints/<ds>/<seq>/model_<iter>.pt \
  --motion_file <ds>/<ds>_processed/<seq>/sharpa_wave \
  --num_envs 10 --num_episodes 20 \
  --voc_scale 0.0 --voc_decay_steps 20 \
  --output_file datasets/<run_name>/<seq>.hdf5     # -> LeRobot dir datasets/<run_name>/<seq>/
```
For all checkpoints of a dataset, use the batch driver (section 7).

## 1. Prerequisites (check FIRST)

- **NVIDIA driver 580.x.** Isaac Sim 5.1.0 in the image is validated on driver 580 or older; **595
  segfaults the RTX renderer** at startup. `nvidia-smi` must read 580.x or older.
- **Container running:** `./workflow/run.sh start latest 0`. The final interactive `exec`
  fails without a TTY but the container stays up. Drive it with:
  `docker exec -w /workspace/video_to_data/robotic_grounding robotic-grounding-latest-gpu0 bash -lc '...'`.
  `python` in the container = `isaaclab.sh -p`. **Avoid heredocs** through it — write a
  script file to the mounted repo and run that instead.
- **Motion and object assets:** prepare them locally using
  `robotic_grounding/docs/SETUP.md` and the per-dataset setup guide. Build the local
  container with `./workflow/run.sh build`; a shared registry is optional. For remote
  images set `V2D_IMAGE_REGISTRY` and follow `robotic_grounding/workflow/README.md`.
- **Optional storage downloads:** use your storage provider's client and credentials.
  The batch driver can use an S3-compatible endpoint configured with `CSS_ENDPOINT_URL`,
  `CSS_ACCESS_KEY`, `CSS_SECRET_KEY`, and optional `CSS_REGION`. These are legacy variable
  names, not access to a repository-provided service. No hosted bucket or credentials
  are supplied. Pass the variables explicitly to the container only if downloads are needed.
- **Checkpoints** (per-sequence policies):
  `Datagen_Checkpoints/floating_sharpa_checkpoints/<ds>/<seq>/{metadata.json, model_<iter>.pt}`.
  `metadata.json` has the reference eval metrics (see section 6). One policy per sequence.

## 2. Data requirements per record run (the #1 failure class)

A record run needs THREE things present locally for the sequence. Missing any → failures:

1. **Processed motion parquet** — 62 columns, **must include `mano_{left,right}_link_contact_normals`**.
   `source/.../assets/human_motion_data/<ds>/<ds>_processed/sequence_id=<seq>/robot_name=sharpa_wave/*.parquet`
   - Generate the processed partition with the retarget pipeline in
     `robotic_grounding/docs/SETUP.md`, or copy it from storage you control.
   - ⚠️ **Older exports may be MISSING `link_contact_normals`** → crash
     `must be real number, not NoneType` in `hand_object_commands.py:_init_contact_data`.
     Verify with `pyarrow.parquet.read_schema(...).names`.

2. **Support surface** — `reconstructed_stage/<seq>_support.usda`.
   `source/.../assets/human_motion_data/<ds>/reconstructed_stage/<seq>_support.usda`
   - Generate it with `python scripts/reconstruct_support_surfaces.py --dataset <ds>
     --sequence_id <seq>`, or copy the matching file from your `reconstructed_stage/` output.
   - ⚠️ **Missing support surface → the object free-falls → `object_away_from_trajectory`
     terminates at ~step 5 → 0% completion.** This looks like a policy failure but isn't.

3. **Object meshes + URDFs** — `assets/{meshes,urdfs}/<ds>/`.
   - Download meshes from the dataset's original source following its setup guide,
     then run `python scripts/generate_rigid_urdfs.py --dataset <ds>`.
     A missing `NNN_rigid.urdf` causes `FileNotFoundError: Missing assets`.

### Optional batch storage layout

`batch_taco_datagen.py` reads the existing local processed motion and support surfaces
first. If either is missing, it downloads from the configured S3-compatible endpoint.
Its bucket (`datasets`) and prefixes (`v2d/human_motion_data/taco/taco_processed` and
`v2d/human_motion_data/taco/reconstructed_stage`) are constants in the script; adapt
these to your storage before using remote downloads. The client uses S3 v4 signatures,
path-style addressing, and `get_object`.

## 3. Running a record (`record_dataset.py`)

- Task **`Sharpa-V2D-Record-v0`** adds two `TiledCamera`s (`front_cam` 3rd-person +
  `ego_cam` egocentric, auto-aimed at the scene's mean object position), grey cubicle walls
  (so cameras don't see neighbouring tiled envs), and a `RecorderManager`.
- Output is **LeRobot v3** by default (a directory next to the `.hdf5` path); pass
  `--output_format hdf5` for raw HDF5.
- Key flags: `--num_envs` (parallel envs), `--num_episodes` (records **≥** this; parallel
  envs overshoot), `--export_mode all|succeeded`, `--voc_scale`/`--voc_decay_steps`
  (section 4), `--domain_randomization` (visual DR, section 4b), `--use_primitive_urdfs`
  (**REQUIRED for taco checkpoints — see section 3b**), `--camera_width`/`--camera_height`
  (override the 256² default; e.g. 512 for high-res viz), `--replay_motion` (section 5),
  `--debug_world_axes` (RGB=XYZ axes at env origin to tune camera poses; **capped at 4 episodes**).
- `--motion_file` shorthand `<ds>/<ds>_processed/<seq>/sharpa_wave` resolves to the
  `sequence_id=.../robot_name=...` partition.

## 3b. Primitive URDFs — REQUIRED for taco checkpoints (do not skip)

The taco floating-Sharpa checkpoints were **TRAINED with primitive (capsule/cylinder) robot
URDFs** (`left/right_sharpa_wave_primitive.urdf`). You **must** record with
`--use_primitive_urdfs` — recording on the full mesh-collision URDF changes the finger contact
model, grasps slip, the object drifts off its path, `object_away_from_trajectory` fires, and
completion craters. Verified: flipping this took the taco batch from **53% → ~80%**
(individual tasks 0% → 85–100%).

- ⚠️ The checkpoint's frozen `params/env_cfg.json` records the **full** URDF path
  (`left_sharpa_wave.urdf`) — this is **misleading**; it does not reflect what training used.
  Trust the empirical result, not that file.
- `batch_taco_datagen.py` defaults `USE_PRIMITIVE_URDFS=1` and passes the flag automatically.

## 3c. Env count & the recorder (sizing a run)

- Per-task wall-clock is **startup-bound** (~3–4 min Isaac boot + data preparation, fixed regardless of
  env count). Env count only sets rollout waves = `ceil(num_episodes / num_envs)`, so **more
  envs = fewer waves = faster, up to the RAM limit** — there is no interior sweet spot.
- **64 envs is the single-container RAM-safe max** (~29–33 GB host RAM). **256 envs thrashes**
  the recorder buffers (>62 GB, swap off) → OOM.
- The `RecorderManager` **buffers every episode in RAM until export**, so peak RAM scales with
  `num_episodes × horizon`, not just envs. **Long-horizon sequences** (e.g. ~500-step) can
  **stall `add_to_episodes` at 64 envs** (stuck at "recorded 1/N" for an hour) — for those,
  **drop to ~16 envs**.
- To speed a full run, **shard** (`launch_sharded_datagen.sh N`): N containers overlap the
  fixed startups (~N× wall-clock). Keep per-shard `num_envs` modest so N shards fit host RAM.

## 4. VOC (virtual object control) semantics — READ THIS

`--voc_scale` is the value VOC decays **TO** (the floor/target), **NOT the start**. Every
episode the per-env factor **resets to 1.0**; in `step` mode it holds 1.0 for
`--voc_decay_steps` steps, then is set to `--voc_scale`. Therefore:

- `--voc_scale 1.0 --voc_decay_steps 20` → **VOC stays 1.0 the whole episode** (object fully
  assisted the entire time; inflates completion). "Decays to 1.0" = no-op.
- `--voc_scale 0.0 --voc_decay_steps 20` → **assist for 20 steps, then OFF** (pure policy
  after). This is usually what you want for a behaviour dataset.
- `--voc_scale 0.0 --voc_decay_steps 0` → VOC off immediately (pure deterministic).
- During the first `--voc_decay_steps` steps the reference trajectory is **frozen at frame 0**
  (a settle phase); it advances afterward. The completion metric subtracts this warmup.
- VOC only controls the **object** — it never helps the **hands**. If a sequence fails by
  *hand* divergence (hands drift off the reference while the object stays put), VOC won't fix it.

## 4b. Domain randomization (`--domain_randomization`)

Per-episode visual DR via `SceneMaterialRandomizer` (`scripts/rsl_rl/domain_randomization.py`):
object material colour/roughness/metallic, support-surface texture, and scene lighting.

- ⚠️ **DR is visual-only** — it randomizes the *rendered camera pixels*, **not** the policy's
  state observations (the 722-dim obs vector is state-based). So **DR does not change completion
  rate**; its value is image diversity for downstream (VLA/visuomotor) training.
- Batch knob: `DOMAIN_RANDOMIZATION=1` (default 0 in `batch_taco_datagen.py`).

## 5. Replay mode (validate trajectories without a policy)

`--replay_motion` (no checkpoint needed) kinematically teleports the hands along the
reference wrist+finger trajectory while the object rides VOC. Use it to confirm the
trajectories/cameras are intact independent of the policy:
`... --replay_motion --voc_scale 1.0 ...` (shared helpers in
`tasks/scene_utils/replay_kinematics.py`).

## 6. Metrics + the success-flag caveat

- Each run prints and stores **from-frame-0 completion**:
  `completion_ratio = clamp((ep_len - warmup)/(horizon - warmup))`,
  `full_completion = ratio >= 0.99` (warmup = `--voc_decay_steps`, horizon =
  `command.retargeted_horizon`). Written as per-episode attrs (HDF5) **and columns in the
  LeRobot episodes parquet** (`meta/episodes/.../*.parquet`), plus a printed aggregate
  `completion_ratio_mean` / `full_completion_pct`.
- Checkpoint `metadata.json` reference metrics: `completion_ratio_mean` (from frame 0),
  `completion_ratio_mean_random` (from a random start frame), `full_completion_pct`
  (% of frame-0 episodes completing the full trajectory).
- ⚠️ **The recorded `success` flag is UNWIRED** — the env has no `success` termination term,
  so it's always `False`. `--export_mode succeeded` would export **0 episodes**. Use
  `--export_mode all` and filter on `completion_ratio` / `full_completion`.
- ⚠️ **Policies are per-sequence OVERFIT** — a checkpoint only tracks the exact sequence it was
  trained on; it fails (≈0%) even on a *sibling* sequence of the same task family. Useful when
  debugging: swapping a checkpoint onto another sequence's env is expected to fail, so it can't
  isolate "bad checkpoint vs bad env". A matched checkpoint that still fails is the real anomaly.

## 7. Batch generation

`scripts/batch_taco_datagen.py` runs taco checkpoints from its configured checkpoint
root: checks local data (optionally downloads missing files from your configured storage),
runs `record_dataset` **serially**, kills zombie kit procs between runs, scores
from the LeRobot episodes parquet, and writes `SUMMARY.md` (per-task + total
full_completion_pct vs metadata). Env-overridable constants: `NUM_ENVS`, `NUM_EPISODES`,
`VOC_SCALE`, `VOC_DECAY_STEPS`, **`USE_PRIMITIVE_URDFS` (default 1)**,
**`DOMAIN_RANDOMIZATION` (default 0)**, `RUN_TIMEOUT`. Smoke-test with env `BATCH_LIMIT=N` or
`BATCH_SEQS=a,b`. `SCORE_ONLY=1` re-scans `OUT_ROOT` and rewrites `SUMMARY.md` over all
exported tasks (use after adding/re-recording a task; pass the real `NUM_ENVS/NUM_EPISODES/...`
so the summary's config line is accurate). It's resumable (skips dirs already exported) —
**clear the output dir when changing config**, else existing tasks are skipped.

- **Sharded runs:** `scripts/launch_sharded_datagen.sh N` launches N containers (disjoint seq
  subsets, isolated Kit caches) then merges via `SCORE_ONLY`. ~N× faster (overlaps startups).
- ⚠️ **Completion-log gotcha:** `record_dataset` prints `Done. Exported episodes -> .hdf5`
  **before** the HDF5→LeRobot conversion runs. **True** completion is the `LeRobot dataset
  ready` line / process exit. When waiting on a run, poll for process exit
  (`ps | grep -v grep`, to avoid the `pgrep`/self-match trap), not that log line.

## 8. Visualization

`python scripts/visualize_dataset.py --dataset <file.hdf5|lerobot_dir> --output_dir <dir>
--data_types rgb depth seg` → one MP4 per episode, tiled (rows = modality, cols = camera).
Multi-camera datasets tile automatically. `--debug_world_axes` records help frame cameras.
For higher-res renders, re-record the sequence with `--camera_width/--camera_height` (e.g. 512).

**Real-ego ↔ sim comparison videos:** composite the human ego video (center-cropped square)
beside the recorded sim episodes (ego_cam|front_cam tiles, stacked) with a small cv2/ffmpeg
script — handy for eyeballing sim-vs-real per task. (Compositing helpers are kept as local
tooling, not shipped in this repo.)

### Egocentric human videos (`datasets/Egocentric_RGB_Videos/`)
Real human demos are keyed `(<action>, <tool>, <target>)/<date>_<num>/color.mp4`, which maps to
seq id `taco_<action>__<tool>__<target>_<date>_<num>` (spaces→underscores). Build the mapping
into `seq_to_ego_video.json`, then per task copy the mp4 to `<task>/egocentric/color.mp4` and
record `egocentric_video_path` (relative to `Egocentric_RGB_Videos`) in `meta/info.json`. The
ego videos are **not** frame-synced to the sim rollouts — they're linked at the task level.

## 9. Operational gotchas

- **Run record jobs serially.** Parallel Isaac record runs in one container collide on the
  recorder HDF5 lock → `BlockingIOError: unable to lock file`.
- **Kill zombie kit processes between runs.** A record run that errors/is killed often leaves
  its `_isaac_sim/.../python3` kit process spinning (~120% CPU) holding the file lock + GPU:
  `docker exec <C> bash -lc "pkill -9 -f record_dataset.py; pkill -9 -f 'isaaclab.sh -p'"`.
  (Don't run this `pkill` from inside a bash whose own command string contains
  `record_dataset.py` — it kills itself.)

## 10. Troubleshooting (symptom → cause)

| Symptom | Cause / fix |
|---|---|
| RTX segfault at startup | Host driver not 580.x (595 too new). Swap to `nvidia-driver-580-open`. |
| `must be real number, not NoneType` @ `_init_contact_data` | Parquet missing `link_contact_normals` (stale processed export). Regenerate or obtain the updated processed parquet. |
| Objects drop, episodes die ~step 5, 0% completion | Missing `reconstructed_stage/<seq>_support.usda`. Pull it. |
| Low completion / grasp slips / object drifts off path (esp. batch-wide) | Recorded on **full** URDF but the ckpt was trained **primitive**. Add `--use_primitive_urdfs` (section 3b). |
| Stuck at "recorded 1/N" for a long time (long-horizon seq) | Recorder saturating at too many envs. Drop `--num_envs` to ~16 (section 3c). |
| Waited on a run but it wasn't actually done | `Done. Exported -> .hdf5` precedes the LeRobot conversion; wait for `LeRobot dataset ready` / process exit (section 7). |
| `FileNotFoundError: Missing assets ... NNN_rigid.urdf` | Run `generate_rigid_urdfs.py --dataset <ds>`. |
| `BlockingIOError: unable to lock file` | Zombie kit process holding the lock. `pkill` (section 9). |
| Hands drift away but objects stable; 0% | Hand-divergence policy failure (per-sequence). VOC can't help hands; not a data bug. |
| `403 access denied` from `osmo` | Verify authentication and pool/storage permissions with your OSMO deployment administrator; see `robotic_grounding/workflow/README.md`. |
| Workflow image pull denied | Build locally, or configure `V2D_IMAGE_REGISTRY` and log in to a registry your OSMO workers can access. For the public Isaac Lab base image, follow the NGC login guidance in `robotic_grounding/docs/SETUP.md`. |
