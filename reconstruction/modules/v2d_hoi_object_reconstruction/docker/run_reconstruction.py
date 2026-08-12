# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end object reconstruction pipeline (HOST-SIDE orchestrator).

Two modes:
  bundlesdf (default) – Two-stage scan (stationary → rotated → stationary) → textured mesh
  sam3d               – Select representative frames → SAM3D per-frame → SRT scale estimation

── BundleSDF steps ──────────────────────────────────────────────────────────────────
  1.  prepare_FP_folder           – copy images, write calibration + video
  2.  CuSFM                       – → job_dir/sfm/keyframes/frames_meta.json with poses
  2b. CuSFM scan quality          – validate two-loop pose trajectory before auto split
  2c. Stage-1 auto-detect         – detect Stage-1 end from CuSFM trajectory slope
  3.  Grounding DINO               – detect object bbox from text prompt
  4a. Depth (parallel workers)    – FoundationStereo depth for ALL frames
  4b. Mask                        – SAM2 masks + optional cleanup for ALL frames
  5.  Stage-1 recon setup         – filter Stage-1 SfM keyframes + depth symlinks
  6.  Stage-1 NeRF                – reconstruct Stage-1 mesh (bottom missing)
  7.  Center mesh                 – shift mesh centroid to origin
  8.  FoundationPose tracking     – track all frames with Stage-1 mesh
  9.  World poses                 – compute T_world_from_obj + auto-detect stages
  10. Merged recon setup          – align Stage-2 keyframes into Stage-1 obj frame
  11. Full NeRF                   – reconstruct complete mesh + output.glb from both stages
  12. FP tracking (final)         – track all frames with final textured mesh
  13. FP render (final)           – render overlay video with final textured mesh

── SAM3D steps ──────────────────────────────────────────────────────────────────────
  1.  prepare_FP_folder           – copy images, write calibration + video
  2.  CuSFM                       – camera poses for frame selection + SRT scale
  2b. CuSFM scan quality          – validate two-loop pose trajectory before auto split
  2c. Stage-1 auto-detect         – detect Stage-1 end (used to exclude transition frames)
  3.  Grounding DINO               – detect object bbox from text prompt
  4b. Mask                        – SAM2 masks + optional cleanup for ALL frames (used for SRT scale)
 [4a. Depth (optional)]           – FoundationStereo depth, used by SRT scale (--sam3d_use_depth)
  S1. Select frames               – pick one frame per azimuthal bin (60° default)
  S2. SAM3D                       – run SAM3D on each selected frame → GLB mesh
  S3. SRT scale                   – estimate scale+pose using Stage-1 silhouettes only
  S4. Render debug                – render debug overlay image for each SAM3D mesh
  S5. Render video                – project SRT mesh onto all Stage-1 keyframes → render_video.mp4
  S6. Select suggested best       – rank candidates and copy best artifacts to sam3d/best/

Two frames_meta.json files are used:
  - mapping_data_dir/frames_meta.json          : input metadata (timestamps, no poses)
  - job_dir/sfm/keyframes/frames_meta.json     : CuSFM output (camera-to-world poses)

Usage:
    python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \\
        --mapping_data_dir /home/.../mapping_data/2026-02-18_..._bowl \\
        --job_dir          /data/hoi_obj_recon/2026-02-18_..._bowl \\
        --prompt           "bowl"

    # SAM3D mode:
    python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \\
        --mapping_data_dir ... --job_dir ... --prompt "bowl" \\
        --mode sam3d [--sam3d_use_depth] [--sam3d_bin_deg 60] [--sam3d_seed 42] \\
        [--sam3d_srt_max_views 25] [--sam3d_srt_maxiter 60] \\
        [--sam3d_srt_top_k 1] [--sam3d_srt_parallel 8]

Skip flags (BundleSDF):
    --skip_prepare  --skip_sfm  --skip_sfm_quality_check  --skip_stage1_detect  --skip_dino  --skip_depth  --skip_mask
    --skip_mask_postprocess
    --skip_stage1_setup  --skip_stage1_nerf  --skip_center_mesh
    --skip_fp_tracking  --skip_fp_render  --skip_world_poses  --skip_merged_setup  --skip_full_nerf
    --skip_glb_export  --skip_final_fp_tracking  --skip_final_fp_render

Skip flags (SAM3D):
    --skip_prepare  --skip_sfm  --skip_sfm_quality_check  --skip_stage1_detect  --skip_dino  --skip_depth  --skip_mask
    --skip_mask_postprocess
    --skip_select_frames  --skip_sam3d  --skip_srt_scale  --skip_render_debug  --skip_render_video
    --skip_select_best_mesh
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses as _run_fp_tracking
from v2d_cusfm.docker.gpu_compatibility import require_compatible_cusfm_gpus
from v2d_hoi_object_reconstruction.docker._container_steps import (
    convert_poses_to_matrix,
    postprocess_masks,
    prepare_job,
    run_grounding_dino,
    run_sam3d_srt,
    select_best_sam3d,
    select_sam3d_frames,
    stitch_mp4,
)
from v2d_hoi_object_reconstruction.docker._trajectory_tools import (
    run_sfm_quality_check,
    run_stage1_detection,
)


# ── Image names ────────────────────────────────────────────────────────────────

IMAGE_HOI                   = "v2d_hoi_object_reconstruction"
IMAGE_CUSFM                 = "v2d_cusfm"
IMAGE_BUNDLESDF             = "v2d_bundlesdf"
IMAGE_FOUNDATION_STEREO     = "v2d_foundation_stereo"
IMAGE_SAM2                  = "v2d_sam2"
IMAGE_FOUNDATIONPOSE_RENDER = "v2d_foundation_pose"
IMAGE_SAM3D                 = "v2d_sam3d"


# ── Constants ──────────────────────────────────────────────────────────────────

from v2d_hoi_object_reconstruction.docker._pipeline_utils import (
    IMAGE_EXTENSIONS, detect_gpu_ids, count_images,
)

_DATA_DIR           = Path(__file__).parents[3] / "data"    # reconstruction/data/
_WEIGHTS_DIR        = _DATA_DIR / "weights"                 # reconstruction/data/weights/
_FP_WEIGHTS_DIR     = _WEIGHTS_DIR / "foundationpose"       # reconstruction/data/weights/foundationpose/
_SAM3D_WEIGHTS_DIR  = _WEIGHTS_DIR / "sam3d"               # reconstruction/data/weights/sam3d/

# Config paths (host-side; mounted into containers at runtime)
_DEFAULT_PIPELINE_CONFIG_HOST = str(
    Path(__file__).parent / "data" / "configs" / "hoi_pipeline.yaml"
)


# ── Docker run helper ──────────────────────────────────────────────────────────

def _run_gpu(image, cmd_args, mounts, gpu_id=None, user=None, extra_env=None):
    """Run a docker container with GPU support and bind mounts.

    Args:
        image: Docker image name
        cmd_args: Command arguments to pass to the container
        mounts: List of (host_path, container_path) tuples
        gpu_id: Specific GPU ID to use, or None for all GPUs
        user: Optional --user argument (e.g. "1000:1000")
        extra_env: Optional dict of additional environment variables
    """
    cmd = ["docker", "run", "--rm"]
    if gpu_id is not None:
        cmd += ["-e", f"CUDA_VISIBLE_DEVICES={gpu_id}",
                "-e", f"NVIDIA_VISIBLE_DEVICES={gpu_id}"]
    cmd += ["--gpus", "all"]
    if user:
        cmd += ["--user", user]
    if extra_env:
        for k, v in extra_env.items():
            cmd += ["-e", f"{k}={v}"]
    for host_path, container_path in mounts:
        cmd += ["-v", f"{os.path.abspath(host_path)}:{container_path}"]
    cmd += [image] + cmd_args
    print(f"\n[run] {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Docker failed: {image} (exit {result.returncode})")



# ── Depth workers ──────────────────────────────────────────────────────────────

def run_depth_workers(job_dir: str, n_frames: int, gpu_ids: list,
                      num_workers: int, model_dir: str = None) -> None:
    """Run parallel FoundationStereo depth workers via docker run."""
    job_dir_abs = os.path.abspath(job_dir)
    n_workers = min(num_workers, n_frames, len(gpu_ids))
    chunk = (n_frames + n_workers - 1) // n_workers
    ranges = [(i * chunk, min((i + 1) * chunk, n_frames)) for i in range(n_workers)]

    print(f"[pipeline] depth: {n_frames} frames, {n_workers} workers, GPUs {gpu_ids}")

    procs = []

    def worker(start_idx, end_idx, worker_id, gpu_id):
        model_dir_abs = os.path.abspath(model_dir) if model_dir else str(_WEIGHTS_DIR / "foundationstereo")
        cmd = [
            "docker", "run", "--rm",
            "-e", f"CUDA_VISIBLE_DEVICES={gpu_id}",
            "-e", f"NVIDIA_VISIBLE_DEVICES={gpu_id}",
            "--gpus", "all",
            "-v", f"{job_dir_abs}:/data/job",
            "-v", f"{model_dir_abs}:/data/foundation_stereo_models",
        ]
        cmd += [
            IMAGE_FOUNDATION_STEREO,
            "python", "-m", "v2d.foundation_stereo.lib.image_list_to_depth",
            "--left_dir",          "/data/job/left",
            "--right_dir",         "/data/job/right",
            "--depth_folder",      "/data/job/depth",
            "--intrinsics_folder", "/data/job/intrinsics",
            "--calibration_file",  "/data/job/calibration.json",
            "--model_dir",         "/data/foundation_stereo_models",
            "--start_idx", str(start_idx),
            "--end_idx",   str(end_idx),
        ]
        print(f"[depth worker {worker_id}] frames {start_idx}–{end_idx-1} (GPU {gpu_id})")
        p = subprocess.Popen(cmd)
        procs.append(p)
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Depth worker {worker_id} failed (exit {rc})")
        print(f"[depth worker {worker_id}] done")

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(worker, s, e, i, gpu_ids[i % len(gpu_ids)]): i
                for i, (s, e) in enumerate(ranges)
            }
            for fut in as_completed(futures):
                fut.result()
    except Exception:
        print("[pipeline] depth worker failed — killing containers", file=sys.stderr)
        for p in procs:
            if p.poll() is None:
                p.kill()
        raise


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end object reconstruction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mapping_data_dir", required=True,
                        help="Raw mapping data directory (frames_meta.json + images)")
    parser.add_argument("--job_dir", required=True,
                        help="Output root (e.g. /data/hoi_obj_recon/<job>)")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt for Grounding DINO (e.g. 'bowl')")
    stage1_group = parser.add_mutually_exclusive_group(required=False)
    stage1_group.add_argument("--stage1_end_frame", type=int,
                        help="Last sequential job-folder index of Stage 1 (0-based, inclusive).")
    stage1_group.add_argument("--stage1_end_timestamp", type=int,
                        help="Nanosecond timestamp of the last Stage-1 frame "
                             "(matches image filename in front_stereo_camera_left/). "
                             "Converted to seq_idx automatically via frames_meta.json.")
    parser.add_argument("--stage1_buffer_deg", type=float, default=None,
                        help="Angle buffer (°) for auto stage-1 detection (default: 10°)")
    parser.add_argument("--config", default=None,
                        help="NeRF/SDF config YAML host path (optional; bundlesdf uses its own default if omitted)")
    parser.add_argument("--bundlesdf_trunc", "--bundlesdf-trunc", dest="bundlesdf_trunc",
                        type=float, default=None,
                        help="Override BundleSDF nerf.trunc / trunc_start in meters")
    parser.add_argument("--bundlesdf_mesh_resolution", "--bundlesdf-mesh-resolution",
                        dest="bundlesdf_mesh_resolution", type=float, default=None,
                        help="Override BundleSDF nerf.mesh_resolution in meters")
    parser.add_argument("--pipeline_config", default=None,
                        help=f"HOI pipeline config YAML, host path (default: {_DEFAULT_PIPELINE_CONFIG_HOST}). "
                             "Controls per-step settings (stage1_detect, sfm, sfm_scan_quality, "
                             "grounding_dino, depth, foundationpose, texture_bake).")
    parser.add_argument("--box_threshold", type=float, default=None,
                        help="Grounding DINO box confidence threshold (overrides pipeline_config)")
    parser.add_argument("--reference_frame", type=int, default=None,
                        help="Reference frame for FoundationPose (overrides pipeline_config)")
    parser.add_argument("--num_depth_workers", type=int, default=None,
                        help="Number of parallel depth workers (overrides pipeline_config)")
    parser.add_argument("--fp_weights_dir", type=str, default=None,
                        help="Path to FoundationPose weights directory (overrides pipeline_config). "
                             f"Default: {_FP_WEIGHTS_DIR}")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="GPU IDs (default: auto-detect)")

    # Mode
    parser.add_argument("--mode", choices=["bundlesdf", "sam3d"], default="bundlesdf",
                        help="Reconstruction mode (default: bundlesdf)")
    parser.add_argument("--sam3d_use_depth", action="store_true",
                        help="SAM3D mode: also run FoundationStereo depth for SRT scale estimation")
    parser.add_argument("--sam3d_bin_deg", type=float, default=60.0,
                        help="SAM3D mode: azimuthal bin size for frame selection (default: 60°)")
    parser.add_argument("--sam3d_seed", type=int, default=42,
                        help="SAM3D mode: random seed passed to SAM3D (default: 42)")
    parser.add_argument("--sam3d_srt_max_views", type=int, default=25,
                        help="SAM3D mode: maximum silhouette views per SRT candidate "
                             "(default: 25; OSMO fast path)")
    parser.add_argument("--sam3d_srt_maxiter", type=int, default=60,
                        help="SAM3D mode: Powell iterations per SRT optimisation "
                             "(default: 60; OSMO fast path)")
    parser.add_argument("--sam3d_srt_top_k", type=int, default=1,
                        help="SAM3D mode: axis orientations fully optimised when no "
                             "orientation prior is available (default: 1; OSMO fast path)")
    parser.add_argument("--sam3d_srt_parallel", type=int, default=8,
                        help="SAM3D mode: parallel SRT candidate workers "
                             "(default: 8; OSMO fast path)")
    parser.add_argument("--sam3d_force_srt", action="store_true",
                        help="SAM3D mode: recompute complete SRT candidates instead of resuming")

    # Skip flags
    parser.add_argument("--skip_prepare",        action="store_true")
    parser.add_argument("--skip_sfm",            action="store_true")
    parser.add_argument("--skip_sfm_quality_check", action="store_true",
                        help="Skip CuSFM two-loop scan quality validation")
    parser.add_argument("--skip_stage1_detect",  action="store_true",
                        help="Skip auto stage-1 end detection (requires manual --stage1_end_frame/timestamp)")
    parser.add_argument("--skip_dino",         action="store_true")
    parser.add_argument("--skip_depth",        action="store_true")
    parser.add_argument("--skip_mask",         action="store_true")
    parser.add_argument("--skip_mask_postprocess", action="store_true",
                        help="Skip configured mask post-processing")
    parser.add_argument("--skip_stage1_setup", action="store_true")
    parser.add_argument("--skip_stage1_nerf",  action="store_true")
    parser.add_argument("--skip_center_mesh",  action="store_true")
    parser.add_argument("--skip_fp_tracking",  action="store_true")
    parser.add_argument("--skip_fp_render",    action="store_true",
                        help="Skip FoundationPose render overlay video")
    parser.add_argument("--skip_world_poses",  action="store_true")
    parser.add_argument("--skip_merged_setup",        action="store_true")
    parser.add_argument("--skip_full_nerf",            action="store_true")
    parser.add_argument("--skip_glb_export",           action="store_true",
                        help="BundleSDF mode: skip exporting textured_mesh.obj to output.glb")
    parser.add_argument("--skip_final_fp_tracking",    action="store_true",
                        help="Skip FoundationPose tracking with final textured mesh")
    parser.add_argument("--skip_final_fp_render",      action="store_true",
                        help="Skip FoundationPose render overlay video with final textured mesh")
    # SAM3D-specific skip flags
    parser.add_argument("--skip_select_frames", action="store_true",
                        help="SAM3D mode: skip representative-frame selection")
    parser.add_argument("--skip_sam3d",         action="store_true",
                        help="SAM3D mode: skip SAM3D mesh reconstruction (reuse existing GLBs)")
    parser.add_argument("--skip_srt_scale",     action="store_true",
                        help="SAM3D mode: skip SRT scale estimation")
    parser.add_argument("--skip_render_debug",  action="store_true",
                        help="SAM3D mode: skip SAM3D debug render")
    parser.add_argument("--skip_render_video",  action="store_true",
                        help="SAM3D mode: skip Stage-1 overlay video render")
    parser.add_argument("--skip_select_best_mesh", action="store_true",
                        help="SAM3D mode: skip suggested best mesh selection")

    args = parser.parse_args()

    for name in (
        "sam3d_srt_max_views",
        "sam3d_srt_maxiter",
        "sam3d_srt_top_k",
        "sam3d_srt_parallel",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")

    gpu_ids = args.gpu_ids or detect_gpu_ids()
    require_compatible_cusfm_gpus(gpu_ids)

    if args.mode == "sam3d":
        from v2d.sam3d.docker.run_image_to_mesh import (
            run_image_to_mesh as _run_sam3d,
        )
        from v2d.sam3d.docker.run_render_debug_image import (
            run_render_debug_image as _run_sam3d_render,
        )

    # ── Load pipeline config ───────────────────────────────────────────────────
    import yaml as _yaml
    _pipeline_cfg_path = args.pipeline_config or _DEFAULT_PIPELINE_CONFIG_HOST
    with open(_pipeline_cfg_path) as _f:
        _pcfg = _yaml.safe_load(_f)

    def _pcfg_get(section, key, default=None):
        return (_pcfg.get(section) or {}).get(key, default)

    # Resolve values: CLI arg wins over pipeline config, pipeline config wins over hardcoded default
    stage1_buffer_deg  = args.stage1_buffer_deg \
                         if args.stage1_buffer_deg is not None \
                         else _pcfg_get("stage1_detect", "buffer_deg", 10.0)
    box_threshold      = args.box_threshold      if args.box_threshold is not None \
                         else _pcfg_get("grounding_dino", "box_threshold", 0.3)
    ref_frame_cfg      = args.reference_frame    if args.reference_frame is not None \
                         else _pcfg_get("foundationpose", "reference_frame", 0)
    num_depth_workers  = args.num_depth_workers  if args.num_depth_workers is not None \
                         else _pcfg_get("depth", "num_workers", 1)
    fp_weights_dir_cfg = args.fp_weights_dir     if args.fp_weights_dir is not None \
                         else _pcfg_get("foundationpose", "weights_dir")
    sfm_quality_cfg = _pcfg.get("sfm_scan_quality") or {}
    sfm_quality_enabled = (
        bool(sfm_quality_cfg.get("enabled", True))
        and not args.skip_sfm_quality_check
    )
    sfm_quality_fail_on_error = bool(sfm_quality_cfg.get("fail_on_error", True))

    mask_postprocess_cfg = _pcfg.get("mask_postprocess") or {}
    mask_postprocess_modes = mask_postprocess_cfg.get("modes")
    if mask_postprocess_modes is None:
        mask_postprocess_modes = ["bundlesdf", "sam3d"]
    elif isinstance(mask_postprocess_modes, str):
        mask_postprocess_modes = [mask_postprocess_modes]
    mask_postprocess_enabled = (
        bool(mask_postprocess_cfg.get("enabled", False))
        and not args.skip_mask_postprocess
        and args.mode in set(mask_postprocess_modes)
    )
    mask_postprocess_effective_cfg = {
        "keep_largest_component": bool(mask_postprocess_cfg.get("keep_largest_component", False)),
        "min_component_area_px": int(mask_postprocess_cfg.get("min_component_area_px", 2000)),
        "min_component_area_frac": float(mask_postprocess_cfg.get("min_component_area_frac", 0.01)),
        "open_px": int(mask_postprocess_cfg.get("open_px", 0)),
        "erode_px": int(mask_postprocess_cfg.get("erode_px", 3)),
    }
    mask_postprocess_fingerprint = hashlib.sha256(
        json.dumps(mask_postprocess_effective_cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()
    job_dir = os.path.abspath(args.job_dir)
    os.makedirs(job_dir, exist_ok=True)

    mapping_data_dir = Path(os.path.abspath(args.mapping_data_dir))

    ref_frame        = ref_frame_cfg
    fp_gpu           = gpu_ids[-1]   # reserve last GPU for single-GPU tasks (NeRF / FoundationPose)
    fp_weights_dir   = os.path.abspath(fp_weights_dir_cfg or str(_FP_WEIGHTS_DIR))

    # Resolve stage1_end_frame from timestamp if provided
    stage1_end_frame = None
    if args.stage1_end_frame is not None:
        stage1_end_frame = args.stage1_end_frame
    elif args.stage1_end_timestamp is not None:
        ts_us = args.stage1_end_timestamp // 1000
        with open(mapping_data_dir / 'frames_meta.json') as f:
            meta = json.load(f)
        cam_params = meta['camera_params_id_to_camera_params']
        left_sids, right_sids = {}, set()
        for kf in meta['keyframes_metadata']:
            sid    = int(kf['synced_sample_id'])
            sensor = cam_params[kf['camera_params_id']]['sensor_meta_data']['sensor_name']
            if 'front_stereo_camera_left' in sensor:
                left_sids[sid] = int(kf['timestamp_microseconds'])
            elif 'front_stereo_camera_right' in sensor:
                right_sids.add(sid)
        common_sids = sorted(set(left_sids) & right_sids)
        ts_to_idx = {left_sids[sid]: i for i, sid in enumerate(common_sids)}
        stage1_end_frame = ts_to_idx.get(ts_us)
        if stage1_end_frame is None:
            raise ValueError(
                f"--stage1_end_timestamp {args.stage1_end_timestamp} "
                f"(={ts_us} µs) not found in frames_meta.json")
        print(f"[pipeline] stage1_end_timestamp {args.stage1_end_timestamp} → seq_idx {stage1_end_frame}")

    # Derived paths (all absolute host paths)
    input_frames_meta = os.path.join(job_dir, "frames_meta.json")
    sfm_poses_meta    = os.path.join(job_dir, "sfm", "keyframes", "frames_meta.json")

    ref_frame_path    = os.path.join(job_dir, "ref_frame.jpg")
    dino_bboxes       = os.path.join(job_dir, "grounding_dino_bboxes.json")
    prompts_json      = os.path.join(job_dir, "prompts.json")
    depth_dir         = os.path.join(job_dir, "depth")
    masks_dir         = os.path.join(job_dir, "masks")
    mask_path         = os.path.join(job_dir, "masks", "0", f"{ref_frame:06d}.png")
    mask_prompt_hash  = os.path.join(job_dir, "masks", ".prompts.sha256")
    mask_post_hash    = os.path.join(job_dir, "masks", ".postprocess.sha256")
    sam2_masks_dir    = os.path.join(job_dir, "masks_raw_sam2") if mask_postprocess_enabled else masks_dir
    sam2_mask_path    = os.path.join(sam2_masks_dir, "0", f"{ref_frame:06d}.png")
    sam2_prompt_hash  = os.path.join(sam2_masks_dir, ".prompts.sha256")
    stage1_recon_dir  = os.path.join(job_dir, "stage1_recon")
    stage1_mesh_raw   = os.path.join(job_dir, "stage1_recon", "textured_mesh.obj")
    stage1_mesh_glb   = os.path.join(job_dir, "stage1_recon", "output.glb")
    centered_mesh     = os.path.join(job_dir, "mesh_input.glb")
    poses_dir         = os.path.join(job_dir, "poses")
    poses_world       = os.path.join(job_dir, "poses_world.json")
    merged_recon_dir  = os.path.join(job_dir, "merged_recon")
    final_mesh_obj    = os.path.join(job_dir, "merged_recon", "textured_mesh.obj")
    final_mesh_glb    = os.path.join(job_dir, "merged_recon", "output.glb")
    poses_final_dir   = os.path.join(job_dir, "poses_final")
    intrinsics_path   = os.path.join(job_dir, "intrinsics", f"{ref_frame:06d}.json")

    def _ensure_ref_frame():
        """Copy reference frame as a real file (symlinks in left/ may not resolve in containers)."""
        if not os.path.exists(ref_frame_path):
            src = Path(os.path.join(job_dir, "left", f"{ref_frame:06d}.jpg")).resolve()
            os.makedirs(os.path.dirname(ref_frame_path), exist_ok=True)
            shutil.copy2(src, ref_frame_path)

    # ── Container-side paths for _run_gpu calls (external images) ─────────────
    c_job         = "/data/job"
    c_prompts_json = f"{c_job}/prompts.json"
    c_depth_dir   = f"{c_job}/depth"
    c_centered_mesh = f"{c_job}/mesh_input.glb"
    c_poses_dir   = f"{c_job}/poses"
    c_final_mesh_obj = f"{c_job}/merged_recon/textured_mesh.obj"
    c_final_mesh_glb = f"{c_job}/merged_recon/output.glb"
    c_poses_final = f"{c_job}/poses_final"
    c_intrinsics  = f"{c_job}/intrinsics/{ref_frame:06d}.json"

    _timings = {}
    _t_total = time.time()

    def _select_stage1_center_mesh() -> str:
        """Prefer the self-contained Stage-1 GLB; fall back to OBJ for reuse runs."""
        if os.path.exists(stage1_mesh_glb):
            return stage1_mesh_glb
        if os.path.exists(stage1_mesh_raw):
            print(
                "[pipeline] Stage-1 GLB not found; centering OBJ mesh for FoundationPose"
            )
            return stage1_mesh_raw
        return stage1_mesh_glb

    def _select_final_fp_mesh() -> tuple[str, str]:
        """Prefer the self-contained GLB; fall back to OBJ for --skip_glb_export reuse."""
        if os.path.exists(final_mesh_glb):
            return final_mesh_glb, c_final_mesh_glb
        if os.path.exists(final_mesh_obj):
            print(
                "[pipeline] final GLB not found; using OBJ mesh for final FoundationPose"
            )
            return final_mesh_obj, c_final_mesh_obj
        return final_mesh_glb, c_final_mesh_glb

    def _step(name, fn):
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        _timings[name] = elapsed
        print(f"[pipeline] {name} done in {elapsed:.1f}s")

    # ── Step 1: Prepare FP folder ─────────────────────────────────────────────
    if not args.skip_prepare:
        print("[pipeline] preparing FP folder")
        _step("prepare", lambda: prepare_job(
            image=IMAGE_HOI,
            input_dir=mapping_data_dir,
            job_dir=job_dir,
        ))

    # NeRF config: optional user override; if not given, bundlesdf uses its own default
    nerf_config_path = args.config
    if nerf_config_path:
        print(f"[pipeline] using NeRF config: {nerf_config_path}")

    # ── Step 2: CuSFM ────────────────────────────────────────────────────────
    if not args.skip_sfm:
        print("[pipeline] running CuSFM")
        sfm_output_dir = os.path.join(job_dir, "sfm")
        def _sfm():
            _run_gpu(
                IMAGE_CUSFM,
                [
                    "python", "-m", "v2d_cusfm.lib.image_list_to_sfm",
                    "--input_dir",  "/data/mapping_data",
                    "--output_dir", "/data/job/sfm",
                ],
                mounts=[
                    (str(mapping_data_dir), "/data/mapping_data"),
                    (job_dir, "/data/job"),
                ],
            )
        _step("sfm", _sfm)

    # ── Step 2b: CuSFM scan-pattern quality gate ─────────────────────────────
    if sfm_quality_enabled:
        if not os.path.exists(sfm_poses_meta):
            raise FileNotFoundError(
                f"CuSFM quality check requested, but poses were not found: {sfm_poses_meta}")
        print("[pipeline] checking CuSFM two-loop scan quality")
        quality_out = Path(job_dir) / "sfm_scan_quality"
        t0 = time.time()
        run_sfm_quality_check(
            image=IMAGE_HOI,
            sfm_keyframes=sfm_poses_meta,
            frames_meta=input_frames_meta,
            output_dir=quality_out,
            config=sfm_quality_cfg,
            fail_on_error=sfm_quality_fail_on_error,
        )
        _timings["sfm_scan_quality"] = time.time() - t0
        print(f"[pipeline] sfm_scan_quality done in {_timings['sfm_scan_quality']:.1f}s")

    # ── Step 2c: Auto-detect Stage-1 end (if not manually specified) ─────────
    # When detection is explicitly skipped, reuse its prior result so downstream
    # SRT/render steps still have the Stage-1 boundary. A normal rerun must
    # recompute this value from the newly generated SfM trajectory.
    if stage1_end_frame is None and args.skip_stage1_detect:
        _existing = Path(job_dir) / "stage1_detect_debug" / "result.json"
        if _existing.exists():
            with open(_existing) as _f:
                stage1_end_frame = json.load(_f).get("stage1_end_frame")
            print(f"[pipeline] stage1_end_frame={stage1_end_frame} (loaded from existing result.json)")

    if stage1_end_frame is None and not args.skip_stage1_detect:
        print("[pipeline] auto-detecting Stage-1 end from CuSFM trajectory")
        detect_out = Path(job_dir) / "stage1_detect_debug"
        t0 = time.time()
        stage1_end_frame = run_stage1_detection(
            image=IMAGE_HOI,
            sfm_keyframes=sfm_poses_meta,
            frames_meta=input_frames_meta,
            output_dir=detect_out,
            buffer_deg=stage1_buffer_deg,
        )
        _timings["stage1_detect"] = time.time() - t0
        print(f"[pipeline] stage1_detect done in {_timings['stage1_detect']:.1f}s → seq_idx {stage1_end_frame}")

    if stage1_end_frame is None and args.mode == "bundlesdf":
        raise ValueError(
            "Stage-1 end frame could not be determined. "
            "Provide --stage1_end_frame, --stage1_end_timestamp, "
            "or remove --skip_stage1_detect to enable auto-detection.")

    # ── Step 3: Grounding DINO ────────────────────────────────────────────────
    if not args.skip_dino or not args.skip_mask:
        _ensure_ref_frame()
    if not args.skip_dino:
        print(f"[pipeline] running Grounding DINO (prompt: '{args.prompt}')")
        dino_model_dir = str(_WEIGHTS_DIR / "grounding_dino")
        def _dino():
            run_grounding_dino(
                image_path=ref_frame_path,
                output_path=dino_bboxes,
                prompt=args.prompt,
                model_dir=dino_model_dir,
                box_threshold=box_threshold,
            )
            with open(dino_bboxes) as f:
                detections = json.load(f)
            if not detections:
                print(f"[error] Grounding DINO found no objects for '{args.prompt}'",
                      file=sys.stderr)
                sys.exit(1)
            top = detections[0]
            print(f"[pipeline] top detection: {top['label']} (conf {top['confidence']:.2f})")
            prompts = {"prompts": [{
                "frame_index": ref_frame,
                "object_id":   0,
                "points":      None,
                "point_labels": None,
                "box":         top["box"],
            }]}
            with open(prompts_json, "w") as f:
                json.dump(prompts, f, indent=2)
        _step("grounding_dino", _dino)

    # Load DINO detections once for all downstream steps that need the bbox.
    dino_detections = None
    dino_bbox_str   = None
    if os.path.exists(dino_bboxes):
        with open(dino_bboxes) as f:
            dino_detections = json.load(f)
        if dino_detections:
            box = dino_detections[0]['box']
            dino_bbox_str = ",".join(str(int(b)) for b in [box['x0'], box['y0'], box['x1'], box['y1']])

    # ── Steps 4a + 4b: Depth + Mask (parallel) ───────────────────────────────
    # In SAM3D mode depth is optional (only if --sam3d_use_depth); mask is always needed
    _run_depth = not args.skip_depth and (args.mode == "bundlesdf" or args.sam3d_use_depth)
    _run_mask  = not args.skip_mask
    if _run_depth or _run_mask:
        n_frames = count_images(os.path.join(job_dir, "left"))

        def run_depth():
            run_depth_workers(job_dir, n_frames, gpu_ids, num_depth_workers)

        def run_mask():
            prompts_hash = file_sha256(prompts_json)

            if mask_postprocess_enabled and not os.path.exists(sam2_mask_path) and os.path.exists(mask_path):
                current_hash = Path(mask_prompt_hash).read_text().strip() if os.path.exists(mask_prompt_hash) else None
                if current_hash == prompts_hash and not os.path.exists(mask_post_hash):
                    print("[pipeline] moving existing raw masks to masks_raw_sam2 for post-processing")
                    if os.path.exists(sam2_masks_dir):
                        shutil.rmtree(sam2_masks_dir)
                    shutil.move(masks_dir, sam2_masks_dir)
                    Path(sam2_prompt_hash).write_text(prompts_hash + "\n")

            stored_hash = None
            if os.path.exists(sam2_mask_path):
                if os.path.exists(sam2_prompt_hash):
                    stored_hash = Path(sam2_prompt_hash).read_text().strip()
                    if stored_hash == prompts_hash:
                        print("[pipeline] SAM2 masks already exist for current prompts")
                    else:
                        print("[pipeline] prompts.json changed; regenerating SAM2 masks")
                        shutil.rmtree(sam2_masks_dir)
                        stored_hash = None
                elif os.path.getmtime(prompts_json) <= os.path.getmtime(sam2_mask_path):
                    print("[pipeline] SAM2 masks already exist; recording current prompt hash")
                    Path(sam2_prompt_hash).write_text(prompts_hash + "\n")
                    stored_hash = prompts_hash
                else:
                    print("[pipeline] prompts.json is newer than SAM2 masks; regenerating masks")
                    shutil.rmtree(sam2_masks_dir)
                    stored_hash = None

            if stored_hash != prompts_hash:
                Path(sam2_masks_dir).mkdir(parents=True, exist_ok=True)
                c_sam2_masks_dir = f"{c_job}/masks_raw_sam2" if mask_postprocess_enabled else f"{c_job}/masks"
                _run_gpu(
                    IMAGE_SAM2,
                    [
                        "python", "-m", "v2d.sam2.lib.video_to_masks",
                        "--video_path",   f"{c_job}/video.mp4",
                        "--prompts_path", c_prompts_json,
                        "--masks_dir",    c_sam2_masks_dir,
                        "--weights_dir",  "/data/sam2_weights",
                    ],
                    mounts=[
                        (job_dir, "/data/job"),
                        (str(_WEIGHTS_DIR / "sam2"), "/data/sam2_weights"),
                    ],
                    user=f"{os.getuid()}:{os.getgid()}",
                    extra_env={"HOME": "/tmp"},
                )
                Path(sam2_prompt_hash).write_text(prompts_hash + "\n")

            if mask_postprocess_enabled:
                output_current = (
                    os.path.exists(mask_path)
                    and os.path.exists(mask_prompt_hash)
                    and Path(mask_prompt_hash).read_text().strip() == prompts_hash
                    and os.path.exists(mask_post_hash)
                    and Path(mask_post_hash).read_text().strip() == mask_postprocess_fingerprint
                )
                if output_current:
                    print("[pipeline] post-processed masks already exist for current prompts/config")
                    return

                print(f"[pipeline] post-processing masks: {mask_postprocess_effective_cfg}")
                if os.path.exists(masks_dir):
                    shutil.rmtree(masks_dir)
                summary = postprocess_masks(
                    image=IMAGE_HOI,
                    input_dir=sam2_masks_dir,
                    output_dir=masks_dir,
                    config=mask_postprocess_effective_cfg,
                )
                Path(mask_prompt_hash).write_text(prompts_hash + "\n")
                Path(mask_post_hash).write_text(mask_postprocess_fingerprint + "\n")
                area_ratio = summary.get("area_ratio_median")
                area_ratio_str = f"{area_ratio:.3f}" if area_ratio is not None else "n/a"
                print(
                    "[pipeline] mask postprocess: "
                    f"frames={summary['frames']} "
                    f"area_ratio_median={area_ratio_str} "
                    f"empty_after_frames={summary['empty_after_frames']}"
                )
            elif sam2_masks_dir != masks_dir:
                raise RuntimeError("internal error: SAM2 mask dir differs while post-processing is disabled")
            else:
                print("[pipeline] mask ready")

        t0 = time.time()
        tasks = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            if _run_depth:
                tasks['depth'] = pool.submit(run_depth)
            if _run_mask:
                tasks['mask']  = pool.submit(run_mask)
            for name, fut in tasks.items():
                fut.result()
                print(f"[pipeline] {name} complete")
        elapsed = time.time() - t0
        _timings["depth+mask"] = elapsed
        print(f"[pipeline] depth+mask done in {elapsed:.1f}s")

    # ── Step 5: Stage-1 recon setup ───────────────────────────────────────────
    if args.mode == "bundlesdf" and not args.skip_stage1_setup:
        print("[pipeline] setting up Stage-1 reconstruction directory")
        _step("stage1_setup", lambda: run_in_container(
            image=IMAGE_HOI,
            module="v2d_hoi_object_reconstruction.lib.setup_reconstruction_for_stage1",
            inputs={
                "frames_meta":   input_frames_meta,
                "sfm_keyframes": sfm_poses_meta,
                "depth_dir":     depth_dir,
                "masks_dir":     masks_dir,
                "left_dir":      os.path.join(job_dir, "left"),
                "right_dir":     os.path.join(job_dir, "right"),
            },
            outputs={"output_dir": stage1_recon_dir},
            extra_args={"stage1_end_frame": stage1_end_frame},
            gpus=True,
        ))

    # ── Step 6: Stage-1 NeRF reconstruction ──────────────────────────────────
    _nerf_inputs = {"weights_dir": str(_WEIGHTS_DIR)}
    if nerf_config_path:
        _nerf_inputs["config"] = nerf_config_path

    def _bundlesdf_extra_args():
        extra = {
            "bbox_str": dino_bbox_str,
            "skip-glb-export": args.skip_glb_export,
        }
        if args.bundlesdf_trunc is not None:
            extra["trunc"] = args.bundlesdf_trunc
        if args.bundlesdf_mesh_resolution is not None:
            extra["mesh_resolution"] = args.bundlesdf_mesh_resolution
        return extra

    if args.mode == "bundlesdf" and not args.skip_stage1_nerf:
        print("[pipeline] running Stage-1 NeRF reconstruction")
        _step("stage1_nerf", lambda: run_in_container(
            image=IMAGE_BUNDLESDF,
            module="v2d_bundlesdf.lib.reconstruct",
            inputs=_nerf_inputs,
            outputs={"output_path": stage1_recon_dir},
            extra_args=_bundlesdf_extra_args(),
            env={"CUDA_VISIBLE_DEVICES": str(fp_gpu), "NVIDIA_VISIBLE_DEVICES": str(fp_gpu)},
            gpus=True,
        ))

    # ── Step 7: Center mesh ───────────────────────────────────────────────────
    if args.mode == "bundlesdf" and not args.skip_center_mesh:
        print("[pipeline] centering mesh")
        _step("center_mesh", lambda: run_in_container(
            image=IMAGE_HOI,
            module="v2d_hoi_object_reconstruction.lib.center_mesh",
            inputs={"input": _select_stage1_center_mesh()},
            outputs={"output": centered_mesh},
        ))

    # ── Step 8: FoundationPose tracking ───────────────────────────────────────
    if args.mode == "bundlesdf" and not args.skip_fp_tracking:
        if not os.path.exists(mask_path):
            print(f"[error] mask not found: {mask_path}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(fp_weights_dir):
            print(
                f"[error] FoundationPose weights not found at: {fp_weights_dir}\n"
                f"  Download them first:\n"
                f"    python modules/v2d_foundation_pose/docker/run_download_weights.py"
                f" --output_dir {fp_weights_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print("[pipeline] running FoundationPose tracking")
        _step("fp_tracking", lambda: _run_fp_tracking(
            video_path=os.path.join(job_dir, "video.mp4"),
            depth_folder=depth_dir,
            masks_folder=os.path.join(job_dir, "masks", "0"),
            camera_intrinsics_path=intrinsics_path,
            mesh_path=centered_mesh,
            poses_dir=poses_dir,
            weights_dir=fp_weights_dir,
            reference_frame=ref_frame,
        ))

    # ── Step 8b: FoundationPose render overlay ────────────────────────────────
    if args.mode == "bundlesdf" and not args.skip_fp_render:
        convert_poses_to_matrix(image=IMAGE_HOI, poses_dir=poses_dir)
        print("[pipeline] rendering FoundationPose overlay video")
        _step("fp_render", lambda: _run_gpu(
            IMAGE_FOUNDATIONPOSE_RENDER,
            [
                "python", "-m", "v2d.foundation_pose.lib.render_overlay",
                "--video_path",             f"{c_job}/video.mp4",
                "--poses_dir",              c_poses_dir,
                "--mesh_path",              c_centered_mesh,
                "--camera_intrinsics_path", c_intrinsics,
                "--output_dir",             f"{c_job}/fp_render",
            ],
            mounts=[(job_dir, "/data/job")],
            gpu_id=fp_gpu,
            user=f"{os.getuid()}:{os.getgid()}",
        ))
        _step("fp_render_stitch", lambda: stitch_mp4(
            image=IMAGE_HOI,
            frames_dir=os.path.join(job_dir, "fp_render"),
            output_mp4=os.path.join(job_dir, "fp_render", "render.mp4")))

    # ── Step 9: World poses + stage detection ─────────────────────────────────
    if args.mode == "bundlesdf" and not args.skip_world_poses:
        print("[pipeline] computing world poses and detecting stages")
        _step("world_poses", lambda: run_in_container(
            image=IMAGE_HOI,
            module="v2d_hoi_object_reconstruction.lib.compute_object_world_poses",
            inputs={
                "poses_dir":     poses_dir,
                "frames_meta":   input_frames_meta,
                "sfm_keyframes": sfm_poses_meta,
            },
            outputs={
                "output": poses_world,
                "plot":   os.path.join(job_dir, "poses_world_debug.png"),
            },
            extra_args={"camera": "left", "stage1_prior_frame": stage1_end_frame},
            gpus=False,
        ))

    # ── Step 10: Merged recon setup ───────────────────────────────────────────
    if args.mode == "bundlesdf" and not args.skip_merged_setup:
        print("[pipeline] setting up merged reconstruction directory")
        _step("merged_setup", lambda: run_in_container(
            image=IMAGE_HOI,
            module="v2d_hoi_object_reconstruction.lib.setup_reconstruction_from_fp_poses",
            inputs={
                "poses_world":   poses_world,
                "frames_meta":   input_frames_meta,
                "sfm_keyframes": sfm_poses_meta,
                "depth_dir":     depth_dir,
                "masks_dir":     masks_dir,
                "left_dir":      os.path.join(job_dir, "left"),
                "right_dir":     os.path.join(job_dir, "right"),
            },
            outputs={"output_dir": merged_recon_dir},
            gpus=False,
        ))

    # ── Step 11: Full NeRF reconstruction ─────────────────────────────────────
    if args.mode == "bundlesdf" and not args.skip_full_nerf:
        print("[pipeline] running full NeRF reconstruction (both stages)")
        _step("full_nerf", lambda: run_in_container(
            image=IMAGE_BUNDLESDF,
            module="v2d_bundlesdf.lib.reconstruct",
            inputs=_nerf_inputs,
            outputs={"output_path": merged_recon_dir},
            extra_args=_bundlesdf_extra_args(),
            env={"CUDA_VISIBLE_DEVICES": str(fp_gpu), "NVIDIA_VISIBLE_DEVICES": str(fp_gpu)},
            gpus=True,
        ))

    # ── Step 12: FP tracking with final textured mesh ────────────────────────
    if args.mode == "bundlesdf" and not args.skip_final_fp_tracking:
        print("[pipeline] running FoundationPose tracking with final textured mesh")

        def _final_fp_tracking():
            final_fp_mesh, _ = _select_final_fp_mesh()
            _run_fp_tracking(
                video_path=os.path.join(job_dir, "video.mp4"),
                depth_folder=depth_dir,
                masks_folder=os.path.join(job_dir, "masks", "0"),
                camera_intrinsics_path=intrinsics_path,
                mesh_path=final_fp_mesh,
                poses_dir=poses_final_dir,
                weights_dir=fp_weights_dir,
                reference_frame=ref_frame,
            )

        _step("final_fp_tracking", _final_fp_tracking)

    # ── Step 13: FP render overlay with final textured mesh ───────────────────
    if args.mode == "bundlesdf" and not args.skip_final_fp_render:
        convert_poses_to_matrix(image=IMAGE_HOI, poses_dir=poses_final_dir)
        print("[pipeline] rendering FoundationPose overlay video with final textured mesh")

        def _final_fp_render():
            _, c_final_fp_mesh = _select_final_fp_mesh()
            _run_gpu(
                IMAGE_FOUNDATIONPOSE_RENDER,
                [
                    "python", "-m", "v2d.foundation_pose.lib.render_overlay",
                    "--video_path",             f"{c_job}/video.mp4",
                    "--poses_dir",              c_poses_final,
                    "--mesh_path",              c_final_fp_mesh,
                    "--camera_intrinsics_path", c_intrinsics,
                    "--output_dir",             f"{c_job}/fp_render_final",
                ],
                mounts=[(job_dir, "/data/job")],
                gpu_id=fp_gpu,
                user=f"{os.getuid()}:{os.getgid()}",
            )

        _step("final_fp_render", _final_fp_render)
        _step("final_fp_render_stitch", lambda: stitch_mp4(
            image=IMAGE_HOI,
            frames_dir=os.path.join(job_dir, "fp_render_final"),
            output_mp4=os.path.join(job_dir, "fp_render_final", "render.mp4")))

    # ══════════════════════════════════════════════════════════════════════════
    # SAM3D pipeline (steps S1–S4)
    # ══════════════════════════════════════════════════════════════════════════
    if args.mode == "sam3d":
        sam3d_dir = Path(job_dir) / "sam3d"
        sam3d_dir.mkdir(parents=True, exist_ok=True)

        # ── Step S1: Select representative frames ─────────────────────────────
        selected_frames: list[str] = []
        if not args.skip_select_frames:
            print(f"[pipeline] selecting SAM3D frames (bin_deg={args.sam3d_bin_deg}°)")
            t0 = time.time()
            selected_frames = select_sam3d_frames(
                image=IMAGE_HOI,
                job_dir=job_dir,
                bin_deg=args.sam3d_bin_deg,
            )
            _timings["select_frames"] = time.time() - t0
            print(f"[pipeline] select_frames done in {_timings['select_frames']:.1f}s "
                  f"→ {len(selected_frames)} frames: {selected_frames}")
        else:
            sel_path = sam3d_dir / "selected_frames.json"
            if sel_path.exists():
                selected_frames = json.loads(sel_path.read_text())
                print(f"[pipeline] skip_select_frames: loaded {len(selected_frames)} frames from {sel_path}")
            else:
                print("[warning] skip_select_frames but selected_frames.json not found; "
                      "SAM3D and SRT steps will have no frames to process", file=sys.stderr)
                sel_path.write_text("[]\n")

        # ── Step S2: SAM3D mesh reconstruction per frame ──────────────────────
        if not args.skip_sam3d:
            for frame_id in selected_frames:
                frame_out = sam3d_dir / frame_id
                frame_out.mkdir(parents=True, exist_ok=True)
                image_path  = os.path.join(job_dir, "left",    f"{frame_id}.jpg")
                mask_path_f = os.path.join(job_dir, "masks", "0", f"{frame_id}.png")
                print(f"[pipeline] SAM3D frame {frame_id}")
                t0 = time.time()
                _run_sam3d(
                    image_path=image_path,
                    mask_path=mask_path_f,
                    mesh_path=str(frame_out / "mesh.glb"),
                    transform_path=str(frame_out / "transform.json"),
                    intrinsics_path=str(frame_out / "intrinsics.json"),
                    weights_dir=str(_SAM3D_WEIGHTS_DIR),
                    seed=args.sam3d_seed,
                )
                elapsed = time.time() - t0
                _timings[f"sam3d_{frame_id}"] = elapsed
                print(f"[pipeline] SAM3D {frame_id} done in {elapsed:.1f}s")

        # ── Step S3: SRT scale estimation ─────────────────────────────────────
        if not args.skip_srt_scale:
            force_srt = args.sam3d_force_srt or not args.skip_sam3d
            srt_summary = run_sam3d_srt(
                image=IMAGE_HOI,
                job_dir=job_dir,
                use_depth=args.sam3d_use_depth,
                stage1_end_frame=stage1_end_frame,
                max_views=args.sam3d_srt_max_views,
                maxiter=args.sam3d_srt_maxiter,
                top_k=args.sam3d_srt_top_k,
                parallel=args.sam3d_srt_parallel,
                force=force_srt,
            )
            for outcome in srt_summary["outcomes"]:
                _timings[f"srt_{outcome['frame_id']}"] = float(outcome["elapsed"])

        # ── Step S4: Render debug images ──────────────────────────────────────
        # Uses the original SAM3D transform/intrinsics so the mesh renders back
        # onto the source image from the SAM3D viewpoint.
        if not args.skip_render_debug:
            for frame_id in selected_frames:
                frame_out = sam3d_dir / frame_id
                glb_path        = frame_out / "mesh.glb"
                transform_path  = frame_out / "transform.json"
                intrinsics_path = frame_out / "intrinsics.json"
                if not glb_path.exists() or not transform_path.exists():
                    continue
                image_path = os.path.join(job_dir, "left", f"{frame_id}.jpg")
                print(f"[pipeline] SAM3D render debug frame {frame_id}")
                t0 = time.time()
                _run_sam3d_render(
                    image_path=image_path,
                    mesh_path=str(glb_path),
                    transform_path=str(transform_path),
                    intrinsics_path=str(intrinsics_path),
                    output_image_path=str(frame_out / "render_debug.jpg"),
                )
                elapsed = time.time() - t0
                _timings[f"render_debug_{frame_id}"] = elapsed
                print(f"[pipeline] render_debug {frame_id} done in {elapsed:.1f}s")

        # ── Step S5: Stage-1 textured overlay video (pyrender inside v2d_sam3d) ──────
        if not args.skip_render_video:
            if stage1_end_frame is None:
                print("[pipeline] skip_render_video: stage1_end_frame unknown, cannot render Stage-1 video",
                      file=sys.stderr)
            else:
                c_job = "/data/job"
                for frame_id in selected_frames:
                    srt_out = sam3d_dir / frame_id / "srt"
                    if not (srt_out / "output_scaled.glb").exists():
                        print(f"[pipeline] render_video {frame_id}: srt output not found, skipping")
                        continue
                    frames_dir = sam3d_dir / frame_id / "render_video_frames"
                    video_path = sam3d_dir / frame_id / "render_video.mp4"
                    c_glb     = f"{c_job}/sam3d/{frame_id}/srt/output_scaled.glb"
                    c_out     = f"{c_job}/sam3d/{frame_id}/render_video_frames"
                    print(f"[pipeline] render_video {frame_id} (stage1_end_frame={stage1_end_frame})")
                    _step(f"render_video_{frame_id}", lambda _fid=frame_id, _cg=c_glb, _co=c_out: _run_gpu(
                        IMAGE_SAM3D,
                        [
                            "python", "-m", "v2d.sam3d.lib.render_textured_video",
                            "--job_dir",          c_job,
                            "--glb_path",         _cg,
                            "--output_dir",       _co,
                            "--stage1_end_frame", str(stage1_end_frame),
                        ],
                        mounts=[(job_dir, "/data/job")],
                        user=f"{os.getuid()}:{os.getgid()}",
                    ))
                    _step(
                        f"render_video_stitch_{frame_id}",
                        lambda _fd=frames_dir, _vp=video_path: stitch_mp4(
                            image=IMAGE_HOI,
                            frames_dir=_fd,
                            output_mp4=_vp,
                        ),
                    )

        # ── Step S6: Suggested best mesh selection ───────────────────────────
        if not args.skip_select_best_mesh:
            print("[pipeline] selecting suggested best SAM3D mesh")
            t0 = time.time()
            try:
                best_summary = select_best_sam3d(
                    image=IMAGE_HOI,
                    job_dir=job_dir,
                )
            except ValueError as exc:
                if args.skip_srt_scale:
                    print(f"[pipeline] select_best skipped: {exc}", file=sys.stderr)
                else:
                    raise
            else:
                elapsed = time.time() - t0
                _timings["sam3d_select_best"] = elapsed
                print(
                    f"[pipeline] suggested best SAM3D frame {best_summary['best_frame']} "
                    f"saved to {sam3d_dir / 'best'} in {elapsed:.1f}s"
                )

    total = time.time() - _t_total
    print("\n[pipeline] ── Timing summary ──────────────────────────")
    for step, elapsed in _timings.items():
        print(f"[pipeline]   {step:<20s}: {elapsed:7.1f}s")
    print(f"[pipeline]   {'TOTAL':<20s}: {total:7.1f}s")
    print("[pipeline] all done!")


if __name__ == "__main__":
    main()
