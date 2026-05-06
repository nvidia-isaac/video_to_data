"""
End-to-end ego hand + object reconstruction pipeline from a single video.

Given an MP4 and a text prompt describing the held object, this script runs:
  1.  Ego hand reconstruction (ViPE + Dyn-HaMR)  [runs first — no prior deps]
  2.  Convert DynHaMR EXR depth → depth_vipe/ + intrinsics_vipe.json
  3.  Frame extraction
  4.  MoGe monocular depth + intrinsics
  5.  Grounding DINO object detection (reference frame only)
  6.  SAM2 object mask tracking (full video)
  7.  SAM3D textured mesh generation  (uses --depth_source depth)
  8.  FoundationPose scale estimation  (uses --depth_source depth)
  9.  FoundationPose 6-DOF tracking    (uses --depth_source depth)
  10. EKF smoothing of object poses    (uses --depth_source intrinsics)
  11. Hand/object depth alignment      (uses --depth_source depth + intrinsics)

The --depth_source flag (moge | vipe) selects which depth map and intrinsics
are fed to SAM3D, FoundationPose, EKF, and the hand-alignment step.
Both depth sources are always computed so results can be compared.

Output directory layout:
  <output_dir>/
  ├── frames/                      # Extracted video frames
  ├── depth/                       # MoGe depth PNGs
  ├── depth_vipe/                  # DynHaMR depth PNGs (converted from EXR)
  ├── intrinsics/                  # Per-frame MoGe intrinsics JSONs
  ├── intrinsics_stable.json       # Temporally stabilised MoGe intrinsics
  ├── intrinsics_vipe.json         # DynHaMR camera intrinsics
  ├── dino_detections.json         # Grounding DINO bboxes (frame 0)
  ├── sam2_prompts.json            # SAM2 prompt file
  ├── masks/1/                     # Per-frame object masks
  ├── hand_reconstruction/         # Dyn-HaMR + ViPE outputs
  │   ├── MANO_RIGHT.pkl
  │   ├── BMC/
  │   └── logs/                    # world_results.npz lives here
  │
  │   The following are created once per --depth_source (ds = moge | vipe):
  ├── mesh_{ds}/
  │   ├── textured_mesh.obj        # SAM3D textured mesh
  │   ├── mesh_transform.json      # SAM3D scale/pose transform
  │   └── mesh_intrinsics.json     # SAM3D-estimated intrinsics
  ├── mesh_pretransformed_{ds}.obj # SAM3D mesh with rotation+scale applied
  ├── mesh_scaled_{ds}.obj         # Depth-aligned, scale-corrected mesh
  ├── scale_{ds}.json              # Estimated mesh scale factor
  ├── poses_{ds}/                  # Raw FoundationPose per-frame JSONs
  ├── poses_smoothed_{ds}/         # EKF-smoothed per-frame pose JSONs
  ├── poses_smoothed_render_{ds}/  # FoundationPose overlay video
  ├── world_results_aligned_{ds}.npz  # Final depth-aligned hand + object poses
  ├── render_aligned_{ds}.mp4      # 2×2 grid render using trans_aligned
  └── render_unaligned_{ds}.mp4    # 2×2 grid render using trans (for comparison)

Usage:
    python modules/v2d_pipelines/run_v2d_ego_e2e.py \\
        --video_path data/my_video.mp4 \\
        --prompt "blue cup" \\
        --output_dir data/outputs/my_video \\
        --depth_source vipe \\
        --moge_weights             data/weights/moge \\
        --grounding_dino_weights   data/weights/grounding_dino \\
        --sam2_weights             data/weights/sam2 \\
        --sam3d_weights            data/weights/sam3d \\
        --foundation_pose_weights  data/weights/foundation_pose \\
        --hand_reconstruction_weights data/weights/hand

Run from reconstruction/.
"""

import argparse
import glob
import json
import os
import zipfile

import numpy as np
import trimesh

from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.hand_alignment.docker.run_align_world_results import run_align_world_results
from v2d.hand_alignment.docker.run_render_dynhamr_video import run_render_dynhamr_video
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d_ego_hand_reconstruction.docker.run_reconstruction import run_reconstruction


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end ego hand + object reconstruction from a single video."
    )
    p.add_argument("--video_path",   required=True,
                   help="Input MP4 video.")
    p.add_argument("--prompt",       required=True,
                   help="Grounding DINO text prompt for the held object (e.g. 'blue cup').")
    p.add_argument("--output_dir",   required=True,
                   help="Root output directory.")

    # Weight paths (all default to data/weights/<model> relative to cwd)
    p.add_argument("--moge_weights",            default="data/weights/moge",
                   help="MoGe model weights directory. (default: data/weights/moge)")
    p.add_argument("--grounding_dino_weights",  default="data/weights/grounding_dino",
                   help="Grounding DINO weights directory. (default: data/weights/grounding_dino)")
    p.add_argument("--sam2_weights",            default="data/weights/sam2",
                   help="SAM2 weights directory. (default: data/weights/sam2)")
    p.add_argument("--sam3d_weights",           default="data/weights/sam3d",
                   help="SAM3D weights directory. (default: data/weights/sam3d)")
    p.add_argument("--foundation_pose_weights", default="data/weights/foundation_pose",
                   help="FoundationPose weights directory. (default: data/weights/foundation_pose)")
    p.add_argument("--hand_reconstruction_weights", default="data/weights/hand",
                   help="Ego hand reconstruction weights directory containing MANO_RIGHT.pkl "
                        "and BMC/. (default: data/weights/hand)")
    p.add_argument("--mano_weights", default=None,
                   help="MANO model directory for hand alignment. "
                        "(default: --hand_reconstruction_weights)")

    # Optional tuning
    p.add_argument("--depth_source", choices=["moge", "vipe"], default="moge",
                   help="Depth source for SAM3D, scale estimation, FP tracking, and hand "
                        "alignment. Both depth sources are always computed. (default: moge)")
    p.add_argument("--reference_frame", type=int, default=0,
                   help="Frame used for DINO detection, SAM3D, and FP registration (default: 0).")
    p.add_argument("--smooth_sigma",    type=float, default=5.0,
                   help="Gaussian sigma (frames) for hand translation smoothing (default: 5.0).")
    p.add_argument("--reregister_iou_thresh", type=float, default=0.3,
                   help="IoU threshold below which FoundationPose re-registers from scratch. "
                        "Set to 0 to disable. (default: 0.3)")
    p.add_argument("--dev", action="store_true",
                   help="Mount local module source into containers (live-edit mode).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


def _step(label: str, done: bool) -> bool:
    if done:
        print(f"  [skip] {label}")
        return True
    print(f"  [run ] {label}")
    return False


def _find_world_results(hand_recon_dir: str) -> str | None:
    """Return the final smooth_fit world_results.npz, or None if not found.

    DynHaMR writes many intermediate checkpoints under smooth_fit/, root_fit/,
    init/, and hamer/.  The canonical final output is the highest-numbered file
    under smooth_fit/.
    """
    candidates = glob.glob(
        f"{hand_recon_dir}/logs/**/smooth_fit/*_world_results.npz", recursive=True
    )
    if not candidates:
        return None
    return max(candidates)


def _apply_sam3d_transform(mesh_path: str, transform_path: str, out_path: str) -> None:
    """Apply the SAM3D rotation+scale (not translation) to mesh vertices and save.

    SAM3D outputs the mesh in a local normalized space. The mesh_transform.json
    carries the rotation R and uniform scale s needed to bring it to metric
    camera space.  We bake R*s into the vertices so FoundationPose sees a
    properly-scaled mesh (translation is omitted — FP estimates it during
    registration).
    """
    with open(transform_path) as f:
        t = json.load(f)

    qw, qx, qy, qz = t["rotation"]
    sx, sy, sz      = t["scale"]
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qw*qz,      2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,      1 - 2*qx*qx - 2*qz*qz,  2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,      2*qy*qz + 2*qw*qx,      1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    RS = R @ np.diag([sx, sy, sz])

    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Scene):
        meshes = list(scene.geometry.values())
        for m in meshes:
            m.vertices = (RS @ m.vertices.T).T
        result = trimesh.util.concatenate(meshes)
    else:
        scene.vertices = (RS @ scene.vertices.T).T
        result = scene

    result.export(out_path)


def _convert_dynhamr_depth(
    hand_recon_dir: str,
    out_dir: str,
    world_results_path: str,
    intrinsics_out_path: str,
) -> None:
    """Convert DynHaMR EXR depth zip → inverse-depth uint16 PNGs + intrinsics JSON.

    DynHaMR writes depth/video.zip containing per-frame EXR files with a
    single Z channel in metric metres.  Converts to the pipeline's standard
    inverse-depth encoding: pixel = uint16(65535 / (depth_m + 1)).

    Intrinsics are read from world_results.npz ('intrins' field = [fx,fy,cx,cy]);
    image dimensions are taken from the first EXR frame.

    EXR names use 5-digit indices (00000.exr); output PNGs use 6-digit
    indices (000000.png) to match the rest of the pipeline.
    """
    import OpenEXR
    import Imath
    import tempfile as _tempfile
    from PIL import Image as _Image

    depth_subdir = os.path.join(hand_recon_dir, "depth")
    zip_files = glob.glob(os.path.join(depth_subdir, "*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No depth zip found in {depth_subdir}")
    zip_path = zip_files[0]

    os.makedirs(out_dir, exist_ok=True)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    def _read_exr(raw_bytes):
        with _tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            exr = OpenEXR.InputFile(tmp_path)
            dw  = exr.header()["dataWindow"]
            w   = dw.max.x - dw.min.x + 1
            h   = dw.max.y - dw.min.y + 1
            buf = exr.channel("Z", pixel_type)
            return np.frombuffer(buf, dtype=np.float32).reshape(h, w), w, h
        finally:
            os.unlink(tmp_path)

    with zipfile.ZipFile(zip_path) as zf:
        exr_names = sorted(n for n in zf.namelist() if n.endswith(".exr"))
        print(f"  Converting {len(exr_names)} EXR frames → {out_dir}")

        # Read image dimensions from first frame
        _, img_w, img_h = _read_exr(zf.read(exr_names[0]))

        for name in exr_names:
            frame   = int(os.path.splitext(name)[0])
            out_png = os.path.join(out_dir, f"{frame:06d}.png")
            depth, _, _ = _read_exr(zf.read(name))
            inv = (65535.0 / (depth.astype(np.float64) + 1.0)).clip(0, 65535)
            _Image.fromarray(inv.astype(np.uint16)).save(out_png)

    # Write intrinsics JSON from world_results.npz
    wr = np.load(world_results_path, allow_pickle=True)
    fx, fy, cx, cy = [float(x) for x in wr['intrins']]
    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                  "width": img_w, "height": img_h}
    with open(intrinsics_out_path, "w") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"  Intrinsics → {intrinsics_out_path}  "
          f"(fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} {img_w}×{img_h})")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_v2d_ego_e2e(
    video_path: str,
    prompt: str,
    output_dir: str,
    moge_weights: str,
    grounding_dino_weights: str,
    sam2_weights: str,
    sam3d_weights: str,
    foundation_pose_weights: str,
    hand_reconstruction_weights: str,
    mano_weights: str | None = None,
    depth_source: str = "moge",
    reference_frame: int = 0,
    smooth_sigma: float = 5.0,
    reregister_iou_thresh: float | None = 0.3,
    dev: bool = False,
) -> None:
    if depth_source not in ("moge", "vipe"):
        raise ValueError(f"depth_source must be 'moge' or 'vipe', got {depth_source!r}")

    video_path   = os.path.abspath(video_path)
    output_dir   = os.path.abspath(output_dir)
    mano_weights = os.path.abspath(mano_weights or hand_reconstruction_weights)
    os.makedirs(output_dir, exist_ok=True)

    OBJECT_ID = 1

    # -- Depth-source-independent output paths --------------------------------
    frames_dir        = f"{output_dir}/frames"
    depth_dir         = f"{output_dir}/depth"
    intrinsics_dir    = f"{output_dir}/intrinsics"
    intrinsics_stable = f"{output_dir}/intrinsics_stable.json"
    depth_vipe_dir    = f"{output_dir}/depth_vipe"
    intrinsics_vipe   = f"{output_dir}/intrinsics_vipe.json"
    hand_recon_dir    = f"{output_dir}/hand_reconstruction"
    dino_detections   = f"{output_dir}/dino_detections.json"
    sam2_prompts      = f"{output_dir}/sam2_prompts.json"
    masks_dir         = f"{output_dir}/masks"

    # -- Active depth source --------------------------------------------------
    if depth_source == "vipe":
        active_depth_dir  = depth_vipe_dir
        active_intrinsics = intrinsics_vipe
    else:
        active_depth_dir  = depth_dir
        active_intrinsics = intrinsics_stable

    # -- Depth-source-dependent output paths (suffixed) -----------------------
    ds                      = depth_source
    mesh_dir                = f"{output_dir}/mesh_{ds}"
    mesh_path               = f"{mesh_dir}/textured_mesh.obj"
    mesh_transform          = f"{mesh_dir}/mesh_transform.json"
    mesh_intrinsics         = f"{mesh_dir}/mesh_intrinsics.json"
    mesh_pretransformed     = f"{output_dir}/mesh_pretransformed_{ds}.obj"
    mesh_scaled             = f"{output_dir}/mesh_scaled_{ds}.obj"
    scale_path              = f"{output_dir}/scale_{ds}.json"
    poses_dir               = f"{output_dir}/poses_{ds}"
    poses_smooth_dir        = f"{output_dir}/poses_smoothed_{ds}"
    poses_smooth_render_dir = f"{output_dir}/poses_smoothed_render_{ds}"
    world_aligned           = f"{output_dir}/world_results_aligned_{ds}.npz"
    render_aligned          = f"{output_dir}/render_aligned_{ds}.mp4"
    render_unaligned        = f"{output_dir}/render_unaligned_{ds}.mp4"

    ref_rgb   = f"{frames_dir}/{reference_frame:06d}.png"
    ref_depth = f"{active_depth_dir}/{reference_frame:06d}.png"
    ref_mask  = f"{masks_dir}/{OBJECT_ID}/{reference_frame:06d}.png"

    print(f"\n{'='*60}")
    print(f"  video        : {os.path.basename(video_path)}")
    print(f"  prompt       : {prompt!r}")
    print(f"  output       : {output_dir}")
    print(f"  depth_source : {depth_source}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Step 1: Ego hand reconstruction (ViPE + Dyn-HaMR)
    # Runs first — only requires video_path, no depth or frame deps.
    # -----------------------------------------------------------------------
    world_results_npz = _find_world_results(hand_recon_dir)
    if not _step("Ego hand reconstruction (ViPE + Dyn-HaMR)",
                 world_results_npz is not None):
        run_reconstruction(
            video_input = video_path,
            output_dir  = hand_recon_dir,
            weights_dir = hand_reconstruction_weights,
        )
        world_results_npz = _find_world_results(hand_recon_dir)
        if world_results_npz is None:
            raise RuntimeError(
                "No smooth_fit world_results.npz found after reconstruction — "
                f"check {hand_recon_dir}/logs/"
            )

    print(f"  world_results: {world_results_npz}")

    # -----------------------------------------------------------------------
    # Step 2: Convert DynHaMR EXR depth → inverse-depth PNGs (depth_vipe)
    # -----------------------------------------------------------------------
    if not _step("Convert DynHaMR depth (EXR → uint16 PNG)", _has_files(depth_vipe_dir)):
        _convert_dynhamr_depth(hand_recon_dir, depth_vipe_dir,
                               world_results_npz, intrinsics_vipe)

    # -----------------------------------------------------------------------
    # Step 3: Extract frames
    # -----------------------------------------------------------------------
    if not _step("Extract frames", _has_files(frames_dir)):
        extract_images(video_path, frames_dir)

    # -----------------------------------------------------------------------
    # Step 4: MoGe depth + intrinsics
    # -----------------------------------------------------------------------
    if not _step("MoGe depth + intrinsics", _has_files(depth_dir)):
        run_moge_depth(
            video_path        = video_path,
            depth_folder      = depth_dir,
            intrinsics_folder = intrinsics_dir,
            weights_path      = moge_weights,
            dev               = dev,
        )

    if not _step("Stabilise intrinsics", os.path.exists(intrinsics_stable)):
        stabilize_intrinsics(intrinsics_dir, intrinsics_stable)

    # -----------------------------------------------------------------------
    # Step 5: Grounding DINO → SAM2 prompts
    # -----------------------------------------------------------------------
    if not _step("Grounding DINO (frame 0)", os.path.exists(dino_detections)):
        run_image_to_object_bboxes(
            image_path  = ref_rgb,
            output_path = dino_detections,
            prompt      = prompt,
            model_dir   = grounding_dino_weights,
            dev         = False,
        )

    if not os.path.exists(sam2_prompts):
        with open(dino_detections) as f:
            detections = json.load(f)
        if not detections:
            raise RuntimeError(
                f"Grounding DINO found no objects — check prompt: {prompt!r}"
            )
        box = BoundingBox.from_dict(detections[0]["box"])
        prompts = Sam2Prompts(
            prompts=[Sam2Prompt(frame_index=reference_frame, object_id=OBJECT_ID, box=box)]
        )
        with open(sam2_prompts, "w") as f:
            json.dump(prompts.to_dict(), f, indent=2)

    # -----------------------------------------------------------------------
    # Step 6: SAM2 object mask tracking
    # -----------------------------------------------------------------------
    if not _step("SAM2 mask tracking", _has_files(f"{masks_dir}/{OBJECT_ID}")):
        run_video_to_masks(
            video_path  = video_path,
            prompts_path= sam2_prompts,
            masks_dir   = masks_dir,
            weights_dir = sam2_weights,
            dev         = dev,
        )

    # -----------------------------------------------------------------------
    # Step 7: SAM3D textured mesh generation
    # Uses: active_depth_dir, active_intrinsics
    # -----------------------------------------------------------------------
    os.makedirs(mesh_dir, exist_ok=True)
    if not _step(f"SAM3D mesh generation ({ds} depth)", os.path.exists(mesh_path)):
        run_image_to_mesh(
            image_path            = ref_rgb,
            mask_path             = ref_mask,
            mesh_path             = mesh_path,
            transform_path        = mesh_transform,
            intrinsics_path       = mesh_intrinsics,
            weights_dir           = sam3d_weights,
            with_texture_baking   = True,
            with_mesh_postprocess = True,
            depth_path            = ref_depth,
            depth_intrinsics_path = active_intrinsics,
            depth_mask_path       = ref_mask,
            dev                   = dev,
        )

    # -----------------------------------------------------------------------
    # Step 7b: Apply SAM3D transform (rotation + scale) to mesh
    # -----------------------------------------------------------------------
    if not _step(f"Apply SAM3D transform ({ds})", os.path.exists(mesh_pretransformed)):
        _apply_sam3d_transform(mesh_path, mesh_transform, mesh_pretransformed)

    # -----------------------------------------------------------------------
    # Step 8: FoundationPose scale estimation
    # Uses: active_depth_dir, active_intrinsics
    # -----------------------------------------------------------------------
    if not _step(f"Scale estimation ({ds} depth)", os.path.exists(mesh_scaled)):
        run_estimate_mesh_scale(
            mesh_path             = mesh_pretransformed,
            rgb_path              = ref_rgb,
            depth_path            = ref_depth,
            mask_path             = ref_mask,
            intrinsics_path       = active_intrinsics,
            weights_dir           = foundation_pose_weights,
            scale_path            = scale_path,
            rescaled_mesh_path    = mesh_scaled,
            lo                    = 0.5,
            hi                    = 2.0,
            n_samples             = 9,
            n_levels              = 4,
            iou_weight            = 1.0,
            depth_weight          = 1.0,
            registration_iterations = 5,
            dev                   = dev,
        )

    # -----------------------------------------------------------------------
    # Step 9: FoundationPose 6-DOF tracking
    # Uses: active_depth_dir, active_intrinsics
    # -----------------------------------------------------------------------
    if not _step(f"FoundationPose tracking ({ds} depth)", _has_files(poses_dir)):
        run_video_to_poses(
            video_path             = video_path,
            depth_folder           = active_depth_dir,
            masks_folder           = f"{masks_dir}/{OBJECT_ID}",
            camera_intrinsics_path = active_intrinsics,
            mesh_path              = mesh_scaled,
            poses_dir              = poses_dir,
            weights_dir            = foundation_pose_weights,
            reference_frame        = reference_frame,
            mask_depth             = True,
            reregister_iou_thresh  = reregister_iou_thresh if reregister_iou_thresh else None,
            dev                    = dev,
        )

    # -----------------------------------------------------------------------
    # Step 10: EKF smoothing
    # Uses: active_intrinsics
    # -----------------------------------------------------------------------
    if not _step(f"EKF pose smoothing ({ds})", _has_files(poses_smooth_dir)):
        run_ekf_smoothing(
            poses_dir            = poses_dir,
            mesh_path            = mesh_scaled,
            intrinsics_path      = active_intrinsics,
            weights_dir          = foundation_pose_weights,
            output_dir           = poses_smooth_dir,
            masks_folder         = f"{masks_dir}/{OBJECT_ID}",
            process_noise_xy     = 0.01,
            process_noise_z      = 0.01,
            process_noise_r      = 0.02,
            measurement_noise_xy = 0.01,
            measurement_noise_z  = 0.04,
            measurement_noise_r  = 0.02,
            dev                  = dev,
        )

    # -----------------------------------------------------------------------
    # Step 10b: Render smoothed poses (FoundationPose overlay video)
    # -----------------------------------------------------------------------
    if not _step(f"Render smoothed poses ({ds})", _has_files(poses_smooth_render_dir)):
        run_render_poses(
            mesh_path       = mesh_scaled,
            poses_dir       = poses_smooth_dir,
            frames_dir      = frames_dir,
            intrinsics_path = active_intrinsics,
            output_dir      = poses_smooth_render_dir,
            dev             = dev,
        )

    # -----------------------------------------------------------------------
    # Step 11: Hand/object depth alignment
    # Uses: active_depth_dir, active_intrinsics
    # -----------------------------------------------------------------------
    if not _step(f"Hand/object depth alignment ({ds})", os.path.exists(world_aligned)):
        run_align_world_results(
            input_hand_data  = world_results_npz,
            depth_dir        = active_depth_dir,
            depth_intrinsics = active_intrinsics,
            mano_model_dir   = mano_weights,
            output_hand_data = world_aligned,
            object_masks_dir = f"{masks_dir}/{OBJECT_ID}",
            object_poses_dir = poses_smooth_dir,
            smooth_sigma     = smooth_sigma,
            dev              = dev,
        )

    # -----------------------------------------------------------------------
    # Step 12: Render aligned result (trans_aligned) — 2×2 grid video
    # -----------------------------------------------------------------------
    if not _step(f"Render aligned (trans_aligned, {ds})", os.path.exists(render_aligned)):
        run_render_dynhamr_video(
            world_results_path = world_aligned,
            frames_folder      = frames_dir,
            mano_assets_root   = mano_weights,
            output_path        = render_aligned,
            use_trans_aligned  = True,
            object_mesh_path   = mesh_scaled,
            object_poses_dir   = poses_smooth_dir,
            intrinsics_path    = active_intrinsics,
            dev                = dev,
        )

    # -----------------------------------------------------------------------
    # Step 13: Render unaligned result (trans) for comparison
    # Raw DynHaMR `trans` was optimized under ViPE intrinsics, so we must
    # project it through ViPE intrinsics regardless of --depth_source.
    # -----------------------------------------------------------------------
    if not _step(f"Render unaligned (trans, {ds})", os.path.exists(render_unaligned)):
        run_render_dynhamr_video(
            world_results_path = world_aligned,
            frames_folder      = frames_dir,
            mano_assets_root   = mano_weights,
            output_path        = render_unaligned,
            use_trans_aligned  = False,
            object_mesh_path   = mesh_scaled,
            object_poses_dir   = poses_smooth_dir,
            intrinsics_path    = intrinsics_vipe,
            dev                = dev,
        )

    print(f"\n{'='*60}")
    print(f"  Done!  (depth_source={depth_source})")
    print(f"  Aligned hand + object: {world_aligned}")
    print(f"  Scaled mesh:           {mesh_scaled}")
    print(f"  Smoothed poses:        {poses_smooth_dir}/")
    print(f"  Render (aligned):      {render_aligned}")
    print(f"  Render (unaligned):    {render_unaligned}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_v2d_ego_e2e(
        video_path                  = args.video_path,
        prompt                      = args.prompt,
        output_dir                  = args.output_dir,
        moge_weights                = args.moge_weights,
        grounding_dino_weights      = args.grounding_dino_weights,
        sam2_weights                = args.sam2_weights,
        sam3d_weights               = args.sam3d_weights,
        foundation_pose_weights     = args.foundation_pose_weights,
        hand_reconstruction_weights = args.hand_reconstruction_weights,
        mano_weights                = args.mano_weights,
        depth_source                = args.depth_source,
        reference_frame             = args.reference_frame,
        smooth_sigma                = args.smooth_sigma,
        reregister_iou_thresh       = args.reregister_iou_thresh,
        dev                         = args.dev,
    )
