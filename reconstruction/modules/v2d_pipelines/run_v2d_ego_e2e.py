"""
End-to-end ego hand + object reconstruction pipeline from a single video.

Given an MP4 and a text prompt describing the held object, this script runs:
  0.  AnyCalib undistortion              [optional, --undistort]
  1.  Ego hand reconstruction (ViPE + Dyn-HaMR)
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
  12. DROID-SLAM trajectory            [optional, --gsplat]
  13. 2nd SAM2 pass (object + hands)   [optional, --gsplat]
  14. Joint hand+object gsplat refine  [optional, --gsplat]

Step 0 (--undistort) calibrates the input video with AnyCalib, undistorts
it to a pinhole stream, and rebinds video_path to the undistorted MP4 for
all downstream steps. Recommended for fisheye / wide-angle footage so
ViPE, MoGe, FoundationPose, and the renderers all see a consistent
pinhole camera model.

The --depth_source flag (moge | vipe) selects which depth map and intrinsics
are fed to SAM3D, FoundationPose, EKF, and the hand-alignment step.
Both depth sources are always computed so results can be compared.

Output directory layout:
  <output_dir>/
  ├── anycalib/                    # [if --undistort] AnyCalib outputs:
  │   ├── intrinsics.json          #   estimated intrinsics of input video
  │   ├── distortion.json          #   estimated distortion params
  │   ├── undistorted.mp4          #   undistorted (pinhole) video
  │   └── undistorted_intrinsics.json
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
  ├── dynhamr_as_hamer_tracks_{ds}/    # Per-frame DynHaMR poses re-expressed as
  │                                    #   v2d_hamer-style detections (cam frame)
  ├── hamer_aligned_from_dynhamr_{ds}/ # Per-frame depth-aligned hand records
  │                                    #   (cam_t, intrinsics, hand_scale,
  │                                    #   diagnostics.cam_t_pre_dz)
  ├── render_aligned_{ds}.mp4      # 2×2 grid render of depth-aligned hands
  ├── render_unaligned_{ds}.mp4    # Same renderer, projects diagnostics.cam_t_pre_dz
  │
  │   The following are populated only when --gsplat is set:
  ├── slam_poses_{ds}/             # DROID-SLAM cam-to-world Transform3d JSONs
  ├── slam_trajectory_{ds}.txt     # TUM-format SLAM trajectory
  ├── hand_silhouettes_{ds}/{2,3}/<frame:06d>.png  # rendered MANO masks
  ├── sam2_prompts_gsplat_{ds}.json # 2nd-pass SAM2 prompts (obj bbox + hand masks)
  ├── sam2_prompts_gsplat_overlay_{ds}/<frame:06d>.png  # debug: prompt boxes drawn
  ├── masks_gsplat_{ds}/{1,2,3}/   # 2nd SAM2 pass: 1=object, 2=left hand, 3=right
  ├── poses_refined_{ds}/          # Gsplat-refined object Transform3d JSONs
  ├── hamer_aligned_refined_{ds}/{2,3}/ # Gsplat-refined hand pose tracks
  ├── refined_overlay_{ds}.mp4     # Gsplat-rendered overlay of refined hand+object
  ├── render_refined_{ds}.mp4      # Mesh-rendered overlay of refined hand+object
  │                                #   (parallel to render_aligned_{ds}.mp4)
  ├── refined_object_scale_{ds}.json    # Learned multiplicative object scale
  │
  │   Final packaged output (always produced):
  └── result/
      ├── result.npz   # Consolidated time-aligned camera + object + hands
      ├── mesh.obj     # Textured object mesh (= mesh_scaled_{ds}.obj)
      ├── material.mtl
      └── material_0.png  # (or whatever textures the .mtl references)

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
import shutil
import zipfile

import numpy as np
import trimesh

from v2d.anycalib.docker.run_video_to_calibration import run_video_to_calibration
from v2d.common.datatypes import BoundingBox, CameraIntrinsics, Sam2Prompt, Sam2Prompts, Transform3d
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.droid_slam.docker.run_video_to_slam import run_video_to_slam
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.gsplat_refinement.docker.run_refine import run_refine
from v2d.hand_alignment.docker.run_dynhamr_to_hamer_tracks import run_dynhamr_to_hamer_tracks
from v2d.hamer.docker.run_align_hands import run_align_hands
from v2d.hamer.docker.run_render_hands_aligned_video import run_render_hands_aligned_video
from v2d.hamer.docker.run_tracks_to_masks import run_tracks_to_masks
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d_ego_hand_reconstruction.docker.run_reconstruction import run_reconstruction

# Track IDs used by the gsplat-refinement SAM2 pass. Object is track 1
# (matches OBJECT_ID below); hand tracks 2/3 mirror run_dynhamr_to_hamer_tracks'
# defaults so dynhamr_as_hamer_tracks_{ds}/{2,3} line up with the new mask dirs.
_LEFT_HAND_ID  = 2
_RIGHT_HAND_ID = 3


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
    p.add_argument("--anycalib_weights", default="data/weights/anycalib",
                   help="AnyCalib weights directory (used only with --undistort). "
                        "(default: data/weights/anycalib)")
    p.add_argument("--droid_slam_weights", default="data/weights/droid_slam",
                   help="DROID-SLAM weights directory (used only with --gsplat). "
                        "(default: data/weights/droid_slam)")

    # Optional tuning
    p.add_argument("--undistort", action="store_true",
                   help="Run AnyCalib first to estimate intrinsics + distortion and "
                        "undistort the video before all other steps. Recommended for "
                        "fisheye / wide-angle footage.")
    p.add_argument("--depth_source", choices=["moge", "vipe"], default="moge",
                   help="Depth source for SAM3D, scale estimation, FP tracking, and hand "
                        "alignment. Both depth sources are always computed. (default: moge)")
    p.add_argument("--reference_frame", type=int, default=0,
                   help="Frame used for DINO detection, SAM3D, and FP registration (default: 0).")
    p.add_argument("--reregister_iou_thresh", type=float, default=0.3,
                   help="IoU threshold below which FoundationPose re-registers from scratch. "
                        "Set to 0 to disable. (default: 0.3)")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed for SAM3D mesh generation. Pinning this "
                        "(default 0) makes the SAM3D-produced mesh "
                        "deterministic across runs. Set to a negative number "
                        "to disable seeding (each run produces a fresh mesh).")

    # Gsplat refinement (DROID-SLAM + 2nd SAM2 pass with hand bbox prompts
    # derived from dynhamr_as_hamer_tracks → joint hand+object gsplat).
    p.add_argument("--gsplat", action="store_true",
                   help="Enable joint hand+object Gaussian-splat refinement. "
                        "Runs DROID-SLAM for background camera poses, a 2nd "
                        "SAM2 pass seeded with the DynHaMR-projected hand "
                        "bboxes (plus the DINO object bbox), then gsplat "
                        "refinement against the result.")
    p.add_argument("--gsplat_min_mask_pixels", type=int, default=256,
                   help="Minimum foreground pixels for a rendered MANO "
                        "silhouette to be eligible as a SAM2 mask prompt. "
                        "(default: 256)")
    p.add_argument("--gsplat_resume", action="store_true",
                   help="Resume gsplat from refine_checkpoint_{ds}.pt if present.")
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


_PROMPT_COLORS = {
    1: (255,  60,  60),   # object   — red
    _LEFT_HAND_ID:  (60, 160, 255),   # left  — blue
    _RIGHT_HAND_ID: (60, 220,  60),   # right — green
}


def _render_gsplat_prompts_overlay(
    prompts_path: str,
    frames_dir: str,
    output_dir: str,
) -> None:
    """Draw each SAM2 prompt on its target frame for visual debug.

    Box prompts → rectangle outline. Mask prompts → translucent color fill
    (alpha-blended). Colors: object=red, left=blue, right=green.
    """
    from PIL import Image, ImageDraw, ImageFont

    with open(prompts_path) as f:
        prompts = Sam2Prompts.from_dict(json.load(f)).prompts
    by_frame: dict[int, list[Sam2Prompt]] = {}
    for p in prompts:
        by_frame.setdefault(int(p.frame_index), []).append(p)

    os.makedirs(output_dir, exist_ok=True)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    n_written = 0
    for frame_idx, frame_prompts in sorted(by_frame.items()):
        frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.png")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
            if not os.path.exists(frame_path):
                continue
        frame_np = np.asarray(Image.open(frame_path).convert("RGB"),
                              dtype=np.float32)
        for p in frame_prompts:
            color = _PROMPT_COLORS.get(int(p.object_id), (255, 255, 0))
            if p.mask_path is not None and os.path.exists(p.mask_path):
                m = np.asarray(Image.open(p.mask_path)) > 0
                if m.ndim == 3:
                    m = m[..., 0]
                if m.shape == frame_np.shape[:2]:
                    overlay = np.asarray(color, dtype=np.float32)[None, None, :]
                    alpha = 0.5 * m[..., None].astype(np.float32)
                    frame_np = frame_np * (1 - alpha) + overlay * alpha
        img = Image.fromarray(frame_np.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        for p in frame_prompts:
            color = _PROMPT_COLORS.get(int(p.object_id), (255, 255, 0))
            side = ("obj" if p.object_id == 1
                    else "L" if p.object_id == _LEFT_HAND_ID
                    else "R" if p.object_id == _RIGHT_HAND_ID
                    else "?")
            label = f"id={p.object_id} {side}"
            if p.box is not None:
                x0, y0, x1, y1 = p.box.x0, p.box.y0, p.box.x1, p.box.y1
                draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
                anchor_x, anchor_y = x0, y0
            elif p.mask_path is not None:
                # Anchor the label at the mask's top-left bounding pixel
                # for legibility.
                m = np.asarray(Image.open(p.mask_path)) > 0
                if m.ndim == 3:
                    m = m[..., 0]
                ys, xs = np.nonzero(m)
                if xs.size == 0:
                    continue
                anchor_x, anchor_y = float(xs.min()), float(ys.min())
            else:
                continue
            tw = draw.textlength(label, font=font)
            draw.rectangle(
                [anchor_x, max(0, anchor_y - 26),
                 anchor_x + tw + 8, anchor_y],
                fill=color,
            )
            draw.text(
                (anchor_x + 4, max(0, anchor_y - 24)),
                label, fill=(0, 0, 0), font=font,
            )
        img.save(os.path.join(output_dir, f"{frame_idx:06d}.png"))
        n_written += 1
    print(f"  Wrote {n_written} prompt-overlay frame(s) → {output_dir}/")


def _gsplat_hand_prompts(
    silhouettes_dir: str,
    frame_index: int,
    min_mask_pixels: int = 256,
) -> list[Sam2Prompt]:
    """Build a single SAM2 mask prompt per hand at the given frame.

    Reads ``<silhouettes_dir>/{2,3}/<frame_index:06d>.png`` and emits one
    mask-style Sam2Prompt per hand whose mask has at least
    ``min_mask_pixels`` foreground pixels at that frame. Multi-frame
    seeding tends to drag SAM2 around when the rendered MANO silhouette
    fit drifts frame-to-frame; one clean seed at the reference frame
    propagates more cleanly.
    """
    from PIL import Image

    prompts: list[Sam2Prompt] = []
    for hand_id in (_LEFT_HAND_ID, _RIGHT_HAND_ID):
        mask_path = os.path.join(
            silhouettes_dir, str(hand_id), f"{frame_index:06d}.png",
        )
        if not os.path.exists(mask_path):
            print(f"  WARNING: no silhouette for hand {hand_id} at frame "
                  f"{frame_index} ({mask_path}); skipping.")
            continue
        npx = int((np.asarray(Image.open(mask_path)) > 0).sum())
        if npx < min_mask_pixels:
            print(f"  WARNING: silhouette for hand {hand_id} at frame "
                  f"{frame_index} too small ({npx} px < "
                  f"{min_mask_pixels}); skipping.")
            continue
        prompts.append(Sam2Prompt(
            frame_index=frame_index,
            object_id=hand_id,
            mask_path=os.path.abspath(mask_path),
        ))
    return prompts


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
# Consolidated NPZ output
# ---------------------------------------------------------------------------
#
# Schema (clip.npz) — all per-frame arrays share the same N-length frame axis:
#
#   camera_to_world_transform        (N, 4, 4) float32   cam-to-world SE(3);
#                                                        world = cam frame 0
#   camera_intrinsics                (4,)       float32  fx, fy, cx, cy
#
#   object_to_camera_transform       (N, 4, 4) float32   obj-to-camera SE(3)
#   object_scale                     ()         float32  multiplier on the mesh
#                                                        in `mesh_path` (sidecar)
#   object_is_valid                  (N,)       bool
#
#   hand_left_betas                  (10,)     float32   MANO shape (per-clip)
#   hand_left_wrist_orient_in_camera (N, 3)    float32   wrist axis-angle, cam frame
#   hand_left_wrist_trans_in_camera  (N, 3)    float32   wrist translation, cam frame
#   hand_left_finger_pose            (N, 15, 3) float32  15 finger joints, axis-angle
#                                                        (does NOT include the wrist)
#   hand_left_scale                  (N,)      float32   per-frame MANO scale
#   hand_left_is_valid               (N,)      bool
#
#   hand_right_*                                          (same shape)
#
# Missing entries (object lost a frame, hand absent) are filled with safe
# defaults: identity 4x4 for transforms, zeros for MANO params, 1.0 for
# scales — consumers should mask by *_is_valid.

def _load_transform_matrices(
    poses_dir: str, n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Transform3d JSONs `{frame:06d}.json` into a stacked (N, 4, 4)
    float32 array + a (N,) bool validity mask. Missing frames get identity."""
    M = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    valid = np.zeros((n_frames,), dtype=bool)
    if not os.path.isdir(poses_dir):
        return M, valid
    for path in sorted(glob.glob(os.path.join(poses_dir, "*.json"))):
        fidx = int(os.path.splitext(os.path.basename(path))[0])
        if not (0 <= fidx < n_frames):
            continue
        M[fidx] = Transform3d.load(path).to_matrix().astype(np.float32)
        valid[fidx] = True
    return M, valid


def _load_hand_track(
    track_dir: str, n_frames: int,
) -> dict[str, np.ndarray]:
    """Load per-frame HaMeR-aligned hand JSONs into stacked arrays.

    Returns a dict with `betas, wrist_orient, wrist_trans, finger_pose,
    scale, is_valid` shaped as per the clip-NPZ schema. Missing frames get
    zeros (params) / 1.0 (scale) and is_valid=False. Betas are taken from
    the first valid frame (MANO shape is per-clip constant).
    """
    betas        = np.zeros((10,),               dtype=np.float32)
    wrist_orient = np.zeros((n_frames, 3),       dtype=np.float32)
    wrist_trans  = np.zeros((n_frames, 3),       dtype=np.float32)
    finger_pose  = np.zeros((n_frames, 15, 3),   dtype=np.float32)
    scale        = np.ones((n_frames,),          dtype=np.float32)
    is_valid     = np.zeros((n_frames,),         dtype=bool)
    if not os.path.isdir(track_dir):
        return dict(betas=betas, wrist_orient=wrist_orient,
                    wrist_trans=wrist_trans, finger_pose=finger_pose,
                    scale=scale, is_valid=is_valid)
    got_betas = False
    for path in sorted(glob.glob(os.path.join(track_dir, "*.json"))):
        fidx = int(os.path.splitext(os.path.basename(path))[0])
        if not (0 <= fidx < n_frames):
            continue
        with open(path) as f:
            rec = json.load(f)
        if not got_betas:
            betas = np.asarray(rec["mano"]["betas"], dtype=np.float32)
            got_betas = True
        wrist_orient[fidx] = np.asarray(rec["mano"]["global_orient"], dtype=np.float32)
        # MANO hand_pose is stored flat (45,) = 15 finger joints x 3 axis-angle.
        finger_pose[fidx]  = np.asarray(rec["mano"]["hand_pose"],
                                        dtype=np.float32).reshape(15, 3)
        wrist_trans[fidx]  = np.asarray(rec["cam_t"], dtype=np.float32)
        scale[fidx]        = float(rec.get("hand_scale", 1.0))
        is_valid[fidx]     = True
    return dict(betas=betas, wrist_orient=wrist_orient,
                wrist_trans=wrist_trans, finger_pose=finger_pose,
                scale=scale, is_valid=is_valid)


def _package_mesh_to_result(
    mesh_path: str, result_dir: str, dest_name: str = "mesh.obj",
) -> None:
    """Copy a textured OBJ + its sibling .mtl + map_* texture files into
    `result_dir`.

    The OBJ is copied unchanged (just renamed to `dest_name`); .mtl and
    texture files keep their original filenames so the OBJ's `mtllib` /
    `map_Kd` references resolve in the new location without rewriting.
    """
    src_dir = os.path.dirname(os.path.abspath(mesh_path))
    os.makedirs(result_dir, exist_ok=True)
    shutil.copy(mesh_path, os.path.join(result_dir, dest_name))

    mtl_names: list[str] = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("mtllib "):
                mtl_names.extend(line.strip().split()[1:])

    for mtl in mtl_names:
        mtl_src = os.path.join(src_dir, mtl)
        if not os.path.exists(mtl_src):
            continue
        shutil.copy(mtl_src, os.path.join(result_dir, mtl))
        with open(mtl_src) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].lower().startswith("map_"):
                    tex = parts[-1]
                    tex_src = os.path.join(src_dir, tex)
                    if os.path.exists(tex_src):
                        shutil.copy(tex_src, os.path.join(result_dir, tex))


def _consolidate_clip_npz(
    output_path: str,
    n_frames: int,
    intrinsics_path: str,
    cam_to_world_dir: str | None,
    object_poses_dir: str,
    object_scale: float,
    hand_left_dir: str,
    hand_right_dir: str,
) -> None:
    """Read final pipeline outputs and write a single clip.npz."""
    intr = CameraIntrinsics.load(intrinsics_path)
    camera_intrinsics = np.array(
        [intr.fx, intr.fy, intr.cx, intr.cy], dtype=np.float32,
    )

    # Camera trajectory. Without SLAM (no --gsplat) the camera is treated
    # as a static origin: cam-to-world = identity at every frame.
    if cam_to_world_dir and os.path.isdir(cam_to_world_dir):
        camera_to_world_transform, _ = _load_transform_matrices(
            cam_to_world_dir, n_frames,
        )
    else:
        camera_to_world_transform = np.tile(
            np.eye(4, dtype=np.float32), (n_frames, 1, 1),
        )

    object_to_camera_transform, object_is_valid = _load_transform_matrices(
        object_poses_dir, n_frames,
    )
    left  = _load_hand_track(hand_left_dir,  n_frames)
    right = _load_hand_track(hand_right_dir, n_frames)

    np.savez(
        output_path,
        camera_to_world_transform         = camera_to_world_transform,
        camera_intrinsics                 = camera_intrinsics,
        object_to_camera_transform        = object_to_camera_transform,
        object_scale                      = np.float32(object_scale),
        object_is_valid                   = object_is_valid,
        hand_left_betas                   = left["betas"],
        hand_left_wrist_orient_in_camera  = left["wrist_orient"],
        hand_left_wrist_trans_in_camera   = left["wrist_trans"],
        hand_left_finger_pose             = left["finger_pose"],
        hand_left_scale                   = left["scale"],
        hand_left_is_valid                = left["is_valid"],
        hand_right_betas                  = right["betas"],
        hand_right_wrist_orient_in_camera = right["wrist_orient"],
        hand_right_wrist_trans_in_camera  = right["wrist_trans"],
        hand_right_finger_pose            = right["finger_pose"],
        hand_right_scale                  = right["scale"],
        hand_right_is_valid               = right["is_valid"],
    )


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
    anycalib_weights: str = "data/weights/anycalib",
    droid_slam_weights: str = "data/weights/droid_slam",
    undistort: bool = False,
    depth_source: str = "moge",
    reference_frame: int = 0,
    reregister_iou_thresh: float | None = 0.3,
    seed: int = 0,
    gsplat: bool = False,
    gsplat_min_mask_pixels: int = 256,
    gsplat_resume: bool = False,
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
    render_aligned          = f"{output_dir}/render_aligned_{ds}.mp4"
    render_unaligned        = f"{output_dir}/render_unaligned_{ds}.mp4"
    # v2d_hamer-style alignment artifacts.
    hamer_tracks_dir        = f"{output_dir}/dynhamr_as_hamer_tracks_{ds}"
    hamer_aligned_dir       = f"{output_dir}/hamer_aligned_from_dynhamr_{ds}"
    hamer_aligned_sentinel  = f"{hamer_aligned_dir}/hand_scale.json"
    # gsplat refinement artifacts (only populated when --gsplat).
    slam_poses_dir          = f"{output_dir}/slam_poses_{ds}"
    slam_trajectory         = f"{output_dir}/slam_trajectory_{ds}.txt"
    gsplat_sam2_prompts     = f"{output_dir}/sam2_prompts_gsplat_{ds}.json"
    gsplat_masks_dir        = f"{output_dir}/masks_gsplat_{ds}"
    refined_poses_dir       = f"{output_dir}/poses_refined_{ds}"
    refined_hand_dir        = f"{output_dir}/hamer_aligned_refined_{ds}"
    refined_overlay         = f"{output_dir}/refined_overlay_{ds}.mp4"
    refined_overlay_pass2   = f"{output_dir}/refined_overlay_{ds}_pass2.mp4"
    render_refined          = f"{output_dir}/render_refined_{ds}.mp4"
    refined_object_scale    = f"{output_dir}/refined_object_scale_{ds}.json"
    refine_checkpoint_pt    = f"{output_dir}/refine_checkpoint_{ds}.pt"
    gsplat_prompts_overlay  = f"{output_dir}/sam2_prompts_gsplat_overlay_{ds}"
    hand_silhouettes_dir    = f"{output_dir}/hand_silhouettes_{ds}"

    ref_rgb   = f"{frames_dir}/{reference_frame:06d}.png"
    ref_depth = f"{active_depth_dir}/{reference_frame:06d}.png"
    ref_mask  = f"{masks_dir}/{OBJECT_ID}/{reference_frame:06d}.png"

    print(f"\n{'='*60}")
    print(f"  video        : {os.path.basename(video_path)}")
    print(f"  prompt       : {prompt!r}")
    print(f"  output       : {output_dir}")
    print(f"  depth_source : {depth_source}")
    print(f"  undistort    : {undistort}")
    print(f"  gsplat       : {gsplat}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Step 0: AnyCalib undistortion (optional)
    # When enabled, calibrate the input video, undistort it, and rebind
    # video_path so all downstream steps operate on the pinhole stream.
    # -----------------------------------------------------------------------
    if undistort:
        anycalib_dir          = f"{output_dir}/anycalib"
        anycalib_intrinsics   = f"{anycalib_dir}/intrinsics.json"
        anycalib_distortion   = f"{anycalib_dir}/distortion.json"
        undistorted_video     = f"{anycalib_dir}/undistorted.mp4"
        undistorted_intr_json = f"{anycalib_dir}/undistorted_intrinsics.json"

        os.makedirs(anycalib_dir, exist_ok=True)
        if not _step("AnyCalib undistortion", os.path.exists(undistorted_video)):
            run_video_to_calibration(
                video_path                  = video_path,
                intrinsics_path             = anycalib_intrinsics,
                distortion_path             = anycalib_distortion,
                weights_path                = anycalib_weights,
                undistorted_video_path      = undistorted_video,
                undistorted_intrinsics_path = undistorted_intr_json,
                dev                         = dev,
            )

        video_path = undistorted_video
        print(f"  → using undistorted video: {video_path}")

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
    # When --undistort is on, hand AnyCalib's intrinsics to MoGe as a
    # focal-length prior (only fov_x is consumed; cx/cy are still recomputed
    # by MoGe at image center, and the saved per-frame intrinsics come from
    # MoGe so they stay geometrically consistent with its depth tensor).
    # -----------------------------------------------------------------------
    moge_input_intrinsics = (
        f"{output_dir}/anycalib/undistorted_intrinsics.json" if undistort else None
    )
    if not _step("MoGe depth + intrinsics", _has_files(depth_dir)):
        run_moge_depth(
            video_path            = video_path,
            depth_folder          = depth_dir,
            intrinsics_folder     = intrinsics_dir,
            weights_path          = moge_weights,
            input_intrinsics_path = moge_input_intrinsics,
            dev                   = dev,
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
            seed                  = (20 if seed >= 0 else None),
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
            n_particles            = 64,
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
    # Step 11: Convert DynHaMR npz → v2d_hamer-style per-frame tracks.
    # Re-expresses each hand pose in DynHaMR's per-frame camera coordinates,
    # dropping the ViPE world / inter-frame extrinsics entirely. The output
    # mirrors what HaMeR/WiLoR produce, so v2d_hamer.align_hands can consume
    # it directly — and the alignment is robust to bad ViPE intrinsics /
    # camera-pose estimates because nothing in the alignment math touches
    # `cam_R, cam_t` per-frame.
    # -----------------------------------------------------------------------
    if not _step(f"DynHaMR → v2d_hamer-style tracks ({ds})",
                 os.path.isdir(hamer_tracks_dir)
                 and any(os.scandir(hamer_tracks_dir))):
        run_dynhamr_to_hamer_tracks(
            input_npz       = world_results_npz,
            output_dir      = hamer_tracks_dir,
            intrinsics_path = active_intrinsics,
            dev             = dev,
        )

    # -----------------------------------------------------------------------
    # Step 12: Per-frame depth alignment with v2d_hamer.align_hands.
    # -----------------------------------------------------------------------
    if not _step(f"HaMeR-style depth alignment ({ds})",
                 os.path.exists(hamer_aligned_sentinel)):
        run_align_hands(
            hamer_dir        = hamer_tracks_dir,
            depth_dir        = active_depth_dir,
            intrinsics_path  = active_intrinsics,
            mano_assets_root = mano_weights,
            output_dir       = hamer_aligned_dir,
            object_masks_dir = f"{masks_dir}/{OBJECT_ID}",
            dev              = dev,
        )

    # -----------------------------------------------------------------------
    # Step 13a: Render aligned hand + object (depth-shifted cam_t).
    # -----------------------------------------------------------------------
    if not _step(f"Render aligned ({ds})", os.path.exists(render_aligned)):
        run_render_hands_aligned_video(
            aligned_dir      = hamer_aligned_dir,
            frames_dir       = frames_dir,
            mano_assets_root = mano_weights,
            output_path      = render_aligned,
            object_mesh_path = mesh_scaled,
            object_poses_dir = poses_smooth_dir,
            dev              = dev,
        )

    # -----------------------------------------------------------------------
    # Step 13b: Render un-aligned hand + object for visual comparison.
    # Same renderer, same aligned tracks, same intrinsics — but reads
    # ``diagnostics.cam_t_pre_dz`` (the rescaled cam_t before depth shift)
    # and forces hand_scale=1. Lets us see what the per-frame depth
    # alignment actually shifts.
    # -----------------------------------------------------------------------
    if not _step(f"Render unaligned ({ds})", os.path.exists(render_unaligned)):
        run_render_hands_aligned_video(
            aligned_dir      = hamer_aligned_dir,
            frames_dir       = frames_dir,
            mano_assets_root = mano_weights,
            output_path      = render_unaligned,
            object_mesh_path = mesh_scaled,
            object_poses_dir = poses_smooth_dir,
            use_pre_dz_cam_t = True,
            dev              = dev,
        )

    # -----------------------------------------------------------------------
    # Step 14+: Optional joint hand+object Gaussian-splat refinement.
    # Runs DROID-SLAM to seed gsplat's background pose field, then a 2nd
    # SAM2 pass prompted with the DynHaMR-projected hand bboxes (+ the DINO
    # object bbox), then run_refine against the result.
    # -----------------------------------------------------------------------
    if gsplat:
        # 14: DROID-SLAM trajectory (cam-to-world per-frame Transform3d JSONs),
        # scale-aligned to the active depth source so the trajectory is in
        # metric units consistent with the rest of the pipeline. Used to seed
        # gsplat's background pose field — the identity init is fine for a
        # near-static camera but not for ego footage.
        if not _step(f"DROID-SLAM trajectory ({ds})", _has_files(slam_poses_dir)):
            run_video_to_slam(
                video_path            = video_path,
                poses_folder          = slam_poses_dir,
                weights_path          = droid_slam_weights,
                input_intrinsics_path = active_intrinsics,
                align_to_depth_folder = active_depth_dir,
                trajectory_path       = slam_trajectory,
                dev                   = dev,
            )

        # 15a: Render per-track per-frame MANO silhouettes from the
        # depth-aligned hand tracks. These mask PNGs become the SAM2 prompts
        # below — silhouettes match the actual hand position much more
        # tightly than the centroid bbox we used previously, especially
        # mid-grasp when the wrist+fingers don't form a rectangle.
        sil_done = any(
            _has_files(os.path.join(hand_silhouettes_dir, d))
            for d in (os.listdir(hand_silhouettes_dir)
                      if os.path.isdir(hand_silhouettes_dir) else [])
        )
        if not _step(f"Render MANO silhouettes ({ds})", sil_done):
            run_tracks_to_masks(
                tracks_dir       = hamer_aligned_dir,
                intrinsics_path  = active_intrinsics,
                mano_assets_root = mano_weights,
                output_dir       = hand_silhouettes_dir,
                dev              = dev,
            )

        # 15b: Build SAM2 prompts. Object box from existing DINO detections
        # at the reference frame (track 1); hand mask PNGs sampled from the
        # rendered silhouettes at evenly-spaced frames (tracks 2/3).
        if not os.path.exists(gsplat_sam2_prompts):
            with open(dino_detections) as f:
                obj_dets = json.load(f)
            if not obj_dets:
                raise RuntimeError(
                    f"DINO detections empty at {dino_detections}; cannot build "
                    f"gsplat SAM2 prompts. Re-run after a successful DINO step."
                )
            prompts: list[Sam2Prompt] = [Sam2Prompt(
                frame_index=reference_frame,
                object_id=OBJECT_ID,
                box=BoundingBox.from_dict(obj_dets[0]["box"]),
            )]
            prompts.extend(_gsplat_hand_prompts(
                hand_silhouettes_dir,
                reference_frame,
                gsplat_min_mask_pixels,
            ))
            if len(prompts) == 1:
                raise RuntimeError(
                    "No usable hand silhouettes in "
                    f"{hand_silhouettes_dir} — gsplat refinement needs at "
                    "least one hand mask. Try lowering "
                    "--gsplat_min_mask_pixels."
                )
            Sam2Prompts(prompts=prompts).save(gsplat_sam2_prompts)
            print(f"  Wrote {len(prompts)} SAM2 prompts → {gsplat_sam2_prompts}")

        # 15b: Render prompt-overlay frames so the boxes fed to SAM2 are
        # inspectable. One PNG per prompted frame at
        # <output_dir>/sam2_prompts_gsplat_overlay_{ds}/<frame:06d>.png.
        if not _step(f"Render gsplat prompts overlay ({ds})",
                     _has_files(gsplat_prompts_overlay)):
            _render_gsplat_prompts_overlay(
                prompts_path = gsplat_sam2_prompts,
                frames_dir   = frames_dir,
                output_dir   = gsplat_prompts_overlay,
            )

        # 16: Second SAM2 pass — object + per-side hand masks share a single
        # propagation so memory bank stays coherent across tracks.
        gsplat_sam2_done = any(
            _has_files(os.path.join(gsplat_masks_dir, d))
            for d in (os.listdir(gsplat_masks_dir)
                      if os.path.isdir(gsplat_masks_dir) else [])
        )
        if not _step(f"SAM2 pass for gsplat (object + hands, {ds})",
                     gsplat_sam2_done):
            run_video_to_masks(
                video_path  = video_path,
                prompts_path= gsplat_sam2_prompts,
                masks_dir   = gsplat_masks_dir,
                weights_dir = sam2_weights,
                dev         = dev,
            )

        # 17: Joint hand+object gsplat refinement. Hand poses come from the
        # already-aligned hamer_aligned_from_dynhamr_{ds}/ track dirs;
        # left/right are tracks 2/3 by convention. Hand+object masks come
        # from the 2nd SAM2 pass. Background pose field seeded with the
        # DROID-SLAM trajectory. Hyperparams mirror run_ego_wilor.py's
        # pass1 — they're the validated defaults for this kind of clip.
        left_pose_in   = os.path.join(hamer_aligned_dir, str(_LEFT_HAND_ID))
        right_pose_in  = os.path.join(hamer_aligned_dir, str(_RIGHT_HAND_ID))
        left_mask_in   = os.path.join(gsplat_masks_dir,  str(_LEFT_HAND_ID))
        right_mask_in  = os.path.join(gsplat_masks_dir,  str(_RIGHT_HAND_ID))
        left_pose_out  = os.path.join(refined_hand_dir,  str(_LEFT_HAND_ID))
        right_pose_out = os.path.join(refined_hand_dir,  str(_RIGHT_HAND_ID))

        def _maybe_dir(d: str) -> str | None:
            return d if os.path.isdir(d) else None
        left_pose_in_arg   = _maybe_dir(left_pose_in)
        right_pose_in_arg  = _maybe_dir(right_pose_in)
        left_mask_in_arg   = _maybe_dir(left_mask_in)
        right_mask_in_arg  = _maybe_dir(right_mask_in)
        left_pose_out_arg  = left_pose_out  if left_pose_in_arg  else None
        right_pose_out_arg = right_pose_out if right_pose_in_arg else None

        # Shared pass-1 kwargs — reused (with overrides) for the pose-only
        # pass-2 finetune below.
        refine_kwargs = dict(
            frames_dir                  = frames_dir,
            intrinsics_path             = active_intrinsics,
            object_mesh_path            = mesh_scaled,
            object_poses_dir            = poses_dir,
            object_mask_dir             = os.path.join(gsplat_masks_dir, str(OBJECT_ID)),
            refined_object_poses_dir    = refined_poses_dir,
            overlay_path                = refined_overlay,
            refined_object_scale_path   = refined_object_scale,
            checkpoint_path             = refine_checkpoint_pt,
            resume_from_checkpoint      = (
                refine_checkpoint_pt
                if gsplat_resume and os.path.exists(refine_checkpoint_pt)
                else None
            ),
            left_hand_pose_dir          = left_pose_in_arg,
            left_hand_mask_dir          = left_mask_in_arg,
            right_hand_pose_dir         = right_pose_in_arg,
            right_hand_mask_dir         = right_mask_in_arg,
            refined_left_hand_pose_dir  = left_pose_out_arg,
            refined_right_hand_pose_dir = right_pose_out_arg,
            depth_dir                   = active_depth_dir,
            mano_assets_root            = mano_weights,
            n_epochs                    = 30,
            batch_size                  = 32,
            render_every                = 25,
            lr_gaussians                = 3e-3,
            lr_hand_gaussians           = 3e-3,
            lr_mul_delta_p              = 0.2,
            lr_mul_quat                 = 0.5,
            lr_mul_scale                = 15.0,
            lr_mul_opacity              = 3.0,
            lr_mul_color                = 1.0,
            lr_mul_obj_global_scale     = 1.0,
            lr_object_pose              = 3e-3,
            lr_object_rot               = 3e-3,
            lr_object_trans             = 3e-3,
            lr_hand_pose                = 3e-3,
            lr_hand_global_orient       = 3e-3,
            lr_hand_finger              = 3e-3,
            lr_hand_trans               = 3e-3,
            lr_betas                    = 3e-3,
            learn_hand_scale            = True,
            lr_hand_scale               = 3e-3,
            w_hand_scale_prior          = 1.0,
            w_photometric               = 1.0,
            w_silhouette                = 1.0,
            w_silhouette_hand           = 1.0,
            w_silhouette_obj            = 1.0,
            w_depth                     = 0.1,
            w_log_depth_grad            = 0.1,
            w_photometric_ssim          = 1.0,
            w_depth_ssim                = 0.0,
            w_delta_p_reg_obj           = 0.001,
            w_delta_p_reg_hand          = 0.001,
            w_delta_p_reg_bg            = 0.001,
            w_smooth_obj_rot            = 0.01,
            w_smooth_obj_trans          = 0.01,
            w_smooth_hand_rot           = 0.01,
            w_smooth_hand_finger        = 0.01,
            w_smooth_hand_trans         = 0.01,
            w_smooth_bg_rot             = 0.1,
            w_smooth_bg_trans           = 0.1,
            w_beta_prior                = 1.0,
            w_obj_scale_prior           = 1.0,
            n_gaussian_only_epochs      = 2,
            with_background             = True,
            background_pose_init_dir    = slam_poses_dir,
            bg_ref_frame                = reference_frame,
            lr_bg_gaussians             = 3e-3,
            lr_bg_pose                  = 3e-3,
            lr_bg_rot                   = 3e-3,
            lr_bg_trans                 = 3e-3,
            bg_max_points               = 40000,
            bg_init_stride              = 10,
            bg_voxel_size               = 0.005,
            w_scale_aniso_bg            = 0.0,
            w_sdf_density_bg            = 0.0,
            w_normal_consistency_bg     = 0.0,
            n_sdf_samples_bg            = 20000,
            n_sdf_neighbors_bg          = 16,
            use_cosine_lr_schedule      = True,
            cosine_lr_min_ratio         = 0.02,
            pose_confidence_ref_frame   = reference_frame,
            w_pose_init_prior           = 0.0,
            use_l2_photometric          = True,
            use_l2_silhouette           = True,
            train_resolution_scale      = 0.5,
            n_obj_gaussians             = 5000,
            object_anchor_mode          = "face",
            hand_anchor_mode            = "face",
            face_normal_thin_factor_obj = 0.25,
            w_face_delta_p_normal_outward_hand = 10.0,
            w_face_delta_p_normal_inward_hand  = 10.0,
            w_face_delta_p_tangent_hand        = 10.0,
            checkpoint_every            = 25,
            w_opacity_binary_hand       = 0.0,
            w_opacity_binary_obj        = 0.0,
            w_opacity_binary_bg         = 0.0,
            w_depth_variance            = 0.0,
            w_depth_ordering            = 0.0,
            n_wrist_gaussians           = 0,
            learn_focal                 = True,
            learn_principal_point       = True,
            w_intrinsics_prior          = 1e3,
            snap_rotation_outliers_every = 0,
            snap_rotation_targets       = "obj,hand_wrist",
            snap_rotation_threshold     = 0.6,
            snap_rotation_window        = 5,
            snap_rotation_causal        = True,
            snap_rotation_anchor_frame  = reference_frame,
            seed                        = 0,
            dev                         = dev,
            random_init_obj_pose        = False,
        )

        if not _step(f"Gaussian-splat refinement ({ds})",
                     os.path.exists(refined_overlay)):
            run_refine(**refine_kwargs)

        # 18: Pose-only pass-2 finetune. Freezes all Gaussian attributes
        # (color/scale/opacity/quat/delta_p across object + hands + bg) and
        # re-finetunes poses against the now-stable scene at ~6x lower LR,
        # with fewer epochs. Resumes from the pass-1 checkpoint but starts
        # Adam moments fresh so the lower LRs aren't fighting pass-1
        # momentum. Writes a separate overlay so both passes stay
        # inspectable; the refined poses / scale / hand poses overwrite the
        # pass-1 outputs (pass 2 is the final).
        if not _step(f"Gaussian-splat pose-only refinement pass 2 ({ds})",
                     os.path.exists(refined_overlay_pass2)):
            if not os.path.exists(refine_checkpoint_pt):
                print(
                    f"  Skipping pass 2 — no checkpoint at "
                    f"{refine_checkpoint_pt} (pass 1 must run first)."
                )
            else:
                run_refine(**{
                    **refine_kwargs,
                    "overlay_path":            refined_overlay_pass2,
                    "resume_from_checkpoint":  refine_checkpoint_pt,
                    "ignore_optimizer_state":  True,
                    "freeze_gaussians":        True,
                    "n_epochs":                10,
                    "n_gaussian_only_epochs":  0,
                    "lr_object_pose":          5e-4,
                    "lr_object_rot":           5e-4,
                    "lr_object_trans":         5e-4,
                    "lr_hand_pose":            5e-4,
                    "lr_hand_global_orient":   5e-4,
                    "lr_hand_finger":          5e-4,
                    "lr_hand_trans":           5e-4,
                    "lr_betas":                5e-4,
                    "lr_hand_scale":           5e-4,
                    "lr_bg_pose":              5e-4,
                    "lr_bg_rot":               5e-4,
                    "lr_bg_trans":             5e-4,
                })

        # 19: Mesh-rendered overlay of gsplat-refined poses. Mirrors
        # render_aligned_{ds}.mp4 but uses the refined hand tracks + refined
        # object poses + the learned multiplicative object_scale. Uses the
        # textured OBJ for the object (not gsplat) so the result is directly
        # comparable to the pre-gsplat aligned render.
        if not _step(f"Render refined ({ds})", os.path.exists(render_refined)):
            with open(refined_object_scale) as f:
                refined_scale = float(json.load(f)["scale"])
            run_render_hands_aligned_video(
                aligned_dir      = refined_hand_dir,
                frames_dir       = frames_dir,
                mano_assets_root = mano_weights,
                output_path      = render_refined,
                object_mesh_path = mesh_scaled,
                object_poses_dir = refined_poses_dir,
                object_scale     = refined_scale,
                dev              = dev,
            )

    # -----------------------------------------------------------------------
    # Final: packaged result/ folder (mesh + textures + result.npz)
    # Bundles the consolidated time-aligned NPZ (camera trajectory, object
    # pose+scale, both hands MANO + scale, validity masks) together with the
    # textured object mesh that the object_to_camera_transform refers to.
    # Uses refined outputs when --gsplat ran, smoothed/aligned otherwise.
    # See _consolidate_clip_npz for the NPZ schema.
    # -----------------------------------------------------------------------
    result_dir  = f"{output_dir}/result"
    result_npz  = f"{result_dir}/result.npz"
    result_mesh = f"{result_dir}/mesh.obj"

    if gsplat and _has_files(refined_poses_dir):
        final_object_poses_dir = refined_poses_dir
        with open(refined_object_scale) as f:
            final_object_scale = float(json.load(f)["scale"])
        final_hand_root = refined_hand_dir
    else:
        final_object_poses_dir = poses_smooth_dir
        final_object_scale     = 1.0
        final_hand_root        = hamer_aligned_dir

    final_hand_left  = os.path.join(final_hand_root, str(_LEFT_HAND_ID))
    final_hand_right = os.path.join(final_hand_root, str(_RIGHT_HAND_ID))

    n_frames = len(glob.glob(f"{frames_dir}/*.png")) or len(
        glob.glob(f"{frames_dir}/*.jpg"))
    if not _step(f"Package result/ ({ds})",
                 os.path.exists(result_npz) and os.path.exists(result_mesh)):
        os.makedirs(result_dir, exist_ok=True)
        _package_mesh_to_result(mesh_scaled, result_dir, dest_name="mesh.obj")
        _consolidate_clip_npz(
            output_path      = result_npz,
            n_frames         = n_frames,
            intrinsics_path  = active_intrinsics,
            cam_to_world_dir = slam_poses_dir if gsplat else None,
            object_poses_dir = final_object_poses_dir,
            object_scale     = final_object_scale,
            hand_left_dir    = final_hand_left,
            hand_right_dir   = final_hand_right,
        )

    print(f"\n{'='*60}")
    print(f"  Done!  (depth_source={depth_source})")
    print(f"  Result bundle:         {result_dir}/")
    print(f"  Aligned hand tracks:   {hamer_aligned_dir}/")
    print(f"  Scaled mesh:           {mesh_scaled}")
    print(f"  Smoothed poses:        {poses_smooth_dir}/")
    print(f"  Render (aligned):      {render_aligned}")
    print(f"  Render (unaligned):    {render_unaligned}")
    if gsplat:
        print(f"  SLAM poses:            {slam_poses_dir}/")
        print(f"  Hand silhouettes:      {hand_silhouettes_dir}/")
        print(f"  Prompts overlay:       {gsplat_prompts_overlay}/")
        print(f"  Gsplat masks:          {gsplat_masks_dir}/")
        print(f"  Refined object poses:  {refined_poses_dir}/")
        print(f"  Refined hand poses:    {refined_hand_dir}/")
        print(f"  Refined overlay:       {refined_overlay}")
        print(f"  Refined overlay (p2):  {refined_overlay_pass2}")
        print(f"  Refined render (mesh): {render_refined}")
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
        anycalib_weights            = args.anycalib_weights,
        droid_slam_weights          = args.droid_slam_weights,
        undistort                   = args.undistort,
        depth_source                = args.depth_source,
        reference_frame             = args.reference_frame,
        reregister_iou_thresh       = args.reregister_iou_thresh,
        seed                        = args.seed,
        gsplat                      = args.gsplat,
        gsplat_min_mask_pixels      = args.gsplat_min_mask_pixels,
        gsplat_resume               = args.gsplat_resume,
        dev                         = args.dev,
    )
