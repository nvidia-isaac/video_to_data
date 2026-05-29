#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Run BundleSDF reconstruction with pre-computed camera poses.

Expects the output directory to already contain (or have custom paths supplied):
  - keyframes.yml  (pre-computed camera poses)    override with --poses_file
  - left/          (RGB images)                   override with --images_dir
  - depth/         (depth maps — one per keyframe) override with --depth_dir
  - masks/         (object masks — one per keyframe) override with --masks_dir
  - calibration.json (camera intrinsics)          override with --intrinsics_file

Outputs written to output_path:
  - mesh_cleaned.obj    untextured SDF mesh
  - textured_mesh.obj   textured mesh (+ .mtl, _0.png atlas)
  - model_latest.pth    saved SDF model

Usage:
  python reconstruct.py \\
    --output_path /path/to/recon_dir \\
    --weights_dir /path/to/weights

  # With custom input directories:
  python reconstruct.py \\
    --output_path /path/to/recon_dir \\
    --weights_dir /path/to/weights \\
    --images_dir /path/to/images \\
    --depth_dir /path/to/depth \\
    --masks_dir /path/to/masks \\
    --poses_file /path/to/keyframes.yml \\
    --intrinsics_file /path/to/calibration.json
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
import yaml
import time
import torch

# Add 3d-object-reconstruction src to path (installed in container at /workspace/3d-object-reconstruction/src)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nvidia.objectreconstruction.networks import NVBundleSDF
from nvidia.objectreconstruction.dataloader import ReconstructionDataLoader

# NVBundleSDF import resets float32_matmul_precision to 'high'; restore to 'highest' for RoMa
torch.set_float32_matmul_precision('highest')

# ReconstructionDataLoader.get_depth() assumes mm encoding (pixel/1000),
# but our depth PNGs use inverse-depth encoding: pixel = 65535/(depth_m+1).
# Patch to decode correctly.
import cv2 as _cv2
import numpy as _np

def _get_depth_inverse(self, idx):
    if not self.has_depth_maps:
        return None
    if idx < 0 or idx >= len(self):
        raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
    depth_file = self.color_files[idx].replace('left/', 'depth/')
    if os.path.exists(depth_file.replace('.png', '.npy')):
        depth_file = depth_file.replace('.png', '.npy')
    try:
        if depth_file.endswith('.npy'):
            depth = _np.load(depth_file)
        else:
            raw = _cv2.imread(depth_file, _cv2.IMREAD_UNCHANGED)
            if raw is None:
                return None
            raw = raw.astype(_np.float32)
            valid = raw > 0
            depth = _np.zeros_like(raw)
            depth[valid] = 1.0 / (raw[valid] / 65535.0) - 1.0  # inverse-depth → meters
        depth = _cv2.resize(depth, (self.W, self.H), interpolation=_cv2.INTER_NEAREST)
        return depth
    except Exception as e:
        print(f"Warning: Failed to load depth map {depth_file}: {e}")
        return None

ReconstructionDataLoader.get_depth = _get_depth_inverse


def _subsample_keyframes_for_texture(
    output_path: Path,
    min_translation: float,
    min_rotation_deg: float,
    min_keyframes: int,
    logger: logging.Logger,
) -> None:
    """Subsample keyframes.yml by minimum camera motion before texture baking."""
    import shutil

    kf_path = output_path / "keyframes.yml"
    with open(kf_path) as f:
        keyframes = yaml.safe_load(f)

    keys = sorted(keyframes.keys())
    if not keys:
        return

    selected = [keys[0]]
    last_T = _np.array(keyframes[keys[0]]["cam_in_ob"]).reshape(4, 4)

    for k in keys[1:]:
        T = _np.array(keyframes[k]["cam_in_ob"]).reshape(4, 4)
        T_rel = T @ _np.linalg.inv(last_T)
        translation = _np.linalg.norm(T_rel[:3, 3])
        cos_angle = _np.clip((_np.trace(T_rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = _np.degrees(_np.arccos(cos_angle))
        if translation >= min_translation or rotation_deg >= min_rotation_deg:
            selected.append(k)
            last_T = T

    if len(selected) < min_keyframes:
        selected_set = set(selected)
        unselected = [k for k in keys if k not in selected_set]
        needed = min_keyframes - len(selected)
        if unselected:
            step = max(1, len(unselected) // needed)
            extras = unselected[::step][:needed]
            selected = sorted(set(selected) | set(extras))
            logger.info(
                f"Keyframe subsampling: padded with {len(extras)} uniform samples "
                f"to meet min_keyframes={min_keyframes}"
            )

    logger.info(
        f"Keyframe subsampling: {len(selected)}/{len(keys)} kept "
        f"(min_translation={min_translation}m, min_rotation={min_rotation_deg}°, "
        f"min_keyframes={min_keyframes})"
    )

    shutil.copy(kf_path, output_path / "keyframes_all.yml")
    with open(kf_path, "w") as f:
        yaml.dump({k: keyframes[k] for k in selected}, f)


def setup_input_dirs(
    output_path: Path,
    images_dir: str = None,
    depth_dir: str = None,
    masks_dir: str = None,
    poses_file: str = None,
    intrinsics_file: str = None,
) -> None:
    """Symlink custom input dirs/files into the expected output_path structure."""
    import shutil

    def _link(src: str, dst: Path, is_dir: bool) -> None:
        if src is None:
            return
        # Resolve src — if it raises OSError (e.g. symlink loop caused by
        # Docker mounting the same host directory at two container paths),
        # the data is already accessible at dst, so skip.
        try:
            src_path = Path(src).resolve(strict=True)
        except OSError:
            if dst.is_symlink():
                dst.unlink()  # remove the looping symlink so validate_inputs sees real data
            return
        # Skip if dst is already the same filesystem object as src.
        if dst.exists() or dst.is_symlink():
            try:
                if os.path.samefile(dst, src_path):
                    return
            except (OSError, ValueError):
                pass
            # Only remove symlinks — never delete real directories/files.
            if dst.is_symlink():
                dst.unlink()
            else:
                raise FileExistsError(
                    f"Cannot create symlink at {dst}: a real file/directory already exists. "
                    f"Remove it manually or do not pass --{'images_dir' if is_dir else 'poses_file'}."
                )
        dst.symlink_to(src_path)

    _link(images_dir,      output_path / "left",           is_dir=True)
    _link(depth_dir,       output_path / "depth",          is_dir=True)
    _link(masks_dir,       output_path / "masks",          is_dir=True)
    _link(poses_file,      output_path / "keyframes.yml",  is_dir=False)
    _link(intrinsics_file, output_path / "calibration.json", is_dir=False)


def validate_inputs(output_path: Path) -> None:
    """Validate that the output directory has required pre-computed inputs."""
    if not output_path.exists():
        raise FileNotFoundError(f"Output path does not exist: {output_path}")
    if not output_path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_path}")

    for name in ("keyframes.yml", ):
        if not (output_path / name).exists():
            raise FileNotFoundError(f"{name} not found in {output_path}")

    for subdir in ("left", "depth", "masks"):
        d = output_path / subdir
        if not d.is_dir() or not any(d.iterdir()):
            raise FileNotFoundError(
                f"{subdir}/ missing or empty in {output_path}. "
                f"Depth and masks must be pre-computed before calling reconstruct."
            )


def main():
    _DEFAULT_CONFIG = "/workspace/v2d_bundlesdf/lib/data/configs/theseus_optimizer_hawk.yaml"

    parser = argparse.ArgumentParser(
        description="BundleSDF reconstruction with pre-computed poses, depth, and masks",
    )
    parser.add_argument("--output_path", "--output-path", dest="output_path", required=True,
                        help="Output directory for mesh results")
    parser.add_argument("--config", default=_DEFAULT_CONFIG,
                        help=f"NeRF/SDF config YAML (default: {_DEFAULT_CONFIG})")
    parser.add_argument("--weights_dir", default=None,
                        help="Root weights directory (roma/ subdirs). Overrides config weight paths.")
    parser.add_argument("--bbox_str", default=None,
                        help="Bounding box 'x1,y1,x2,y2' passed to SAM2 config (informational only)")
    parser.add_argument("--skip-texture", action="store_true",
                        help="Skip texture baking (faster; produces untextured mesh only)")
    parser.add_argument("--skip-sdf", action="store_true",
                        help="Skip SDF training; reuse existing model_latest.pth and run texture baking only")
    parser.add_argument("--images_dir", default=None,
                        help="Directory of RGB images (default: <output_path>/left/)")
    parser.add_argument("--depth_dir", default=None,
                        help="Directory of depth maps (default: <output_path>/depth/)")
    parser.add_argument("--masks_dir", default=None,
                        help="Directory of object masks (default: <output_path>/masks/)")
    parser.add_argument("--poses_file", default=None,
                        help="Camera poses YAML file (default: <output_path>/keyframes.yml)")
    parser.add_argument("--intrinsics_file", default=None,
                        help="Camera intrinsics JSON file (default: <output_path>/calibration.json)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    try:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        setup_input_dirs(
            output_path,
            images_dir=args.images_dir,
            depth_dir=args.depth_dir,
            masks_dir=args.masks_dir,
            poses_file=args.poses_file,
            intrinsics_file=args.intrinsics_file,
        )
        validate_inputs(output_path)

        config = {}
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config not found: {args.config}")
        with open(args.config) as f:
            config = yaml.safe_load(f)

        if args.weights_dir:
            w = args.weights_dir.rstrip("/")
            config['roma']['weights']       = f"{w}/roma/roma_outdoor.pth"
            config['roma']['dinov2_weights'] = f"{w}/roma/dinov2_vitl14_pretrain.pth"

        calibration_path = output_path / "calibration.json"
        if not calibration_path.exists():
            calibration_path = output_path.parent / "calibration.json"
        if calibration_path.exists():
            with open(calibration_path) as f:
                cal = json.load(f)
            intrinsic_flat = [cal['fx'], 0, cal['cx'], 0, cal['fy'], cal['cy'], 0, 0, 1]
            config['camera_config']['intrinsic']     = intrinsic_flat
            config['foundation_stereo']['intrinsic'] = intrinsic_flat
            config['foundation_stereo']['baseline']  = cal['baseline']
            logger.info(f"Loaded intrinsics from {calibration_path}")

        config['workdir'] = output_path
        config['bundletrack']['debug_dir'] = output_path / "bundletrack"
        config['nerf']['save_dir'] = output_path

        bundletrack_config  = config['bundletrack']
        nerf_config         = config['nerf']
        roma_config         = config['roma']
        texture_config      = config['texture_bake']
        texture_enabled     = bool(texture_config.get("enabled", True))
        do_texture          = (not args.skip_texture) and texture_enabled
        do_sdf              = not args.skip_sdf

        logger.info("=" * 60)
        logger.info("BundleSDF Reconstruction")
        logger.info(f"  output_path: {output_path}")
        logger.info("=" * 60)

        start_total = time.time()

        logger.info("Initializing reconstruction components...")
        start_init = time.time()
        tracker = NVBundleSDF(nerf_config, bundletrack_config, roma_config, texture_config, logger=logger)
        nerf_dataset = ReconstructionDataLoader(
            str(output_path), config,
            downscale=nerf_config['downscale'],
            min_resolution=nerf_config['min_resolution'],
        )
        texture_dataset = None
        if do_texture:
            texture_dataset = ReconstructionDataLoader(
                str(output_path), config,
                downscale=texture_config['downscale'],
                min_resolution=texture_config['min_resolution'],
            )
        time_init = time.time() - start_init
        logger.info(f"Components initialized in {time_init:.1f}s")

        time_sdf = 0.0
        if do_sdf:
            logger.info("Running SDF training...")
            start_sdf = time.time()
            tracker.run_global_sdf(nerf_dataset)
            time_sdf = time.time() - start_sdf
            logger.info(f"SDF training done in {time_sdf:.1f}s")
        else:
            logger.info("Skipping SDF training (--skip-sdf)")

        time_texture = 0.0
        if do_texture:
            logger.info("Running texture baking...")
            start_texture = time.time()
            _tb = (texture_config or {})
            min_t  = float(_tb.get("min_keyframe_translation", 0.0))
            min_r  = float(_tb.get("min_keyframe_rotation_deg", 0.0))
            min_kf = int(_tb.get("min_keyframes", 0))
            if min_t > 0.0 or min_r > 0.0:
                _subsample_keyframes_for_texture(output_path, min_t, min_r, min_kf, logger)
            tracker.run_texture_bake(texture_dataset)
            time_texture = time.time() - start_texture
            logger.info(f"Texture baking done in {time_texture:.1f}s")
        else:
            logger.info("Skipping texture baking")

        total = time.time() - start_total
        times = {
            "total": total,
            "init": time_init,
            "sdf": time_sdf,
            "texture": time_texture,
            "gpu_name": torch.cuda.get_device_name(0),
        }
        with open(output_path / "run_time.yaml", "w") as f:
            yaml.dump(times, f)

        logger.info("=" * 60)
        logger.info(f"Done in {total:.1f}s  (init={time_init:.1f}s  sdf={time_sdf:.1f}s  texture={time_texture:.1f}s)")
        logger.info("=" * 60)
        return 0

    except (FileNotFoundError, NotADirectoryError) as e:
        logging.error(str(e))
        return 2
    except yaml.YAMLError as e:
        logging.error(f"Config error: {e}")
        return 3
    except RuntimeError as e:
        logging.error(f"Processing error: {e}")
        return 4
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
