# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""I/O for the gsplat-refinement optimizer.

All loaders return torch tensors on a target device so the optimizer can
keep everything on GPU. All savers write back the same on-disk formats the
upstream pipeline produces (Transform3d JSON for object poses, HaMeR
aligned-style JSON for hands), so refined outputs slot into existing
downstream renderers without changes.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from PIL import Image

from v2d.common.datatypes import CameraIntrinsics, Mask, Transform3d


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _frame_idx(path: str) -> int:
    return int(os.path.splitext(os.path.basename(path))[0])


def list_frame_indices(directory: str, ext: str = ".png") -> list[int]:
    files = sorted(glob.glob(os.path.join(directory, f"*{ext}")))
    return [_frame_idx(f) for f in files]


# ---------------------------------------------------------------------------
# Per-frame raster inputs
# ---------------------------------------------------------------------------

def load_rgb(path: str, device: str) -> torch.Tensor:
    """Load an RGB frame as a (H, W, 3) float32 tensor in [0, 1]."""
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).to(device)


def load_mask(path: str, device: str) -> torch.Tensor:
    """Load a binary mask as a (H, W) float32 tensor in {0, 1}."""
    m = Mask.load(path).mask.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0.5).astype(np.float32)
    return torch.from_numpy(m).to(device)


def load_depth(path: str, device: str) -> torch.Tensor:
    """Load MoGe inverse-depth PNG as a (H, W) float32 metric depth tensor.

    Pixels at infinity (inverse_depth == 0) become +inf; callers should mask
    them out before computing losses.
    """
    px = np.asarray(Image.open(path)).astype(np.float32)
    with np.errstate(divide="ignore"):
        depth = 1.0 / (px / 65535.0) - 1.0
    return torch.from_numpy(depth).to(device)


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def load_intrinsics(path: str, device: str) -> tuple[torch.Tensor, int, int]:
    """Return ((3, 3) K tensor, width, height)."""
    intr = CameraIntrinsics.load(path)
    K = torch.tensor(intr.to_matrix(), dtype=torch.float32, device=device)
    return K, int(intr.width), int(intr.height)


# ---------------------------------------------------------------------------
# Object poses (Transform3d JSON, object-to-camera)
# ---------------------------------------------------------------------------

@dataclass
class ObjectPoseTrack:
    """Per-frame object→camera transforms, stacked as torch tensors.

    rotations:    (T, 4) quaternion (w, x, y, z)
    translations: (T, 3)
    scales:       (T, 3)            -- usually [1, 1, 1] post-FoundationPose
    frame_indices: list[int]
    """
    rotations: torch.Tensor
    translations: torch.Tensor
    scales: torch.Tensor
    frame_indices: list[int]


def load_object_poses(poses_dir: str, device: str) -> ObjectPoseTrack:
    files = sorted(glob.glob(os.path.join(poses_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No Transform3d JSONs in {poses_dir}")
    rots, trans, scls, idxs = [], [], [], []
    for f in files:
        t = Transform3d.load(f)
        rots.append(t.rotation)
        trans.append(t.translation)
        scls.append(t.scale)
        idxs.append(_frame_idx(f))
    return ObjectPoseTrack(
        rotations    = torch.tensor(rots,  dtype=torch.float32, device=device),
        translations = torch.tensor(trans, dtype=torch.float32, device=device),
        scales       = torch.tensor(scls,  dtype=torch.float32, device=device),
        frame_indices= idxs,
    )


def save_object_poses(track: ObjectPoseTrack, poses_dir: str) -> None:
    os.makedirs(poses_dir, exist_ok=True)
    rots  = track.rotations.detach().cpu().tolist()
    trans = track.translations.detach().cpu().tolist()
    scls  = track.scales.detach().cpu().tolist()
    for i, fidx in enumerate(track.frame_indices):
        Transform3d(
            rotation    = rots[i],
            translation = trans[i],
            scale       = scls[i],
        ).save(os.path.join(poses_dir, f"{fidx:06d}.json"))


# ---------------------------------------------------------------------------
# Hand poses (HaMeR aligned-style JSON, per-frame in a single track dir)
# ---------------------------------------------------------------------------

@dataclass
class HandPoseTrack:
    """Per-frame MANO params + cam_t in real intrinsics for a single hand.

    global_orient: (T, 3) axis-angle (root)
    hand_pose:     (T, 45) axis-angle (15 finger joints x 3)
    betas:         (T, 10)
    cam_t:         (T, 3) translation in real-camera frame
    is_right:      bool
    frame_indices: list[int]
    raw_records:   list[dict]   -- preserved verbatim so we can round-trip
                                   non-optimized fields (intrinsics, image_size,
                                   diagnostics, track_id, hand_scale) on save.
    hand_scale:    float        -- per-track multiplicative depth correction
                                   emitted by align_hands. cam_t carries the
                                   additive (dz) shift; hand_scale corrects
                                   multiplicative depth mismatch by scaling
                                   the MANO mesh around its centroid.
                                   Defaults to 1.0 when absent.
    """
    global_orient: torch.Tensor
    hand_pose: torch.Tensor
    betas: torch.Tensor
    cam_t: torch.Tensor
    is_right: bool
    frame_indices: list[int]
    raw_records: list[dict]
    hand_scale: float = 1.0


def load_hand_poses(track_dir: str, device: str) -> HandPoseTrack:
    files = sorted(glob.glob(os.path.join(track_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No hand JSONs in {track_dir}")
    go, hp, be, ct, idxs, recs = [], [], [], [], [], []
    is_right = None
    hand_scale: float | None = None
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        if is_right is None:
            is_right = bool(r["is_right"])
        elif bool(r["is_right"]) != is_right:
            raise ValueError(f"Mixed handedness in {track_dir}")
        # hand_scale is per-track-constant; first non-default record wins.
        if hand_scale is None and "hand_scale" in r:
            hand_scale = float(r["hand_scale"])
        go.append(r["mano"]["global_orient"])
        hp.append(r["mano"]["hand_pose"])
        be.append(r["mano"]["betas"])
        ct.append(r["cam_t"])
        idxs.append(_frame_idx(f))
        recs.append(r)
    return HandPoseTrack(
        global_orient = torch.tensor(go, dtype=torch.float32, device=device),
        hand_pose     = torch.tensor(hp, dtype=torch.float32, device=device),
        betas         = torch.tensor(be, dtype=torch.float32, device=device),
        cam_t         = torch.tensor(ct, dtype=torch.float32, device=device),
        is_right      = bool(is_right),
        frame_indices = idxs,
        raw_records   = recs,
        hand_scale    = float(hand_scale) if hand_scale is not None else 1.0,
    )


def save_hand_poses(track: HandPoseTrack, track_dir: str) -> None:
    os.makedirs(track_dir, exist_ok=True)
    go = track.global_orient.detach().cpu().tolist()
    hp = track.hand_pose.detach().cpu().tolist()
    be = track.betas.detach().cpu().tolist()
    ct = track.cam_t.detach().cpu().tolist()
    for i, fidx in enumerate(track.frame_indices):
        rec = dict(track.raw_records[i])  # shallow copy preserves diagnostics etc.
        rec["mano"] = {
            "global_orient": go[i],
            "hand_pose":     hp[i],
            "betas":         be[i],
        }
        rec["cam_t"] = ct[i]
        with open(os.path.join(track_dir, f"{fidx:06d}.json"), "w") as fh:
            json.dump(rec, fh, indent=2)


# ---------------------------------------------------------------------------
# Object mesh (anchor positions for object Gaussians)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Frame-level supervision cache (RGB + masks + depth)
# ---------------------------------------------------------------------------

class FrameCache:
    """Pre-loaded RGB / mask / depth tensors for the whole sequence.

    All channels are stacked into single contiguous tensors of shape (T, ...)
    held on CPU. Per-step access is a tensor index + ``.to(device)`` — no PNG
    decode, no disk I/O after construction.

    Memory cost (T frames at HxW):
        rgb:        T * H * W * 3 * 4   bytes
        each mask:  T * H * W * 4       bytes
        depth:      T * H * W * 4       bytes (only if depth_dir provided)

    So a 500-frame 720x720 clip with object mask + 2 hand masks + depth uses
    ~7 GB of CPU RAM. This is a deliberate tradeoff vs. lazy loading: a
    typical refinement run iterates each frame dozens of times, so the
    upfront one-shot decode pays for itself within the first epoch.
    """

    def __init__(
        self,
        frame_indices: list[int],
        frames_dir: str,
        object_mask_dir: str,
        hand_mask_dirs: list[str],
        depth_dir: str | None,
        height: int,
        width: int,
        valid_mask_threshold: float = 0.04,
        valid_mask_erode_iters: int = 2,
    ) -> None:
        from tqdm import tqdm

        n = len(frame_indices)
        self.frame_indices = list(frame_indices)
        self.height = height
        self.width  = width
        self.has_depth = depth_dir is not None

        self.rgb        = torch.zeros((n, height, width, 3), dtype=torch.float32)
        self.obj_mask   = torch.zeros((n, height, width),    dtype=torch.float32)
        self.hand_masks = [
            torch.zeros((n, height, width), dtype=torch.float32)
            for _ in hand_mask_dirs
        ]
        self.depth = (
            torch.zeros((n, height, width), dtype=torch.float32)
            if self.has_depth else None
        )

        for t, fidx in enumerate(tqdm(frame_indices, ncols=80,
                                       desc="caching frames", unit="frame")):
            # RGB. Resize during load (PIL) if source resolution differs
            # from the cache target (height, width). PIL.size is (W, H).
            rgb_path = _find_frame_file(frames_dir, fidx)
            img = Image.open(rgb_path).convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            self.rgb[t] = torch.from_numpy(arr)

            # Object mask (zero if missing or wrong-shape).
            self.obj_mask[t] = _load_mask_or_zero(
                object_mask_dir, fidx, height, width)

            # Per-hand masks.
            for k, mdir in enumerate(hand_mask_dirs):
                self.hand_masks[k][t] = _load_mask_or_zero(
                    mdir, fidx, height, width)

            # Optional depth (infinity-fills propagate into the loss mask).
            # Resize via PIL bilinear before decoding inverse-depth — the
            # tiny smoothing introduced by interpolating raw uint16 values
            # is fine for our soft depth prior.
            if self.has_depth:
                dpath = os.path.join(depth_dir, f"{fidx:06d}.png")
                if os.path.exists(dpath):
                    img_d = Image.open(dpath)
                    if img_d.size != (width, height):
                        img_d = img_d.resize((width, height), Image.BILINEAR)
                    px = np.asarray(img_d).astype(np.float32)
                    with np.errstate(divide="ignore"):
                        d = 1.0 / (px / 65535.0) - 1.0
                    self.depth[t] = torch.from_numpy(d)
                else:
                    self.depth[t].fill_(float("inf"))

        # Static valid-pixel mask derived from the input video itself.
        # Detects fixed black regions (fisheye crop, vignette, dead border)
        # by checking the per-pixel max brightness across all frames: any
        # pixel that never exceeds ``valid_mask_threshold`` is treated as
        # invalid and excluded from photometric / depth / SuGaR supervision.
        # Eroded by ``valid_mask_erode_iters`` 3x3 steps to peel back the
        # soft transition at the boundary.
        #
        # Single static mask (shape (H, W)), shared across all frames —
        # the artifact is image-space-fixed, so cross-frame consistency is
        # automatic and there's no per-frame degeneracy.
        max_brightness = self.rgb.amax(dim=0).amax(dim=-1)               # (H, W)
        valid = (max_brightness > float(valid_mask_threshold)).float()
        if valid_mask_erode_iters > 0:
            m = valid.unsqueeze(0).unsqueeze(0)
            for _ in range(int(valid_mask_erode_iters)):
                m = 1.0 - torch.nn.functional.max_pool2d(
                    1.0 - m, kernel_size=3, stride=1, padding=1,
                )
            valid = m.squeeze(0).squeeze(0)
        self.valid_mask = valid                                          # (H, W)
        n_valid = int(self.valid_mask.sum().item())
        n_total = height * width
        print(f"  Valid-pixel mask: {n_valid}/{n_total} pixels "
              f"({100.0 * n_valid / n_total:.1f}%) "
              f"(threshold={valid_mask_threshold:.3f}, "
              f"erode_iters={valid_mask_erode_iters})")

        # Pin memory for async H2D transfers under non_blocking=True.
        # Pinning can fail on systems with low locked-memory limits — fall
        # back silently if so; transfers still work, just slightly slower.
        try:
            self.rgb        = self.rgb.pin_memory()
            self.obj_mask   = self.obj_mask.pin_memory()
            self.hand_masks = [m.pin_memory() for m in self.hand_masks]
            if self.depth is not None:
                self.depth = self.depth.pin_memory()
            self.valid_mask = self.valid_mask.pin_memory()
        except RuntimeError:
            pass

    def get(self, t: int, device) -> dict:
        return {
            "rgb":       self.rgb[t].to(device, non_blocking=True),
            "obj_mask":  self.obj_mask[t].to(device, non_blocking=True),
            "hand_masks":[m[t].to(device, non_blocking=True) for m in self.hand_masks],
            "depth":     (self.depth[t].to(device, non_blocking=True)
                          if self.depth is not None else None),
        }


def _find_frame_file(frames_dir: str, fidx: int) -> str:
    for ext in (".png", ".jpg"):
        p = os.path.join(frames_dir, f"{fidx:06d}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Frame {fidx:06d} not found in {frames_dir}")


def _load_mask_or_zero(mask_dir: str, fidx: int, H: int, W: int) -> torch.Tensor:
    p = os.path.join(mask_dir, f"{fidx:06d}.png")
    if not os.path.exists(p):
        return torch.zeros((H, W), dtype=torch.float32)
    # Resize via PIL nearest-neighbor before decoding so binary mask
    # values stay {0, 1}. PIL.size is (W, H).
    img = Image.open(p)
    if img.size != (W, H):
        img = img.resize((W, H), Image.NEAREST)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr / arr.max() if arr.max() > 1.0 else arr
    return torch.from_numpy((arr > 0.5).astype(np.float32))


# ---------------------------------------------------------------------------
# Object mesh
# ---------------------------------------------------------------------------

def load_object_mesh(
    mesh_path: str, device: str
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load (N_vertices, 3) vertex positions, (N_vertices, 3) per-vertex
    diffuse colors in [0, 1], and (N_faces, 3) face indices.

    Vertex positions are in the mesh's own (canonical / object) frame.
    Faces are returned as a numpy array (consumed by trimesh-based SDF
    construction, no need to live on GPU).
    """
    scene = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(scene, trimesh.Scene):
        scene = trimesh.util.concatenate(list(scene.geometry.values()))
    verts = np.asarray(scene.vertices, dtype=np.float32)
    faces = np.asarray(scene.faces, dtype=np.int64)

    colors = None
    if hasattr(scene.visual, "vertex_colors") and scene.visual.vertex_colors is not None:
        vc = np.asarray(scene.visual.vertex_colors, dtype=np.float32)
        if vc.shape[0] == verts.shape[0] and vc.shape[1] >= 3:
            colors = vc[:, :3] / 255.0
    if colors is None and hasattr(scene.visual, "to_color"):
        try:
            vc = np.asarray(scene.visual.to_color().vertex_colors, dtype=np.float32)
            if vc.shape[0] == verts.shape[0] and vc.shape[1] >= 3:
                colors = vc[:, :3] / 255.0
        except Exception:
            pass
    if colors is None:
        colors = np.full((verts.shape[0], 3), 0.5, dtype=np.float32)

    return (
        torch.from_numpy(verts).to(device),
        torch.from_numpy(colors).to(device),
        faces,
    )
