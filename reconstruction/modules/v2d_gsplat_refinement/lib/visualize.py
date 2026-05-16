"""Interactive OpenCV viewer for a v2d_gsplat_refinement checkpoint.

Reconstructs the trained Gaussian sets + pose fields from a checkpoint and
shows them in a free-orbit camera. Controls:

    Mouse
      LMB drag           orbit around target
      RMB drag           pan target (in screen plane)
      Wheel              zoom (changes orbit distance)

    Keyboard
      [   /   ]          previous / next frame
      ,   /   .          previous / next 10 frames
      space              toggle play (auto-advance at --fps)
      1                  toggle object visibility
      2                  toggle hands visibility
      3                  toggle background visibility
      w                  toggle MANO wireframe overlay (per-hand color)
      0                  reset to source-camera view (identity, frame intrinsics)
      r                  reset orbit (target = current object centroid)
      h                  print help to stdout
      q   /   ESC        quit

The "world" coordinate frame is the *current* timestep's camera frame —
each Gaussian set produces its means already in that frame's camera coords,
matching the convention used by ``_render_overlay_video_streaming``.

Usage:
    python -m v2d.gsplat_refinement.lib.visualize \\
        --checkpoint        /data/refine_checkpoint.pt \\
        --intrinsics_path   /data/intrinsics_stable.json \\
        --mano_assets_root  /data/weights/wilor/pretrained_models
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch

from .background import BackgroundGaussians, BackgroundPoseField
from .gaussians import (
    FaceGaussians,
    GaussianFrame,
    HandFaceGaussians,
    HandGaussians,
    ObjectGaussians,
    WristAttachedGaussians,
    concat_frames,
    quat_mul,
    rotmat_to_quat,
)
from .io import HandPoseTrack, ObjectPoseTrack
from .pose_fields import HandPoseField, IntrinsicsField, ObjectPoseField
from .refine import _orbit_viewmat_cv  # already in CV convention


def _load_K(path: str) -> tuple[torch.Tensor, int, int]:
    with open(path) as f:
        d = json.load(f)
    K = torch.tensor([
        [d["fx"], 0.0,    d["cx"]],
        [0.0,    d["fy"], d["cy"]],
        [0.0,    0.0,    1.0],
    ], dtype=torch.float32)
    return K, int(d["width"]), int(d["height"])


def _scale_K(K: torch.Tensor, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> torch.Tensor:
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K_out = K.clone()
    K_out[0, 0] *= sx; K_out[0, 2] *= sx
    K_out[1, 1] *= sy; K_out[1, 2] *= sy
    return K_out


def _build_modules_from_ckpt(
    ckpt: dict, mano_assets_root: str, device: torch.device,
) -> dict:
    """Reconstruct trained modules from a checkpoint state_dict bundle.

    The checkpoint only stores state_dicts, so each module is built with
    placeholder buffers of the right shape, then loaded.
    """
    frame_indices: list[int] = list(ckpt["frame_indices"])
    T = len(frame_indices)

    # --- Object pose + Gaussians ---------------------------------------
    # Detect anchor mode from state-dict keys: vertex-anchored has an
    # "anchor" buffer; face-anchored has "centroid_canon" + "TBN_canon".
    obj_state = ckpt["obj_gaussians"]
    if "centroid_canon" in obj_state:
        # Face-anchored object. Construct with a placeholder valid triangle
        # mesh of the right face count; load_state_dict overwrites buffers.
        obj_n = obj_state["centroid_canon"].shape[0]
        placeholder_verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32, device=device,
        )
        placeholder_faces = torch.zeros(obj_n, 3, dtype=torch.long, device=device)
        placeholder_faces[:, 1] = 1
        placeholder_faces[:, 2] = 2
        placeholder_face_colors = torch.full(
            (obj_n, 3), 0.5, dtype=torch.float32, device=device,
        )
        obj_gaussians = FaceGaussians(
            vertices    = placeholder_verts,
            faces       = placeholder_faces,
            face_colors = placeholder_face_colors,
        ).to(device)
    else:
        obj_n = obj_state["anchor"].shape[0]
        placeholder_anchor = torch.zeros(obj_n, 3, dtype=torch.float32, device=device)
        placeholder_color  = torch.full((obj_n, 3), 0.5, dtype=torch.float32, device=device)
        obj_gaussians = ObjectGaussians(
            anchor_positions = placeholder_anchor,
            init_color       = placeholder_color,
            init_scale       = 0.01,
        ).to(device)
    obj_gaussians.load_state_dict(obj_state)
    obj_gaussians.eval()

    obj_track = ObjectPoseTrack(
        rotations     = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * T, dtype=torch.float32, device=device),
        translations  = torch.zeros(T, 3, dtype=torch.float32, device=device),
        scales        = torch.ones(T, 3, dtype=torch.float32, device=device),
        frame_indices = frame_indices,
    )
    obj_pose_field = ObjectPoseField(obj_track).to(device)
    obj_pose_field.load_state_dict(ckpt["obj_pose_field"])
    obj_pose_field.eval()

    # --- Hands ---------------------------------------------------------
    hand_sides:      list[str]         = list(ckpt.get("hand_sides", []))
    hand_pose_dicts: list[dict]        = list(ckpt.get("hand_pose_fields", []))
    hand_gauss_dicts: list[dict]       = list(ckpt.get("hand_gaussians", []))
    hand_gaussians_list: list = []
    hand_pose_fields_list: list[HandPoseField] = []
    for side, hp_state, hg_state in zip(hand_sides, hand_pose_dicts, hand_gauss_dicts):
        # Pose field — dummy track of correct T then load.
        ht = HandPoseTrack(
            global_orient = torch.zeros(T, 3, dtype=torch.float32, device=device),
            hand_pose     = torch.zeros(T, 45, dtype=torch.float32, device=device),
            betas         = torch.zeros(T, 10, dtype=torch.float32, device=device),
            cam_t         = torch.zeros(T, 3, dtype=torch.float32, device=device),
            is_right      = (side == "right"),
            frame_indices = frame_indices,
            raw_records   = [{} for _ in range(T)],
        )
        hpf = HandPoseField(track=ht, mano_assets_root=mano_assets_root, device=device).to(device)
        hpf.load_state_dict(hp_state)
        hpf.eval()
        hand_pose_fields_list.append(hpf)

        # Detect anchor mode: face-anchored hands have a "_faces" buffer.
        if "_faces" in hg_state:
            n_faces = hg_state["_faces"].shape[0]
            # Placeholder rest mesh: non-degenerate triangle replicated for
            # n_faces so the constructor's TBN math doesn't NaN. State-dict
            # load then overwrites _faces / _log_scale / _delta_p / etc.
            placeholder_verts = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32, device=device,
            )
            placeholder_faces = torch.zeros(n_faces, 3, dtype=torch.long, device=device)
            placeholder_faces[:, 1] = 1
            placeholder_faces[:, 2] = 2
            hg = HandFaceGaussians(
                rest_vertices = placeholder_verts,
                faces         = placeholder_faces,
                is_right      = (side == "right"),
                init_color    = torch.tensor(
                    [0.6, 0.45, 0.35], dtype=torch.float32, device=device,
                ),
                device        = device,
            ).to(device)
        else:
            subsample = hg_state.get("_subsample_idx", None)
            hg = HandGaussians(
                n_verts           = 778,
                is_right          = (side == "right"),
                init_scale        = 0.005,
                init_color        = torch.tensor([0.6, 0.45, 0.35], dtype=torch.float32, device=device),
                device            = device,
                subsample_indices = subsample if subsample is not None else None,
            ).to(device)
        # If shapes mismatch (e.g. checkpoint from a different anchor mode
        # than expected), load_state_dict will raise — surfaces the issue
        # cleanly rather than silently corrupting state.
        hg.load_state_dict(hg_state)
        hg.eval()
        hand_gaussians_list.append(hg)

    # --- Background ----------------------------------------------------
    bg_gaussians  = None
    bg_pose_field = None
    if "bg_gaussians" in ckpt:
        bg_state = ckpt["bg_gaussians"]
        bg_n = bg_state["anchor"].shape[0]
        bg_gaussians = BackgroundGaussians(
            anchor_positions = torch.zeros(bg_n, 3, dtype=torch.float32, device=device),
            init_color       = torch.full((bg_n, 3), 0.5, dtype=torch.float32, device=device),
            init_scale       = torch.full((bg_n,),  0.01, dtype=torch.float32, device=device),
        ).to(device)
        bg_gaussians.load_state_dict(bg_state)
        bg_gaussians.eval()

        bg_pose_field = BackgroundPoseField(n_frames=T, device=device).to(device)
        bg_pose_field.load_state_dict(ckpt["bg_pose_field"])
        bg_pose_field.eval()

    # Learnable intrinsics (optional). Build with a placeholder K — the
    # state-dict load overwrites all four params + their init buffers, so
    # the placeholder values never leak through.
    intrinsics_field = None
    if "intrinsics_field" in ckpt:
        placeholder_K = torch.eye(3, dtype=torch.float32, device=device)
        intrinsics_field = IntrinsicsField(
            placeholder_K, learn_focal=False, learn_principal_point=False,
        ).to(device)
        intrinsics_field.load_state_dict(ckpt["intrinsics_field"])
        intrinsics_field.eval()

    # Wrist-attached Gaussians (optional, per-hand). Detect via checkpoint
    # "wrist_gaussians" key; entries may be None for hands without them.
    wrist_gaussians_list: list[WristAttachedGaussians | None] = []
    ckpt_wrist = ckpt.get("wrist_gaussians")
    for slot_idx in range(len(hand_pose_fields_list)):
        wg_state = None
        if ckpt_wrist is not None and slot_idx < len(ckpt_wrist):
            wg_state = ckpt_wrist[slot_idx]
        if wg_state is None:
            wrist_gaussians_list.append(None)
            continue
        n_wg = wg_state["anchor"].shape[0]
        wrist_gaussians_list.append(WristAttachedGaussians(
            anchor_positions = torch.zeros(n_wg, 3, dtype=torch.float32, device=device),
            init_color       = torch.full((n_wg, 3), 0.5, dtype=torch.float32, device=device),
            init_scale       = 0.03,
        ).to(device))
        wrist_gaussians_list[-1].load_state_dict(wg_state)
        wrist_gaussians_list[-1].eval()

    # Cache per-hand unique MANO edge list (for wireframe overlay).
    # Build once at load: faces are static, so the edge set is too.
    hand_edges_list: list[torch.Tensor] = []
    for hpf in hand_pose_fields_list:
        faces = hpf.mano.th_faces.to(device=device, dtype=torch.long)    # (F, 3)
        edges = torch.cat([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ], dim=0)                                                         # (3F, 2)
        edges, _ = torch.sort(edges, dim=1)                               # undirected
        edges = torch.unique(edges, dim=0)                                # dedupe
        hand_edges_list.append(edges)

    return {
        "frame_indices":   frame_indices,
        "T":               T,
        "obj_gaussians":   obj_gaussians,
        "obj_pose_field":  obj_pose_field,
        "hand_gaussians":  hand_gaussians_list,
        "hand_pose_fields": hand_pose_fields_list,
        "hand_edges":      hand_edges_list,
        "hand_sides":      hand_sides,
        "wrist_gaussians": wrist_gaussians_list,
        "intrinsics_field": intrinsics_field,
        "bg_gaussians":    bg_gaussians,
        "bg_pose_field":   bg_pose_field,
        "step_count":      int(ckpt.get("step_count", 0)),
    }


def _transform_frame_to_world(
    frame: GaussianFrame, R_w2c: torch.Tensor, t_w2c: torch.Tensor,
) -> GaussianFrame:
    """Pull a GaussianFrame from current-cam coords back into gsplat world.

    Object / hand gaussians always emit means in the current frame's camera
    coords (they apply the per-frame pose internally). When rendering with a
    static background, the world is the gsplat-world (reference-frame camera),
    so we undo (R_w2c, t_w2c) on means and per-Gaussian quats.
    """
    means_world = (frame.means - t_w2c) @ R_w2c                 # = R_w2c^T (x - t)
    q_w2c = rotmat_to_quat(R_w2c)                               # (4,) wxyz unit
    q_c2w = torch.stack([q_w2c[0], -q_w2c[1], -q_w2c[2], -q_w2c[3]])
    quats_world = quat_mul(q_c2w.expand_as(frame.quats), frame.quats)
    return GaussianFrame(
        means     = means_world,
        quats     = quats_world,
        scales    = frame.scales,
        opacities = frame.opacities,
        colors    = frame.colors,
    )


def _camera_frustum_segments_world(
    R_w2c: np.ndarray, t_w2c: np.ndarray, K: np.ndarray,
    W: int, H: int, depth: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """World-space line segments for a pyramid frustum at the given camera pose."""
    R_c2w = R_w2c.T
    cam_center = -R_c2w @ t_w2c
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth
    corners_cam = np.array([
        [(0 - cx) * z / fx, (0 - cy) * z / fy, z],
        [(W - cx) * z / fx, (0 - cy) * z / fy, z],
        [(W - cx) * z / fx, (H - cy) * z / fy, z],
        [(0 - cx) * z / fx, (H - cy) * z / fy, z],
    ], dtype=np.float64)
    corners_world = corners_cam @ R_c2w.T + cam_center
    starts, ends = [], []
    for c in corners_world:                                     # pyramid edges
        starts.append(cam_center); ends.append(c)
    for i in range(4):                                          # far-plane rect
        starts.append(corners_world[i]); ends.append(corners_world[(i + 1) % 4])
    return np.array(starts), np.array(ends)


def _draw_world_segments(
    img: np.ndarray, starts: np.ndarray, ends: np.ndarray,
    viewmat: np.ndarray, K: np.ndarray, color: tuple[int, int, int],
    thickness: int = 1, near: float = 0.01,
) -> np.ndarray:
    """Project world-space segments through (viewmat, K) and draw with cv2.line.

    Each segment is clipped to z >= near in view space before projection so a
    segment with one endpoint behind the camera still draws its visible part.
    """
    out = img
    R = viewmat[:3, :3]; t = viewmat[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    for s, e in zip(starts, ends):
        sc = R @ s + t
        ec = R @ e + t
        if sc[2] < near and ec[2] < near:
            continue
        if sc[2] < near:
            alpha = (near - sc[2]) / (ec[2] - sc[2])
            sc = sc + alpha * (ec - sc)
        elif ec[2] < near:
            alpha = (near - ec[2]) / (sc[2] - ec[2])
            ec = ec + alpha * (sc - ec)
        u0 = int(round(fx * sc[0] / sc[2] + cx))
        v0 = int(round(fy * sc[1] / sc[2] + cy))
        u1 = int(round(fx * ec[0] / ec[2] + cx))
        v1 = int(round(fy * ec[1] / ec[2] + cy))
        cv2.line(out, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
    return out


def _draw_world_segments_batched(
    img: np.ndarray,
    starts_world: np.ndarray,   # (E, 3)
    ends_world:   np.ndarray,   # (E, 3)
    viewmat: np.ndarray,        # (4, 4) world→cam
    K: np.ndarray,              # (3, 3)
    color: tuple[int, int, int],
    thickness: int = 1,
    near: float = 0.01,
) -> np.ndarray:
    """Vectorized projection + per-edge cv2.line draw.

    Same semantics as ``_draw_world_segments`` but with all matrix math
    done in numpy batches. Worth the rewrite when E is large (MANO has
    ~2300 edges; the per-segment Python loop in the legacy helper would
    cost ~50–200 ms per frame).
    """
    if starts_world.shape[0] == 0:
        return img
    R = viewmat[:3, :3]; t = viewmat[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    sc = starts_world @ R.T + t                                       # (E, 3)
    ec = ends_world   @ R.T + t                                       # (E, 3)

    keep = ~((sc[:, 2] < near) & (ec[:, 2] < near))
    sc = sc[keep]; ec = ec[keep]
    if sc.shape[0] == 0:
        return img

    # Clip endpoints that fall behind the near plane.
    s_behind = sc[:, 2] < near
    if s_behind.any():
        denom = (ec[s_behind, 2] - sc[s_behind, 2])
        alpha = ((near - sc[s_behind, 2]) / np.where(np.abs(denom) > 1e-6, denom, 1e-6))[:, None]
        sc[s_behind] = sc[s_behind] + alpha * (ec[s_behind] - sc[s_behind])
    e_behind = ec[:, 2] < near
    if e_behind.any():
        denom = (sc[e_behind, 2] - ec[e_behind, 2])
        alpha = ((near - ec[e_behind, 2]) / np.where(np.abs(denom) > 1e-6, denom, 1e-6))[:, None]
        ec[e_behind] = ec[e_behind] + alpha * (sc[e_behind] - ec[e_behind])

    u0 = np.rint(fx * sc[:, 0] / sc[:, 2] + cx).astype(np.int32)
    v0 = np.rint(fy * sc[:, 1] / sc[:, 2] + cy).astype(np.int32)
    u1 = np.rint(fx * ec[:, 0] / ec[:, 2] + cx).astype(np.int32)
    v1 = np.rint(fy * ec[:, 1] / ec[:, 2] + cy).astype(np.int32)

    for x0, y0, x1, y1 in zip(u0.tolist(), v0.tolist(), u1.tolist(), v1.tolist()):
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    return img


# Per-hand-slot wireframe colors (BGR). Cycles for >2 hands.
_HAND_WIREFRAME_COLORS = [
    (80, 220, 80),     # 0: green
    (220, 120, 80),    # 1: orange-ish
    (80, 220, 220),    # 2: yellow
    (220, 80, 220),    # 3: magenta
]


def _draw_hand_wireframes(
    img_bgr: np.ndarray,
    modules: dict,
    t: int,
    viewmat: np.ndarray,        # (4, 4) world→cam in current view
    K_np: np.ndarray,           # (3, 3)
    static_bg: bool,
    device: torch.device,
    thickness: int = 1,
) -> np.ndarray:
    """Overlay each hand's MANO mesh as wireframe lines on ``img_bgr``.

    Verts come straight from ``posed_verts_and_rotmats_camera`` (per-frame
    LBS + cam_t). When ``static_bg`` is on, they're transported into the
    gsplat-world frame so the view matches the rendered Gaussians.
    """
    with torch.no_grad():
        if static_bg and modules["bg_gaussians"] is not None:
            R_bg, t_bg = modules["bg_pose_field"](t)
            R_bg_np = R_bg.detach().cpu().numpy()
            t_bg_np = t_bg.detach().cpu().numpy()
        else:
            R_bg_np = None
            t_bg_np = None

        for i, (hpf, edges) in enumerate(
            zip(modules["hand_pose_fields"], modules["hand_edges"])
        ):
            v_cam, _ = hpf.posed_verts_and_rotmats_camera(t)              # (N, 3)
            v_cam_np = v_cam.detach().cpu().numpy()
            if R_bg_np is not None:
                # cam → world: v_world = R_bg.T @ (v_cam - t_bg).
                v_world_np = (v_cam_np - t_bg_np) @ R_bg_np
            else:
                v_world_np = v_cam_np
            edges_np = edges.detach().cpu().numpy()
            starts = v_world_np[edges_np[:, 0]]
            ends   = v_world_np[edges_np[:, 1]]
            color = _HAND_WIREFRAME_COLORS[i % len(_HAND_WIREFRAME_COLORS)]
            img_bgr = _draw_world_segments_batched(
                img_bgr, starts, ends, viewmat, K_np,
                color=color, thickness=thickness,
            )
    return img_bgr


def _build_per_frame_frames(
    modules: dict, t: int, show_obj: bool, show_hands: bool, show_bg: bool,
    static_bg: bool,
):
    """Build the (possibly empty) list of GaussianFrames for the current view.

    When ``static_bg`` is True, the rendering world is the gsplat-world
    (reference-frame camera) rather than the current frame's camera. The
    background uses identity pose, and object/hand frames are pulled back into
    world by inverting the current frame's background pose.
    """
    frames = []
    R_bg = t_bg = None
    if static_bg and modules["bg_gaussians"] is not None:
        R_bg, t_bg = modules["bg_pose_field"](t)
    if show_obj:
        R, tt = modules["obj_pose_field"](t)
        f = modules["obj_gaussians"](R, tt)
        if R_bg is not None:
            f = _transform_frame_to_world(f, R_bg, t_bg)
        frames.append(f)
    if show_hands:
        for hpf, hg, wg in zip(
            modules["hand_pose_fields"],
            modules["hand_gaussians"],
            modules.get("wrist_gaussians") or [None] * len(modules["hand_pose_fields"]),
        ):
            v, R = hpf.posed_verts_and_rotmats_camera(t)
            f = hg(v, R)
            if R_bg is not None:
                f = _transform_frame_to_world(f, R_bg, t_bg)
            frames.append(f)
            # Wrist-attached arm Gaussians: ride along with hand visibility.
            if wg is not None:
                Rw, tw = hpf.wrist_pose_camera(t)
                fw = wg(Rw, tw)
                if R_bg is not None:
                    fw = _transform_frame_to_world(fw, R_bg, t_bg)
                frames.append(fw)
    if show_bg and modules["bg_gaussians"] is not None:
        if static_bg:
            R = torch.eye(3, dtype=torch.float32, device=modules["bg_gaussians"].anchor.device)
            tt = torch.zeros(3, dtype=torch.float32, device=R.device)
        else:
            R, tt = modules["bg_pose_field"](t)
        frames.append(modules["bg_gaussians"](R, tt))
    return frames


def _object_centroid(modules: dict, t: int, static_bg: bool) -> np.ndarray:
    """Object centroid in the active world frame; used as orbit target."""
    R, tt = modules["obj_pose_field"](t)
    f = modules["obj_gaussians"](R, tt)
    if static_bg and modules["bg_gaussians"] is not None:
        R_bg, t_bg = modules["bg_pose_field"](t)
        f = _transform_frame_to_world(f, R_bg, t_bg)
    return f.means.mean(dim=0).detach().cpu().numpy()


def _orbit_eye(target: np.ndarray, yaw: float, pitch: float, distance: float) -> np.ndarray:
    """Spherical coords → eye position in the same frame as ``target``.

    Pitch and yaw are in the orbit's local frame: pitch=0 keeps the eye in the
    target's horizontal plane (CV-cam's +X / +Z axes). pitch>0 lifts the eye
    above the target. Yaw rotates around CV-cam's +Y (down) axis, so positive
    yaw moves the eye to the target's right in screen space.
    """
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw);   sy = math.sin(yaw)
    # Local direction from target to eye (CV cam: +X right, +Y down, +Z fwd).
    d = np.array([sy * cp, -sp, -cy * cp], dtype=np.float64)
    return target + d * distance


def _render(
    modules: dict, t: int, K: torch.Tensor, W: int, H: int,
    viewmat: np.ndarray,                       # (4,4) world→camera
    show_obj: bool, show_hands: bool, show_bg: bool,
    static_bg: bool,
    device: torch.device,
) -> np.ndarray:
    """Single rasterization call → RGB uint8 (H, W, 3)."""
    from gsplat.rendering import rasterization

    frames = _build_per_frame_frames(
        modules, t, show_obj, show_hands, show_bg, static_bg,
    )
    if not frames:
        return np.zeros((H, W, 3), dtype=np.uint8)
    combined = concat_frames(frames)

    Vt = torch.from_numpy(viewmat).to(device, dtype=torch.float32).unsqueeze(0)
    Kt = K.to(device).unsqueeze(0)
    with torch.no_grad():
        rgb, _, _ = rasterization(
            means      = combined.means,
            quats      = combined.quats,
            scales     = combined.scales,
            opacities  = combined.opacities,
            colors     = combined.colors,
            viewmats   = Vt,
            Ks         = Kt,
            width      = W,
            height     = H,
            near_plane = 0.01,
            far_plane  = 100.0,
            packed     = False,
        )
    img = rgb[0].clamp(0, 1).detach().cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def _annotate(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    pad = 8
    for i, line in enumerate(lines):
        y = pad + 16 + i * 18
        cv2.putText(out, line, (pad + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (pad, y),         cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


_HELP = """
Mouse:
  LMB drag       orbit around target
  RMB drag       pan target (screen plane)
  Wheel          zoom
Keyboard:
  [ / ]          frame -1 / +1
  , / .          frame -10 / +10
  space          play / pause
  1              toggle object
  2              toggle hands
  3              toggle background
  w              toggle MANO wireframe overlay
  s              toggle static-background world (frustum overlay in orbit mode)
  0              source-camera view
  r              reset orbit (target = current obj centroid)
  h              this help
  q / ESC        quit
"""


def visualize(
    checkpoint: str,
    intrinsics_path: str,
    mano_assets_root: str,
    width: int | None = None,
    height: int | None = None,
    fps: float = 30.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[visualize] device = {device}")

    K_full, W_full, H_full = _load_K(intrinsics_path)
    W = int(width)  if width  is not None else W_full
    H = int(height) if height is not None else H_full
    if (W, H) != (W_full, H_full):
        K = _scale_K(K_full, (W_full, H_full), (W, H))
    else:
        K = K_full
    print(f"[visualize] render @ {W}×{H}")

    def _load_modules(path: str) -> tuple[dict, float]:
        mtime = os.path.getmtime(path)
        ckpt_local = torch.load(path, map_location=device, weights_only=False)
        return _build_modules_from_ckpt(ckpt_local, mano_assets_root, device), mtime

    print(f"[visualize] loading checkpoint: {checkpoint}")
    modules, ckpt_mtime = _load_modules(checkpoint)
    print(f"[visualize] T = {modules['T']} frames, "
          f"object N = {modules['obj_gaussians'].num_gaussians()}, "
          f"hands = {len(modules['hand_sides'])} ({', '.join(modules['hand_sides']) or 'none'}), "
          f"bg = {'yes' if modules['bg_gaussians'] is not None else 'no'}")

    # If the checkpoint carries refined intrinsics, override the JSON's K.
    # The field's fx_init buffer equals JSON.fx × train_resolution_scale —
    # we recover that scale to express the refined K at native resolution,
    # then the existing _scale_K path scales to the visualizer's W/H.
    if modules.get("intrinsics_field") is not None:
        field = modules["intrinsics_field"]
        train_scale = float(field.fx_init.detach().cpu() / K_full[0, 0])
        if abs(train_scale) < 1e-6:
            print("[visualize] WARNING: intrinsics_field has fx_init~0; "
                  "skipping refined-K override.")
        else:
            K_trained = field.K().detach().cpu()                    # at training res
            K_refined_native = K_trained.clone()
            K_refined_native[0, 0] /= train_scale; K_refined_native[0, 2] /= train_scale
            K_refined_native[1, 1] /= train_scale; K_refined_native[1, 2] /= train_scale
            df = (K_refined_native - K_full).abs().max().item()
            print(f"[visualize] using refined intrinsics from checkpoint "
                  f"(max Δ vs JSON: {df:.3f} px; train_resolution_scale ≈ "
                  f"{train_scale:.3f}).")
            K_full = K_refined_native
            # Re-apply the visualizer's resolution scaling now that K_full
            # has been refreshed.
            if (W, H) != (W_full, H_full):
                K = _scale_K(K_full, (W_full, H_full), (W, H))
            else:
                K = K_full

    has_bg = modules["bg_gaussians"] is not None
    # --- Viewer state ------------------------------------------------------
    state = {
        "t":            0,
        "show_obj":       True,
        "show_hands":     True,
        "show_bg":        has_bg,
        "show_wireframe": False,
        "static_bg":      has_bg,                 # static-world mode (default on if bg)
        "mode":         "src",                  # 'src' or 'orbit'
        # Orbit state (used only in 'orbit' mode)
        "target":       _object_centroid(modules, 0, has_bg),
        "yaw":          0.0,
        "pitch":        0.0,
        "distance":     0.5,
        # Mouse
        "lmb":          False,
        "rmb":          False,
        "last_xy":      (0, 0),
        # Playback
        "playing":      False,
        "last_advance": 0.0,
    }

    def _reset_orbit_for_t(t: int) -> None:
        state["target"] = _object_centroid(modules, t, state["static_bg"])
        # Initial distance: enough to comfortably see the object.
        with torch.no_grad():
            R, tt = modules["obj_pose_field"](t)
            f = modules["obj_gaussians"](R, tt)
            if state["static_bg"] and modules["bg_gaussians"] is not None:
                R_bg, t_bg = modules["bg_pose_field"](t)
                f = _transform_frame_to_world(f, R_bg, t_bg)
            r = (f.means - torch.from_numpy(state["target"]).to(device)).norm(dim=-1).max().item()
        state["distance"] = max(0.2, r * 3.0)
        state["yaw"] = 0.0
        state["pitch"] = 0.0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["lmb"] = True; state["last_xy"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["lmb"] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["rmb"] = True; state["last_xy"] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            state["rmb"] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - state["last_xy"][0]
            dy = y - state["last_xy"][1]
            state["last_xy"] = (x, y)
            if state["mode"] != "orbit":
                # Any drag switches us into orbit mode for the current frame.
                if state["lmb"] or state["rmb"]:
                    _reset_orbit_for_t(state["t"])
                    state["mode"] = "orbit"
            if state["lmb"]:
                state["yaw"]   += dx * 0.005
                state["pitch"] += dy * 0.005
                state["pitch"]  = max(-1.4, min(1.4, state["pitch"]))
            elif state["rmb"]:
                # Pan in the orbit camera's screen plane.
                # Approximate: move target in -x and +y by drag magnitude * distance scale.
                scale = state["distance"] / float(W)
                # Build right/up vectors of the current orbit cam.
                eye  = _orbit_eye(state["target"], state["yaw"], state["pitch"], state["distance"])
                fwd  = state["target"] - eye
                fwd /= max(np.linalg.norm(fwd), 1e-8)
                world_up = np.array([0.0, -1.0, 0.0])  # CV-cam: +Y is down
                right = np.cross(world_up, fwd)
                rn = np.linalg.norm(right)
                if rn > 1e-6:
                    right /= rn
                up = np.cross(fwd, right)
                state["target"] = state["target"] - dx * scale * right + dy * scale * up
        elif event == cv2.EVENT_MOUSEWHEEL:
            # OpenCV encodes wheel delta in the high 16 bits of flags.
            delta = (flags >> 16)
            sign = 1 if delta > 0 else -1
            if state["mode"] != "orbit":
                _reset_orbit_for_t(state["t"])
                state["mode"] = "orbit"
            state["distance"] *= 0.9 ** sign
            state["distance"]  = max(0.02, min(50.0, state["distance"]))

    win = "v2d gsplat viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    cv2.setMouseCallback(win, on_mouse)
    print(_HELP)

    # Track last successful mtime separately from the in-flight observation
    # so a transient torch.load failure (e.g., we caught a rename mid-flight)
    # only triggers a retry, not a state corruption.
    last_loaded_mtime = ckpt_mtime
    while True:
        # --- hot-reload: poll checkpoint mtime each tick. The writer side
        # uses tmp+rename, so a successful stat → torch.load on the same path
        # is always self-consistent; we still try/except for robustness.
        try:
            new_mtime = os.path.getmtime(checkpoint)
        except OSError:
            new_mtime = last_loaded_mtime
        if new_mtime > last_loaded_mtime + 1e-6:
            try:
                new_modules, new_mtime_loaded = _load_modules(checkpoint)
            except Exception as e:
                print(f"[visualize] hot-reload skipped (transient): {e}")
            else:
                modules = new_modules
                last_loaded_mtime = new_mtime_loaded
                # Frame count and bg-presence can in principle change between
                # writes; clamp t and the bg toggles so the viewer stays sane.
                if state["t"] >= modules["T"]:
                    state["t"] = modules["T"] - 1
                if modules["bg_gaussians"] is None:
                    state["show_bg"] = False
                    state["static_bg"] = False
                print(f"[visualize] hot-reloaded checkpoint "
                      f"(T={modules['T']}, step="
                      f"{modules.get('step_count', '?')})")

        t = state["t"]

        # In static-bg mode, the src view sits at the per-frame camera pose,
        # so viewmat = world→cam_t from the background pose field. Otherwise
        # the world IS the current cam, so identity reproduces the source.
        if state["mode"] == "src":
            if state["static_bg"] and modules["bg_gaussians"] is not None:
                with torch.no_grad():
                    R_bg, t_bg = modules["bg_pose_field"](t)
                viewmat = np.eye(4)
                viewmat[:3, :3] = R_bg.detach().cpu().numpy()
                viewmat[:3, 3]  = t_bg.detach().cpu().numpy()
            else:
                viewmat = np.eye(4)
        else:
            eye = _orbit_eye(state["target"], state["yaw"], state["pitch"], state["distance"])
            viewmat = _orbit_viewmat_cv(eye, state["target"], up=np.array([0.0, -1.0, 0.0]))

        img = _render(
            modules, t, K, W, H, viewmat,
            state["show_obj"], state["show_hands"], state["show_bg"],
            state["static_bg"],
            device,
        )
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # MANO wireframe overlay (under the frustum so it doesn't get hidden).
        # Tied to hand visibility — hiding the Gaussians hides the wireframe too.
        if (state["show_wireframe"]
                and state["show_hands"]
                and modules["hand_pose_fields"]):
            K_np = K.detach().cpu().numpy()
            img_bgr = _draw_hand_wireframes(
                img_bgr, modules, t, viewmat, K_np,
                state["static_bg"], device,
            )

        # Camera frustum overlay: only meaningful in orbit mode with a static
        # world (in src mode the cam center is at the eye and degenerates;
        # in non-static mode the cam is always at world origin).
        if (state["static_bg"]
                and state["mode"] == "orbit"
                and modules["bg_gaussians"] is not None):
            with torch.no_grad():
                R_bg, t_bg = modules["bg_pose_field"](t)
            K_np = K.detach().cpu().numpy()
            starts, ends = _camera_frustum_segments_world(
                R_w2c=R_bg.detach().cpu().numpy(),
                t_w2c=t_bg.detach().cpu().numpy(),
                K=K_np, W=W, H=H,
                depth=max(0.015, 0.05 * state["distance"]),
            )
            img_bgr = _draw_world_segments(
                img_bgr, starts, ends, viewmat, K_np,
                color=(0, 220, 255), thickness=1,
            )
        flags = (("O" if state["show_obj"]       else "_")
                 + ("H" if state["show_hands"]     else "_")
                 + ("B" if state["show_bg"]        else "_")
                 + ("W" if state["show_wireframe"] else "_")
                 + ("S" if state["static_bg"]      else "_"))
        info = [
            f"frame {modules['frame_indices'][t]:06d}  ({t+1}/{modules['T']})  step {modules.get('step_count', 0)}",
            f"mode  {state['mode']}   visible {flags}"
            + (f"   yaw={state['yaw']:+.2f} pitch={state['pitch']:+.2f} d={state['distance']:.3f}"
               if state["mode"] == "orbit" else ""),
            "[/]=frame  ,/.=10  space=play  1/2/3=obj/hands/bg  w=wireframe  s=static  0=src  r=orbit  h=help  q=quit",
        ]
        cv2.imshow(win, _annotate(img_bgr, info))

        # Playback: advance frames at fps when 'playing'.
        wait_ms = 1 if state["playing"] else 30
        k = cv2.waitKey(wait_ms) & 0xFF
        if state["playing"]:
            now = time.monotonic()
            if now - state["last_advance"] >= 1.0 / max(fps, 0.1):
                state["t"] = (state["t"] + 1) % modules["T"]
                state["last_advance"] = now

        if k == 255 or k == 0:
            # No key — only the playback path drives advancement.
            pass
        elif k in (ord("q"), 27):
            break
        elif k == ord("["):
            state["t"] = max(0, state["t"] - 1)
        elif k == ord("]"):
            state["t"] = min(modules["T"] - 1, state["t"] + 1)
        elif k == ord(","):
            state["t"] = max(0, state["t"] - 10)
        elif k == ord("."):
            state["t"] = min(modules["T"] - 1, state["t"] + 10)
        elif k == ord(" "):
            state["playing"] = not state["playing"]
            state["last_advance"] = time.monotonic()
        elif k == ord("1"):
            state["show_obj"]   = not state["show_obj"]
        elif k == ord("2"):
            state["show_hands"] = not state["show_hands"]
        elif k == ord("3"):
            if modules["bg_gaussians"] is None:
                print("(no background in this checkpoint)")
            else:
                state["show_bg"] = not state["show_bg"]
        elif k == ord("w"):
            if not modules["hand_pose_fields"]:
                print("(no hands in this checkpoint)")
            else:
                state["show_wireframe"] = not state["show_wireframe"]
        elif k == ord("s"):
            if modules["bg_gaussians"] is None:
                print("(no background in this checkpoint — static mode requires bg)")
            else:
                state["static_bg"] = not state["static_bg"]
                # Orbit target lived in the previous world frame; recenter so
                # the view doesn't jump unexpectedly after the convention swap.
                if state["mode"] == "orbit":
                    _reset_orbit_for_t(state["t"])
        elif k == ord("0"):
            state["mode"] = "src"
        elif k == ord("r"):
            _reset_orbit_for_t(state["t"])
            state["mode"] = "orbit"
        elif k == ord("h"):
            print(_HELP)

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--intrinsics_path",  required=True,
                        help="JSON with {fx, fy, cx, cy, width, height} — "
                             "typically <output_dir>/intrinsics_stable.json.")
    parser.add_argument("--mano_assets_root", required=True,
                        help="Dir whose models/MANO_RIGHT.pkl is readable by "
                             "manotorch (e.g. <wilor_weights>/pretrained_models "
                             "or <hamer_weights>/_DATA/data).")
    parser.add_argument("--width",  type=int, default=None,
                        help="Override render width (defaults to intrinsics' width).")
    parser.add_argument("--height", type=int, default=None,
                        help="Override render height (defaults to intrinsics' height).")
    parser.add_argument("--fps",    type=float, default=30.0,
                        help="Playback FPS when space toggles play.")
    args = parser.parse_args()
    visualize(
        checkpoint       = args.checkpoint,
        intrinsics_path  = args.intrinsics_path,
        mano_assets_root = args.mano_assets_root,
        width            = args.width,
        height           = args.height,
        fps              = args.fps,
    )


if __name__ == "__main__":
    main()
