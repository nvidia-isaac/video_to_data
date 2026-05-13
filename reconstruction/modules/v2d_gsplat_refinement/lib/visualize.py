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
from .gaussians import HandGaussians, ObjectGaussians, concat_frames
from .io import HandPoseTrack, ObjectPoseTrack
from .pose_fields import HandPoseField, ObjectPoseField
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
    obj_state    = ckpt["obj_gaussians"]
    obj_n        = obj_state["anchor"].shape[0]
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
    hand_gaussians_list: list[HandGaussians]  = []
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

        # Gaussians — detect optional subsample.
        n_gauss = hg_state["_log_scale"].shape[0]
        subsample = hg_state.get("_subsample_idx", None)
        hg = HandGaussians(
            n_verts           = 778,
            is_right          = (side == "right"),
            init_scale        = 0.005,
            init_color        = torch.tensor([0.6, 0.45, 0.35], dtype=torch.float32, device=device),
            device            = device,
            subsample_indices = subsample if subsample is not None else None,
        ).to(device)
        # If subsample_indices was None at construct time but state has more
        # than 778 (shouldn't happen) we just trust load_state_dict to error.
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

    return {
        "frame_indices":   frame_indices,
        "T":               T,
        "obj_gaussians":   obj_gaussians,
        "obj_pose_field":  obj_pose_field,
        "hand_gaussians":  hand_gaussians_list,
        "hand_pose_fields": hand_pose_fields_list,
        "hand_sides":      hand_sides,
        "bg_gaussians":    bg_gaussians,
        "bg_pose_field":   bg_pose_field,
    }


def _build_per_frame_frames(modules: dict, t: int, show_obj: bool, show_hands: bool, show_bg: bool):
    """Build the (possibly empty) list of GaussianFrames for the current view."""
    frames = []
    if show_obj:
        R, tt = modules["obj_pose_field"](t)
        frames.append(modules["obj_gaussians"](R, tt))
    if show_hands:
        for hpf, hg in zip(modules["hand_pose_fields"], modules["hand_gaussians"]):
            v, R = hpf.posed_verts_and_rotmats_camera(t)
            frames.append(hg(v, R))
    if show_bg and modules["bg_gaussians"] is not None:
        R, tt = modules["bg_pose_field"](t)
        frames.append(modules["bg_gaussians"](R, tt))
    return frames


def _object_centroid(modules: dict, t: int) -> np.ndarray:
    """Object centroid in the current frame's camera coords; for orbit target."""
    R, tt = modules["obj_pose_field"](t)
    f = modules["obj_gaussians"](R, tt)
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
    device: torch.device,
) -> np.ndarray:
    """Single rasterization call → RGB uint8 (H, W, 3)."""
    from gsplat.rendering import rasterization

    frames = _build_per_frame_frames(modules, t, show_obj, show_hands, show_bg)
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
  0              source-camera view (identity)
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

    print(f"[visualize] loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    modules = _build_modules_from_ckpt(ckpt, mano_assets_root, device)
    print(f"[visualize] T = {modules['T']} frames, "
          f"object N = {modules['obj_gaussians'].num_gaussians()}, "
          f"hands = {len(modules['hand_sides'])} ({', '.join(modules['hand_sides']) or 'none'}), "
          f"bg = {'yes' if modules['bg_gaussians'] is not None else 'no'}")

    # --- Viewer state ------------------------------------------------------
    state = {
        "t":            0,
        "show_obj":     True,
        "show_hands":   True,
        "show_bg":      modules["bg_gaussians"] is not None,
        "mode":         "src",                  # 'src' or 'orbit'
        # Orbit state (used only in 'orbit' mode)
        "target":       _object_centroid(modules, 0),
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
        state["target"] = _object_centroid(modules, t)
        # Initial distance: enough to comfortably see the object.
        with torch.no_grad():
            R, tt = modules["obj_pose_field"](t)
            f = modules["obj_gaussians"](R, tt)
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

    while True:
        t = state["t"]

        if state["mode"] == "src":
            viewmat = np.eye(4)
        else:
            eye = _orbit_eye(state["target"], state["yaw"], state["pitch"], state["distance"])
            viewmat = _orbit_viewmat_cv(eye, state["target"], up=np.array([0.0, -1.0, 0.0]))

        img = _render(
            modules, t, K, W, H, viewmat,
            state["show_obj"], state["show_hands"], state["show_bg"],
            device,
        )
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        flags = (("O" if state["show_obj"]   else "_")
                 + ("H" if state["show_hands"] else "_")
                 + ("B" if state["show_bg"]    else "_"))
        info = [
            f"frame {modules['frame_indices'][t]:06d}  ({t+1}/{modules['T']})",
            f"mode  {state['mode']}   visible {flags}"
            + (f"   yaw={state['yaw']:+.2f} pitch={state['pitch']:+.2f} d={state['distance']:.3f}"
               if state["mode"] == "orbit" else ""),
            "[/]=frame   ,/.=10  space=play  1/2/3=obj/hands/bg  0=src  r=orbit  h=help  q=quit",
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
