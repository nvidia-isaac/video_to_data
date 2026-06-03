# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gaussian primitive sets for joint hand+object refinement.

Two sets, both following the standard 3DGS parameterization (means / quats /
scales / opacities / colors), but with different per-frame deformation:

  ObjectGaussians  -- anchored to mesh vertices in object frame; per frame the
                      whole set is rigidly transformed by the object pose
                      (R_obj_t, t_obj_t).
  HandGaussians    -- anchored to MANO rest-pose vertices; per frame each
                      Gaussian's world position is the corresponding posed
                      vertex (LBS via manotorch). Hand Gaussians are isotropic
                      to avoid having to also rotate per-Gaussian quaternions
                      under articulation -- one less coupled DOF and the
                      photometric loss isn't sensitive to it at typical
                      Gaussian sizes.

Both sets expose canonical-frame parameters as ``nn.Parameter``s so they're
captured by a single ``optim.Adam(model.parameters(), ...)`` in the trainer.
Per-frame attributes are computed in ``forward(frame_idx)`` and returned as
a flat record the renderer consumes.

Object pose, MANO global rot+trans, MANO articulation, and (optionally) MANO
shape are *not* attributes of these classes -- they live in their own
``nn.Module`` wrappers (``ObjectPoseField`` / ``HandPoseField``) so the
canonical Gaussians are decoupled from the per-frame state.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def axis_angle_to_quat(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle → (..., 4) quaternion (w, x, y, z)."""
    angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication a*b. Both (..., 4) in (w, x, y, z)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) quat (w, x, y, z) → (..., 3, 3) rotation matrix."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=-1),
    ], dim=-2)


# ---------------------------------------------------------------------------
# Frame record (per-frame world-space Gaussians, ready for the rasterizer)
# ---------------------------------------------------------------------------

@dataclass
class GaussianFrame:
    """World-space Gaussian attributes for a single camera at a single frame.

    means:     (N, 3)
    quats:     (N, 4) -- (w, x, y, z), unnormalized OK; rasterizer normalizes
    scales:    (N, 3)
    opacities: (N,)   -- in [0, 1]
    colors:    (N, 3) -- in [0, 1] approx; rasterizer treats as RGB
    """
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor


def concat_frames(frames: list[GaussianFrame]) -> GaussianFrame:
    if len(frames) == 1:
        return frames[0]
    return GaussianFrame(
        means     = torch.cat([f.means     for f in frames], dim=0),
        quats     = torch.cat([f.quats     for f in frames], dim=0),
        scales    = torch.cat([f.scales    for f in frames], dim=0),
        opacities = torch.cat([f.opacities for f in frames], dim=0),
        colors    = torch.cat([f.colors    for f in frames], dim=0),
    )


# ---------------------------------------------------------------------------
# Object Gaussians: rigidly attached to a mesh in object frame
# ---------------------------------------------------------------------------

class ObjectGaussians(nn.Module):
    """One Gaussian per anchor point (typically mesh vertex), in object frame.

    Learnable params (all in canonical / object frame):
      _delta_p     : (N, 3)         -- offset from anchor, init 0
      _quat_canon  : (N, 4)         -- canonical orientation, init identity
      _log_scale   : (N, 3)         -- log of axis-aligned scales
      _opacity_logit: (N,)
      _color       : (N, 3)         -- raw RGB in [0, 1]-ish; clamped at render

    Anchor positions and the object pose are *inputs* to ``forward``.
    """

    def __init__(
        self,
        anchor_positions: torch.Tensor,    # (N, 3) in object frame
        init_color: torch.Tensor,          # (N, 3) in [0, 1]
        init_scale: float,                 # mean edge-length / 3 is typical
        init_opacity: float = 0.9,
    ) -> None:
        super().__init__()
        N = anchor_positions.shape[0]
        device = anchor_positions.device

        self.register_buffer("anchor", anchor_positions.contiguous())
        self._delta_p       = nn.Parameter(torch.zeros(N, 3, device=device))
        self._quat_canon    = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)
        )
        self._log_scale     = nn.Parameter(
            torch.full((N, 3), float(torch.log(torch.tensor(init_scale))),
                       device=device)
        )
        self._opacity_logit = nn.Parameter(
            torch.full((N,), float(_logit(init_opacity)), device=device)
        )
        self._color         = nn.Parameter(init_color.contiguous().clone())
        # Single global object scale, learned in log-space (init log=0 ⇒ s=1).
        # Multiplies anchor positions AND per-Gaussian scales so the visual
        # Gaussian sizes track geometric scale. Saved into the refined
        # Transform3d JSON's scale field at export time.
        self._log_scale_global = nn.Parameter(torch.zeros((), device=device))

    def num_gaussians(self) -> int:
        return self.anchor.shape[0]

    def forward(
        self,
        R_obj: torch.Tensor,    # (3, 3) object-to-camera rotation
        t_obj: torch.Tensor,    # (3,)   object-to-camera translation
    ) -> GaussianFrame:
        """Transform canonical Gaussians into camera/world frame for one camera.

        We render in *camera* frame (viewmat = identity), so the object pose
        is applied directly: positions go to camera frame, and so do quats.
        Global object scale ``s_obj = exp(_log_scale_global)`` multiplies
        canonical positions and per-Gaussian scales together — visual size
        and geometric extent stay coupled so the rendered object grows /
        shrinks coherently rather than producing a sparse-or-clumped splat
        cloud at the wrong scale.
        """
        s_obj = self._log_scale_global.exp()                          # ()
        # Canonical position in object frame, scaled by s_obj.
        p_obj = (self.anchor + self._delta_p) * s_obj                # (N, 3)
        # Apply object pose: x_cam = R_obj @ x_obj + t_obj.
        means = p_obj @ R_obj.T + t_obj                              # (N, 3)
        # Compose object rotation onto canonical quat: q_cam = q_obj * q_canon.
        q_obj = _rotmat_to_quat(R_obj).expand_as(self._quat_canon)   # (N, 4)
        quats = quat_mul(q_obj, self._quat_canon)                    # (N, 4)
        scales    = self._log_scale.exp() * s_obj                    # (N, 3)
        opacities = torch.sigmoid(self._opacity_logit)
        colors    = self._color
        return GaussianFrame(means, quats, scales, opacities, colors)

    def object_scale(self) -> torch.Tensor:
        """Current learned global object scale (scalar)."""
        return self._log_scale_global.exp()


# ---------------------------------------------------------------------------
# Hand Gaussians: anchored to MANO rest-pose verts, deformed each frame by
# manotorch (LBS). Isotropic so we don't have to articulate per-Gaussian quat.
# ---------------------------------------------------------------------------

class HandGaussians(nn.Module):
    """One Gaussian per MANO vertex (778 typically), per hand. Same
    parameterization as ``ObjectGaussians`` (anisotropic scale + per-Gaussian
    quaternion + Δp position offset) so the two sets have comparable degrees
    of freedom.

    Positions and rotations follow MANO's LBS each frame:
      - ``posed_verts_cam`` (from manotorch) is the MANO-rest vertex
        transported into world / camera frame.
      - ``per_vertex_rotmat_cam`` is the per-vertex deformation rotation in
        camera frame, computed by skinning the per-joint deformation
        rotations with MANO's vertex weights.

    Forward uses these to:
      - Apply Δp_canon as a rest-frame offset transformed by the vertex's
        LBS rotation, so Δp follows finger articulation.
      - Compose the LBS rotation onto the canonical per-Gaussian quat.

    Learnable params:
      _delta_p        : (N, 3)   -- Δp in MANO-rest frame, init 0
      _quat_canon     : (N, 4)   -- canonical orientation in rest frame, init identity
      _log_scale      : (N, 3)   -- anisotropic axis-aligned scale, init from skin spacing
      _opacity_logit  : (N,)
      _color          : (N, 3)
    """

    def __init__(
        self,
        n_verts: int,
        is_right: bool,
        init_scale: float,
        init_color: torch.Tensor,    # (3,) skin tone
        init_opacity: float = 0.9,
        device: str | torch.device = "cuda",
        subsample_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.is_right = bool(is_right)
        # Optionally allocate params for a strict subset of MANO vertices
        # to control hand Gaussian count without the LBS skinning getting
        # complex. The forward pass gathers the corresponding posed verts
        # and rotation matrices by these indices.
        if subsample_indices is not None:
            self.register_buffer(
                "_subsample_idx",
                subsample_indices.to(device, dtype=torch.long).contiguous(),
            )
            n_g = int(subsample_indices.numel())
        else:
            n_g = n_verts
        self._delta_p       = nn.Parameter(torch.zeros(n_g, 3, device=device))
        self._quat_canon    = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_g, 1)
        )
        self._log_scale     = nn.Parameter(
            torch.full((n_g, 3), float(torch.log(torch.tensor(init_scale))),
                       device=device)
        )
        self._opacity_logit = nn.Parameter(
            torch.full((n_g,), float(_logit(init_opacity)), device=device)
        )
        self._color = nn.Parameter(
            init_color.to(device).expand(n_g, 3).contiguous().clone()
        )

    def num_gaussians(self) -> int:
        return self._log_scale.shape[0]

    def forward(
        self,
        posed_verts_cam: torch.Tensor,       # (N_verts, 3)
        per_vertex_rotmat_cam: torch.Tensor, # (N_verts, 3, 3)
    ) -> GaussianFrame:
        # Subsample down to the Gaussian set if requested, otherwise use
        # all vertices.
        if hasattr(self, "_subsample_idx"):
            posed_verts_cam = posed_verts_cam[self._subsample_idx]
            per_vertex_rotmat_cam = per_vertex_rotmat_cam[self._subsample_idx]
        # Δp is in MANO-rest frame; the vertex's LBS rotation carries it into
        # world frame so it follows finger articulation rather than dangling
        # in fixed world coordinates.
        delta_world = torch.einsum("nij,nj->ni", per_vertex_rotmat_cam, self._delta_p)
        means = posed_verts_cam + delta_world
        # Compose LBS rotation onto canonical Gaussian rotation.
        q_lbs = rotmat_to_quat(per_vertex_rotmat_cam)            # (N, 4)
        quats = quat_mul(q_lbs, self._quat_canon)                # (N, 4)
        return GaussianFrame(
            means     = means,
            quats     = quats,
            scales    = self._log_scale.exp(),
            opacities = torch.sigmoid(self._opacity_logit),
            colors    = self._color,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    p = max(min(p, 1 - 1e-6), 1e-6)
    return float(torch.logit(torch.tensor(p)))


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) → (..., 4) quat in (w, x, y, z) convention. Differentiable
    and robust for arbitrary rotations.

    Wraps roma.rotmat_to_unitquat (which uses xyzw) and reorders to wxyz to
    match the convention used elsewhere in this module.
    """
    import roma  # local import — keeps this file usable without roma for tests
    q_xyzw = roma.rotmat_to_unitquat(R)                         # (..., 4)
    return torch.cat([q_xyzw[..., 3:4], q_xyzw[..., :3]], dim=-1)


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Single-matrix wrapper around ``rotmat_to_quat`` for backward
    compatibility with code that passes (3, 3)."""
    return rotmat_to_quat(R)


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def resample_mesh_surface(
    vertices: torch.Tensor,        # (V, 3)
    vertex_colors: torch.Tensor,   # (V, 3) in [0, 1]
    faces: "np.ndarray",            # (F, 3)
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``n`` points uniformly on the mesh surface with barycentric
    color interpolation. Decouples Gaussian count from mesh topology.

    Used to control object Gaussian density independent of whatever vertex
    count SAM3D / FoundationPose produced. Returns (positions, colors)
    matching the device/dtype of ``vertices``/``vertex_colors``.
    """
    import numpy as np
    import trimesh

    verts_np  = vertices.detach().cpu().numpy()
    colors_np = vertex_colors.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces, process=False)

    pts, face_idx = trimesh.sample.sample_surface_even(mesh, n)
    # sample_surface_even drops points that fail an even-spacing check;
    # top up via plain random sampling so we end up at exactly ``n``.
    if pts.shape[0] < n:
        extra_pts, extra_face = trimesh.sample.sample_surface(
            mesh, n - pts.shape[0])
        pts = np.vstack([pts, extra_pts])
        face_idx = np.concatenate([face_idx, extra_face])

    # Barycentric interpolation of vertex colors on the sampled triangles.
    face_verts = faces[face_idx]                                      # (n, 3)
    tri_verts  = mesh.vertices[face_verts]                            # (n, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(tri_verts, pts)    # (n, 3)
    tri_colors = colors_np[face_verts]                                # (n, 3, 3)
    sampled_colors = (bary[:, :, None] * tri_colors).sum(axis=1)      # (n, 3)

    return (
        torch.from_numpy(pts.astype(np.float32)).to(vertices.device),
        torch.from_numpy(sampled_colors.astype(np.float32)).to(vertices.device),
    )


def init_object_gaussians_from_mesh(
    vertices: torch.Tensor,          # (N, 3) object frame
    vertex_colors: torch.Tensor,     # (N, 3) in [0, 1]
    scale_factor: float = 1.0,       # Gaussian scale = scale_factor * mean
                                      # nearest-neighbor vertex distance.
                                      # 1.0 fills the mesh surface; smaller
                                      # values leave gaps that no single
                                      # Gaussian "owns".
) -> ObjectGaussians:
    """Initialize Gaussians at mesh vertices with scale set from inter-vertex spacing."""
    if vertices.shape[0] < 2:
        init_scale = 0.005
    else:
        # Approx mean nearest-neighbor distance via random pairs.
        n = min(vertices.shape[0], 4096)
        idx = torch.randperm(vertices.shape[0], device=vertices.device)[:n]
        sub = vertices[idx]
        d = torch.cdist(sub, sub)
        d.fill_diagonal_(float("inf"))
        init_scale = float(d.min(dim=1).values.mean()) * scale_factor
        init_scale = max(init_scale, 1e-4)
    return ObjectGaussians(
        anchor_positions = vertices,
        init_color       = vertex_colors,
        init_scale       = init_scale,
    )


def init_hand_gaussians(
    n_verts: int,
    is_right: bool,
    init_scale: float = 0.005,           # ~5 mm: MANO has ~5 mm vertex spacing
                                          # (778 verts spread over a ~10 cm hand);
                                          # smaller scales leave gaps between
                                          # Gaussians that no single splat
                                          # "owns", which weakens the per-vert
                                          # gradient and slows hand convergence.
    skin_tone: tuple[float, float, float] = (0.72, 0.55, 0.45),
    device: str | torch.device = "cuda",
    subsample_indices: torch.Tensor | None = None,
) -> HandGaussians:
    init_color = torch.tensor(skin_tone, dtype=torch.float32)
    return HandGaussians(
        n_verts           = n_verts,
        is_right          = is_right,
        init_scale        = init_scale,
        init_color        = init_color,
        device            = device,
        subsample_indices = subsample_indices,
    )


# ---------------------------------------------------------------------------
# Wrist-attached Gaussians: rigidly attached to a hand's wrist 6DOF pose.
#
# Same parameterization as ObjectGaussians (anchor + Δp + per-Gaussian quat
# + anisotropic scale + opacity + color), but per-frame transform comes from
# the hand's wrist (HandPoseField.batched_wrist_pose_camera), not an
# independent pose field. Used for arm geometry that we don't want to
# distort MANO to represent.
#
# Class labels are all-zero in the trainer's labels_static tensor — these
# Gaussians get no silhouette supervision (no arm mask), only photometric.
# ---------------------------------------------------------------------------

class WristAttachedGaussians(nn.Module):
    """Free 3D Gaussians attached to a hand wrist's rigid 6DOF pose.

    Learnable params (canonical wrist-local frame):
      _delta_p      : (N, 3)   -- offset from anchor, init 0. Loosely
                                  regularized so Gaussians can drift to fill
                                  the arm volume far from the wrist.
      _quat_canon   : (N, 4)
      _log_scale    : (N, 3)
      _opacity_logit: (N,)
      _color        : (N, 3)

    No ``_log_scale_global`` — there's no exported scale field for these.
    """

    def __init__(
        self,
        anchor_positions: torch.Tensor,    # (N, 3) in wrist-local frame
        init_color: torch.Tensor,          # (N, 3) in [0, 1]
        init_scale: float,                 # large by default (~3 cm) — these
                                            # are arm-sized blobs at init.
        init_opacity: float = 0.5,         # mid-opacity so opacity_binary can
                                            # drive each to either extreme.
    ) -> None:
        super().__init__()
        N = anchor_positions.shape[0]
        device = anchor_positions.device

        self.register_buffer("anchor", anchor_positions.contiguous())
        self._delta_p       = nn.Parameter(torch.zeros(N, 3, device=device))
        self._quat_canon    = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)
        )
        self._log_scale     = nn.Parameter(
            torch.full((N, 3), float(torch.log(torch.tensor(init_scale))),
                       device=device)
        )
        self._opacity_logit = nn.Parameter(
            torch.full((N,), float(_logit(init_opacity)), device=device)
        )
        self._color         = nn.Parameter(init_color.contiguous().clone())

    def num_gaussians(self) -> int:
        return self.anchor.shape[0]

    def forward(
        self,
        R_wrist: torch.Tensor,    # (3, 3) wrist-to-camera rotation
        t_wrist: torch.Tensor,    # (3,)   wrist-to-camera translation
    ) -> GaussianFrame:
        """Rigidly transform canonical Gaussians by the wrist pose."""
        p_local = self.anchor + self._delta_p                       # (N, 3)
        means   = p_local @ R_wrist.T + t_wrist                     # (N, 3)
        q_wrist = _rotmat_to_quat(R_wrist).expand_as(self._quat_canon)
        quats   = quat_mul(q_wrist, self._quat_canon)
        return GaussianFrame(
            means     = means,
            quats     = quats,
            scales    = self._log_scale.exp(),
            opacities = torch.sigmoid(self._opacity_logit),
            colors    = self._color,
        )


def init_wrist_attached_gaussians(
    n: int,
    init_scale: float = 0.03,                                       # ~3 cm
    init_radius: float = 0.0,                                        # 0 → all at origin
    skin_tone: tuple[float, float, float] = (0.72, 0.55, 0.45),
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> WristAttachedGaussians:
    """Sprinkle ``n`` Gaussians clustered at the wrist origin.

    ``init_radius=0`` → all anchors exactly at (0, 0, 0) in wrist-local frame.
    ``init_radius>0`` → anchors sampled uniformly in a ball of that radius
    (useful if you want some initial spatial spread; otherwise the optimizer
    has to break the perfect overlap via gradient noise).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if init_radius > 0:
        # Uniform-in-ball sampling.
        v = torch.randn(n, 3, generator=g)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        r = torch.rand(n, 1, generator=g) ** (1.0 / 3.0)
        anchors = (v * r * float(init_radius)).to(device)
    else:
        anchors = torch.zeros(n, 3, device=device)
    init_color = torch.tensor(skin_tone, dtype=torch.float32).expand(n, 3).contiguous()
    return WristAttachedGaussians(
        anchor_positions = anchors,
        init_color       = init_color.to(device),
        init_scale       = init_scale,
    )


# ---------------------------------------------------------------------------
# Face-anchored Gaussians
#
# One Gaussian per mesh face. Anchor = face centroid; orientation derived from
# the face's TBN frame (tangent = an edge, normal = face normal, bitangent =
# N x T). Δp is parameterized in face-local (T, B, N) coordinates, so its
# three components have physical meaning:
#     [0] tangent slide        — along the face surface
#     [1] bitangent slide      — along the face surface (orthogonal to T)
#     [2] normal depth         — into the volume (negative N) or outside (+N)
# Combined with an asymmetric per-axis regularizer (face_delta_p_regularizer in
# losses.py), this lets Gaussians slide freely on / sink into the mesh while
# being strongly penalized for leaking outside the surface.
#
# Same parameter-name conventions as ObjectGaussians / HandGaussians so the
# trainer's parameter-group builder works unchanged.
# ---------------------------------------------------------------------------

def _faces_to_centroid_and_tbn(
    vertices: torch.Tensor,    # (V, 3)
    faces:    torch.Tensor,    # (F, 3) long
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (centroid, TBN, mean_edge_length) for each face.

    Returns:
        centroid:        (F, 3) — face centroids.
        TBN:             (F, 3, 3) — columns are (T, B, N), an orthonormal frame
                         per face. T is the v0->v1 edge direction, N is the
                         outward face normal (using v0->v1 × v0->v2),
                         B = N × T.
        mean_edge_length: (F,) — mean of the three edge lengths, used as a
                         scale init for the Gaussian.
    """
    v0 = vertices[faces[:, 0]]                                      # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    centroid = (v0 + v1 + v2) / 3.0
    e01 = v1 - v0
    e02 = v2 - v0
    t = F.normalize(e01, dim=-1)
    n_unnorm = torch.cross(e01, e02, dim=-1)
    n = F.normalize(n_unnorm, dim=-1)
    b = torch.cross(n, t, dim=-1)
    TBN = torch.stack([t, b, n], dim=-1)                            # cols T,B,N
    edge_len = (
        e01.norm(dim=-1)
        + (v2 - v1).norm(dim=-1)
        + (v0 - v2).norm(dim=-1)
    ) / 3.0
    return centroid, TBN, edge_len


class FaceGaussians(nn.Module):
    """One Gaussian per mesh face, rigidly attached to the object frame.

    Forward signature matches ``ObjectGaussians``: ``(R_obj, t_obj)`` returns
    a per-frame ``GaussianFrame`` in camera-/world-frame.

    Learnable params (all interpreted in face-local frame):
      _delta_p     : (F, 3) — offset from face centroid in (T, B, N) coords.
                     Init 0. Component 2 is the outward-normal axis; one-sided
                     regularization on it expresses "stay inside the mesh".
      _quat_canon  : (F, 4) — extra rotation in the face-local frame, composed
                     onto the face's TBN. Init identity. Provides a small
                     amount of in-plane / cross-axis freedom on top of TBN.
      _log_scale   : (F, 3) — anisotropic axis-aligned log-scale. Init from
                     mean edge length: (edge/2, edge/2, edge/8) — disk-shaped
                     by default (thin along the normal axis).
      _opacity_logit: (F,)
      _color       : (F, 3) — per-face RGB; init as the mean of the three
                     vertex colors.
      _log_scale_global: scalar — same role as in ObjectGaussians; multiplies
                     positions and per-Gaussian scales together so the visual
                     extent tracks geometric extent.

    Buffers:
      centroid_canon: (F, 3) — face centroids in object/rest frame.
      TBN_canon:     (F, 3, 3) — orthonormal face frame in object/rest frame.
    """

    is_face_anchored = True

    def __init__(
        self,
        vertices: torch.Tensor,         # (V, 3) in object/rest frame
        faces:    torch.Tensor,         # (F, 3) long
        face_colors: torch.Tensor,      # (F, 3) in [0, 1]
        normal_thin_factor: float = 0.25,  # init normal-axis sigma = factor*tangent
        init_opacity: float = 0.9,
    ) -> None:
        super().__init__()
        device = vertices.device
        centroid, TBN, edge_len = _faces_to_centroid_and_tbn(vertices, faces)
        Fn = faces.shape[0]

        self.register_buffer("centroid_canon", centroid.contiguous())
        self.register_buffer("TBN_canon",      TBN.contiguous())

        self._delta_p       = nn.Parameter(torch.zeros(Fn, 3, device=device))
        self._quat_canon    = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(Fn, 1)
        )
        s_in_plane = (edge_len * 0.5).clamp_min(1e-4)               # (F,)
        s_normal   = (s_in_plane * float(normal_thin_factor)).clamp_min(1e-4)
        init_scale = torch.stack([s_in_plane, s_in_plane, s_normal], dim=-1)
        self._log_scale     = nn.Parameter(init_scale.log())
        self._opacity_logit = nn.Parameter(
            torch.full((Fn,), float(_logit(init_opacity)), device=device)
        )
        self._color         = nn.Parameter(face_colors.contiguous().clone())
        # Same role as in ObjectGaussians.
        self._log_scale_global = nn.Parameter(torch.zeros((), device=device))

    def num_gaussians(self) -> int:
        return self.centroid_canon.shape[0]

    def forward(
        self,
        R_obj: torch.Tensor,    # (3, 3) object-to-camera rotation
        t_obj: torch.Tensor,    # (3,)   object-to-camera translation
    ) -> GaussianFrame:
        s_obj = self._log_scale_global.exp()                                 # ()
        # Δp lives in face-local (T, B, N) coords; transport to canonical
        # (object) frame by TBN_canon, then scale together with the centroid.
        delta_canon = torch.einsum("fij,fj->fi", self.TBN_canon, self._delta_p)
        p_canon = (self.centroid_canon + delta_canon) * s_obj                # (F, 3)
        means = p_canon @ R_obj.T + t_obj                                    # (F, 3)
        # Per-face world rotation: R_obj @ TBN_canon composed with _quat_canon.
        TBN_world = torch.einsum("ij,fjk->fik", R_obj, self.TBN_canon)       # (F, 3, 3)
        q_face_world = rotmat_to_quat(TBN_world)                             # (F, 4)
        quats = quat_mul(q_face_world, self._quat_canon)                     # (F, 4)
        scales    = self._log_scale.exp() * s_obj                            # (F, 3)
        opacities = torch.sigmoid(self._opacity_logit)
        colors    = self._color
        return GaussianFrame(means, quats, scales, opacities, colors)

    def object_scale(self) -> torch.Tensor:
        return self._log_scale_global.exp()


class HandFaceGaussians(nn.Module):
    """One Gaussian per MANO face. TBN frame is recomputed each frame from
    the deformed face vertices — exact (no LBS-weight inheritance), because
    the three vertices' deformed positions fully determine the face's local
    frame and centroid.

    Forward signature matches ``HandGaussians``: ``(posed_verts_cam,
    per_vertex_rotmat_cam)``. The per-vertex rotmat input is *ignored* in
    this class (we derive the face rotation from the deformed face vertices
    instead).

    Learnable params: same names and roles as ``FaceGaussians`` (minus
    ``_log_scale_global``, matching HandGaussians).
    """

    is_face_anchored = True

    def __init__(
        self,
        rest_vertices: torch.Tensor,    # (V, 3) MANO rest-pose vertices
        faces:         torch.Tensor,    # (F, 3) MANO faces (long)
        is_right:      bool,
        init_color:    torch.Tensor,    # (3,) skin tone
        normal_thin_factor: float = 0.25,
        init_opacity:  float = 0.9,
        device: str | torch.device = "cuda",
        subsample_face_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.is_right = bool(is_right)
        rest_vertices = rest_vertices.to(device)
        faces = faces.to(device, dtype=torch.long)
        if subsample_face_indices is not None:
            faces = faces[subsample_face_indices.to(device, dtype=torch.long)]
        # Buffer the face indices so per-frame TBN reconstruction can look up
        # the right vertices from posed_verts_cam.
        self.register_buffer("_faces", faces.contiguous())

        # Canonical (rest-frame) centroid + TBN + edge length, used purely for
        # initialization (scale, color); the runtime path recomputes TBN.
        centroid, _TBN, edge_len = _faces_to_centroid_and_tbn(rest_vertices, faces)
        Fn = faces.shape[0]

        self._delta_p       = nn.Parameter(torch.zeros(Fn, 3, device=device))
        self._quat_canon    = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(Fn, 1)
        )
        s_in_plane = (edge_len * 0.5).clamp_min(1e-4)
        s_normal   = (s_in_plane * float(normal_thin_factor)).clamp_min(1e-4)
        init_scale = torch.stack([s_in_plane, s_in_plane, s_normal], dim=-1)
        self._log_scale     = nn.Parameter(init_scale.log())
        self._opacity_logit = nn.Parameter(
            torch.full((Fn,), float(_logit(init_opacity)), device=device)
        )
        self._color = nn.Parameter(
            init_color.to(device).expand(Fn, 3).contiguous().clone()
        )

    def num_gaussians(self) -> int:
        return self._faces.shape[0]

    def forward(
        self,
        posed_verts_cam: torch.Tensor,       # (N_verts, 3)
        per_vertex_rotmat_cam: torch.Tensor, # (N_verts, 3, 3) -- unused
    ) -> GaussianFrame:
        del per_vertex_rotmat_cam  # face rotation comes from vertex positions
        v0 = posed_verts_cam[self._faces[:, 0]]
        v1 = posed_verts_cam[self._faces[:, 1]]
        v2 = posed_verts_cam[self._faces[:, 2]]
        centroid = (v0 + v1 + v2) / 3.0
        e01 = v1 - v0
        e02 = v2 - v0
        t = F.normalize(e01, dim=-1)
        n_unnorm = torch.cross(e01, e02, dim=-1)
        n = F.normalize(n_unnorm, dim=-1)
        b = torch.cross(n, t, dim=-1)
        TBN_world = torch.stack([t, b, n], dim=-1)                  # (F, 3, 3)
        delta_world = torch.einsum("fij,fj->fi", TBN_world, self._delta_p)
        means = centroid + delta_world
        q_face_world = rotmat_to_quat(TBN_world)
        quats = quat_mul(q_face_world, self._quat_canon)
        return GaussianFrame(
            means     = means,
            quats     = quats,
            scales    = self._log_scale.exp(),
            opacities = torch.sigmoid(self._opacity_logit),
            colors    = self._color,
        )


def init_object_face_gaussians_from_mesh(
    vertices: torch.Tensor,        # (V, 3) object frame
    faces:    "np.ndarray | torch.Tensor",   # (F, 3)
    vertex_colors: torch.Tensor,   # (V, 3) in [0, 1]
    normal_thin_factor: float = 0.25,
) -> FaceGaussians:
    """One Gaussian per object face. ``faces`` may be numpy or torch."""
    import numpy as np
    if isinstance(faces, np.ndarray):
        faces_t = torch.from_numpy(faces.astype(np.int64)).to(vertices.device)
    else:
        faces_t = faces.to(vertices.device, dtype=torch.long)
    # Face colors = mean of the three vertex colors.
    face_colors = vertex_colors[faces_t].mean(dim=1)                # (F, 3)
    return FaceGaussians(
        vertices    = vertices,
        faces       = faces_t,
        face_colors = face_colors,
        normal_thin_factor = normal_thin_factor,
    )


def init_hand_face_gaussians(
    rest_vertices: torch.Tensor,   # (V, 3) MANO rest-pose vertices
    faces:         "np.ndarray | torch.Tensor",  # (F, 3) MANO faces
    is_right:      bool,
    skin_tone: tuple[float, float, float] = (0.72, 0.55, 0.45),
    normal_thin_factor: float = 0.25,
    hand_scale_init: float = 1.0,
    device: str | torch.device = "cuda",
    subsample_face_indices: torch.Tensor | None = None,
) -> HandFaceGaussians:
    import numpy as np
    if isinstance(faces, np.ndarray):
        faces_t = torch.from_numpy(faces.astype(np.int64))
    else:
        faces_t = faces.to(dtype=torch.long)
    init_color = torch.tensor(skin_tone, dtype=torch.float32)
    # If hand_scale enlarges the mesh, the face frame's edge_len enlarges too —
    # _faces_to_centroid_and_tbn reads scaled vertices, so pre-scale them here.
    rest_scaled = rest_vertices * float(hand_scale_init)
    return HandFaceGaussians(
        rest_vertices = rest_scaled,
        faces         = faces_t,
        is_right      = is_right,
        init_color    = init_color,
        normal_thin_factor = normal_thin_factor,
        device        = device,
        subsample_face_indices = subsample_face_indices,
    )
