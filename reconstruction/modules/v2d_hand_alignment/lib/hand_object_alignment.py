"""
HandObjectAlignment — align DynHaMR hand parameters to monocular depth via pyrender.

Extracted from experiments/mano.ipynb.  Pre-computes MANO FK for all frames in
__init__, then exposes compute_offset(side, t) which renders the hand mesh and
returns the camera-space [dx, dy, dz] shift needed to align it to the depth image.
"""

from __future__ import annotations

import json
import os

import numpy as np
import PIL.Image
import pyrender
import torch
import trimesh
from manotorch.manolayer import ManoLayer


os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

_CV_TO_GL = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


class HandObjectAlignment:
    """
    Args:
        pose_data_path:        Path to DynHaMR world_results.npz.
        depth_folder:          Folder of depth PNGs (000000.png, …).
        depth_intrinsics_path: JSON with {fx, fy, cx, cy, width, height}.
        mano_assets_root:      Root dir passed to ManoLayer (expects models/ subdir).
        occlusion_mask_folder: Per-frame mask PNGs; masked pixels excluded from
                               depth comparison.  None → no exclusion.
        image_folder:          RGB frames folder (optional; only needed for
                               visualisation helpers).
    """

    def __init__(
        self,
        pose_data_path: str,
        depth_folder: str,
        depth_intrinsics_path: str,
        mano_assets_root: str,
        occlusion_mask_folder: str | None = None,
        image_folder: str | None = None,
    ) -> None:
        self.pose_data = np.load(pose_data_path, allow_pickle=True)

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            side="right",
            center_idx=None,
            mano_assets_root=mano_assets_root,
        )

        self.depth_folder = depth_folder
        self.depth_intrinsics_path = depth_intrinsics_path
        self.occlusion_mask_folder = occlusion_mask_folder
        self.image_folder = image_folder

        self.left_verts_world,  self.left_verts_cam,  self.left_faces  = self._compute_hand(0)
        self.right_verts_world, self.right_verts_cam, self.right_faces = self._compute_hand(1)
        self.left_verts_2d  = self.render_verts_2d(0, self.pose_data['intrins'])
        self.right_verts_2d = self.render_verts_2d(1, self.pose_data['intrins'])

    # ------------------------------------------------------------------
    # Hand FK
    # ------------------------------------------------------------------

    def _compute_hand(self, side: int):
        """Run MANO FK for all frames.  Returns (verts_world, verts_cam, faces)."""
        pose_data = self.pose_data
        root_pose   = pose_data['root_orient'][side]                          # (T, 3)
        finger_pose = pose_data['pose_body'][side].reshape(
            pose_data['pose_body'].shape[1], -1)                              # (T, 45)
        hand_pose  = torch.from_numpy(np.concatenate([root_pose, finger_pose], axis=1))
        hand_betas = torch.from_numpy(pose_data['betas'][side])[None, :].repeat(
            hand_pose.shape[0], 1)

        mano_output = self.mano_layer(hand_pose, hand_betas)

        verts_world = (torch.from_numpy(pose_data['trans'][side][:, None, :])
                       + mano_output.verts)                                   # (T, 778, 3)

        if side == 0:  # left hand — flip x, reverse face winding
            verts_world[..., 0] = -verts_world[..., 0]
            faces = self.mano_layer.th_faces[:, [0, 2, 1]]
        else:
            faces = self.mano_layer.th_faces

        cam_R = torch.from_numpy(pose_data['cam_R'][side])   # (T, 3, 3)
        cam_t = torch.from_numpy(pose_data['cam_t'][side])   # (T, 3)
        verts_cam = (torch.bmm(cam_R, verts_world.transpose(1, 2)).transpose(1, 2)
                     + cam_t[:, None, :])                                     # (T, 778, 3)

        return verts_world, verts_cam, faces

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def get_depth_image(self, t: int) -> np.ndarray:
        """Load depth PNG → float32 metres (encoding: px = 65535/(depth+1))."""
        px = np.array(PIL.Image.open(
            os.path.join(self.depth_folder, f"{t:06d}.png"))).astype(np.float32)
        return 1.0 / (px / 65535.0) - 1.0

    def get_depth_intrinsics(self) -> np.ndarray:
        """Return [fx, fy, cx, cy] from the depth intrinsics JSON."""
        with open(self.depth_intrinsics_path) as f:
            d = json.load(f)
        return np.array([d["fx"], d["fy"], d["cx"], d["cy"]], dtype=np.float64)

    def get_image(self, t: int) -> PIL.Image.Image:
        if self.image_folder is None:
            raise RuntimeError("image_folder not provided")
        return PIL.Image.open(os.path.join(self.image_folder, f"{t:06d}.png"))

    def get_occlusion_mask(self, t: int) -> np.ndarray | None:
        """Return bool mask (True = occluded/object pixel), or None if unavailable."""
        if self.occlusion_mask_folder is None:
            return None
        path = os.path.join(self.occlusion_mask_folder, f"{t:06d}.png")
        if not os.path.exists(path):
            return None
        return np.array(PIL.Image.open(path)) > 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_verts_2d(self, side: int, intrinsics) -> torch.Tensor:
        """Project camera-space verts to pixel coords. Returns (T, 778, 3) [u, v, z]."""
        v_cam = self.left_verts_cam if side == 0 else self.right_verts_cam
        fx, fy, cx, cy = intrinsics
        z = v_cam[..., 2]
        u = (v_cam[..., 0] / z) * fx + cx
        v = (v_cam[..., 1] / z) * fy + cy
        return torch.stack([u, v, z], dim=-1)

    def get_mesh(self, side: int, t: int) -> trimesh.Trimesh:
        verts = self.left_verts_cam if side == 0 else self.right_verts_cam
        faces = self.left_faces     if side == 0 else self.right_faces
        return trimesh.Trimesh(verts[t].numpy(), faces.numpy(), process=False)

    def render_mesh(self, mesh: trimesh.Trimesh, resolution: tuple[int, int],
                    intrinsics) -> tuple[np.ndarray, np.ndarray]:
        """Render mesh with pyrender, returning (color, depth) arrays."""
        camera = pyrender.IntrinsicsCamera(
            fx=float(intrinsics[0]), fy=float(intrinsics[1]),
            cx=float(intrinsics[2]), cy=float(intrinsics[3]),
        )
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        scene.add(camera, pose=_CV_TO_GL)
        renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, depth = renderer.render(scene)
        renderer.delete()
        return color, depth

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def compute_offset(self, side: int, t: int) -> np.ndarray:
        """Camera-space [dx, dy, dz] to shift the hand to match the depth image.

        Renders the hand using DynHaMR intrinsics, computes the median depth
        difference over visible, non-occluded pixels, then scales dx/dy so the
        centroid moves along the camera ray.

        Returns:
            offset (3,) float64 in the same units as verts_cam (DynHaMR world
            units, since _compute_hand does not apply world_scale).
        """
        mesh         = self.get_mesh(side, t)
        depth_image  = self.get_depth_image(t)
        occ_mask     = self.get_occlusion_mask(t)
        h, w         = depth_image.shape
        _, depth     = self.render_mesh(mesh, (w, h), self.pose_data['intrins'])

        hand_mask = depth != 0
        if occ_mask is not None:
            hand_mask &= ~occ_mask

        dz = float(np.median(depth_image[hand_mask] - depth[hand_mask]))

        z = float(np.mean(mesh.vertices[:, 2]))
        x = float(np.mean(mesh.vertices[:, 0]))
        y = float(np.mean(mesh.vertices[:, 1]))
        z_p = z + dz
        return np.array([x / z * z_p - x,
                         y / z * z_p - y,
                         dz], dtype=np.float64)

    def _reproject(self, v: np.ndarray, src_intrinsics, target_intrinsics) -> np.ndarray:
        """Reproject camera-space point(s) from src to target intrinsics (z unchanged)."""
        fx,   fy,   cx,   cy   = src_intrinsics
        fx_p, fy_p, cx_p, cy_p = target_intrinsics
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        x_p = (z / fx_p) * (x / z * fx + cx - cx_p)
        y_p = (z / fy_p) * (y / z * fy + cy - cy_p)
        return np.stack([x_p, y_p, np.broadcast_to(z, x_p.shape)], axis=-1)

    def compute_offset_reprojected(self, side: int, t: int) -> np.ndarray:
        """Camera-space [dx, dy, dz] accounting for DynHaMR vs depth intrinsics.

        compute_offset renders with DynHaMR intrinsics, but the target depth image
        was produced with different intrinsics (MoGe / ViPE).  This method applies
        the raw depth offset to the hand centroid, then reprojects from DynHaMR
        pixel space into depth pixel space so the x/y shift is expressed in the
        depth image's coordinate system.

        Returns:
            offset (3,) float64 in depth-intrinsics camera space.
        """
        offset = self.compute_offset(side, t)          # (3,) under DynHaMR intrinsics

        verts   = self.right_verts_cam if side == 1 else self.left_verts_cam
        centroid = verts[t].numpy().mean(axis=0)        # (3,) original cam-space centroid

        shifted  = centroid + offset                    # centroid after depth shift

        reproj = self._reproject(
            shifted.reshape(1, 3),
            self.pose_data['intrins'],
            self.get_depth_intrinsics(),
        ).reshape(3)

        return (reproj - centroid).astype(np.float64)

    def compute_scale(self, side: int, t: int) -> float:
        """Depth scale factor: median(depth_image / rendered_depth) over hand pixels."""
        mesh         = self.get_mesh(side, t)
        depth_image  = self.get_depth_image(t)
        occ_mask     = self.get_occlusion_mask(t)
        h, w         = depth_image.shape
        _, depth     = self.render_mesh(mesh, (w, h), self.pose_data['intrins'])
        hand_mask = depth != 0
        if occ_mask is not None:
            hand_mask &= ~occ_mask
        return float(np.median(depth_image[hand_mask] / depth[hand_mask]))
