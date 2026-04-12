# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import trimesh
import viser

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compute_faces(basis_dirs: np.ndarray) -> np.ndarray:
    """Compute triangulation from the angular arrangement of 3D basis directions."""
    norms = np.linalg.norm(basis_dirs, axis=1, keepdims=True).clip(1e-8)
    unit_dirs = basis_dirs / norms
    hull = trimesh.convex.convex_hull(unit_dirs)
    return hull.faces


def build_mesh(
    supports: np.ndarray, basis_dirs: np.ndarray, faces: np.ndarray
) -> trimesh.Trimesh:
    """Build mesh with vertices at supports * basis_dirs using precomputed triangulation."""
    vertices = supports[:, None] * basis_dirs  # (N, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def make_basis_segments(basis_dirs: np.ndarray, scale: float = 0.3) -> np.ndarray:
    """Create line segments from origin to each basis direction."""
    n = basis_dirs.shape[0]
    points = np.zeros((n, 2, 3), dtype=np.float32)
    points[:, 1, :] = basis_dirs * scale
    return points


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class WrenchSupportData:
    """Parsed wrench-support arrays ready for visualisation."""

    force_basis: np.ndarray  # (N, 3)
    torque_basis: np.ndarray  # (N, 3)
    force_faces: np.ndarray
    torque_faces: np.ndarray
    sides: dict[str, np.ndarray]  # side_name -> supports (H, N)

    @property
    def num_timesteps(self) -> int:
        """Return the number of timesteps from the first side's support array."""
        return next(iter(self.sides.values())).shape[0]

    @staticmethod
    def from_pt(path: str, body_idx: int = 0) -> WrenchSupportData:
        """Load wrench support data from a ``.pt`` file.

        Args:
            path: Path to the ``.pt`` file.
            body_idx: Which body index to slice from the (H, nb, N) arrays.
        """
        data = torch.load(path)
        basis_np = data["basis"][body_idx].cpu().numpy()  # (N, 6)
        force_basis = basis_np[:, :3]
        torque_basis = basis_np[:, 3:]

        sides = {
            "right": data["right_contact_wrench_supports"][:, body_idx].cpu().numpy(),
            "left": data["left_contact_wrench_supports"][:, body_idx].cpu().numpy(),
        }
        return WrenchSupportData(
            force_basis=force_basis,
            torque_basis=torque_basis,
            force_faces=compute_faces(force_basis),
            torque_faces=compute_faces(torque_basis),
            sides=sides,
        )


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------


class WrenchSupportVisualizer:
    """Reusable viser-based visualiser for contact wrench supports."""

    def __init__(
        self,
        server: viser.ViserServer,
        data: WrenchSupportData,
        *,
        force_color: tuple[int, int, int] = (0, 255, 255),
        torque_color: tuple[int, int, int] = (255, 215, 0),
        force_x_offset: float = 0.0,
        torque_x_offset: float = 2.0,
        side_z_gap: float = 1.5,
        scene_prefix: str = "/wrench",
    ) -> None:
        """Initialise the visualiser and populate the scene."""
        self._server = server
        self._data = data
        self._force_color = force_color
        self._torque_color = torque_color
        self._force_x = force_x_offset
        self._torque_x = torque_x_offset
        self._prefix = scene_prefix
        self._mesh_handles: dict[str, Any] = {}
        self._playing = threading.Event()

        # Assign z-offsets per side, symmetrically around zero.
        side_names = list(data.sides.keys())
        n_sides = len(side_names)
        self._side_z: dict[str, float] = {}
        for i, name in enumerate(side_names):
            self._side_z[name] = side_z_gap * (1 - 2 * i / max(n_sides - 1, 1))

        self._add_static_scene()
        self._add_gui()
        self.update_frame(0)

    # -- static scene elements ------------------------------------------------

    def _add_static_scene(self) -> None:
        """Add reference frames, labels, and basis direction lines."""
        for side, z in self._side_z.items():
            for kind, x_off, basis_dirs in [
                ("force", self._force_x, self._data.force_basis),
                ("torque", self._torque_x, self._data.torque_basis),
            ]:
                pos = (x_off, 0.0, z)
                self._server.scene.add_frame(
                    f"{self._prefix}/{side}_{kind}_origin",
                    position=pos,
                    axes_length=0.2,
                    axes_radius=0.01,
                )
                self._server.scene.add_label(
                    f"{self._prefix}/{side}_{kind}_label",
                    text=f"{side.title()} {kind.title()}",
                    position=(x_off, 0.15, z),
                )
                self._server.scene.add_line_segments(
                    f"{self._prefix}/{side}_{kind}_basis",
                    points=make_basis_segments(basis_dirs),
                    colors=(200, 200, 200),
                    position=pos,
                )

    # -- dynamic mesh update --------------------------------------------------

    def update_frame(self, t: int) -> None:
        """Update the displayed wrench support meshes for timestep *t*."""
        for handle in self._mesh_handles.values():
            handle.remove()
        self._mesh_handles.clear()

        for side, supports_all in self._data.sides.items():
            supports = supports_all[t]
            z = self._side_z[side]

            force_mesh = build_mesh(
                supports, self._data.force_basis, self._data.force_faces
            )
            self._mesh_handles[f"{side}_force"] = self._server.scene.add_mesh_simple(
                name=f"{self._prefix}/{side}/force",
                vertices=np.array(force_mesh.vertices, dtype=np.float32),
                faces=np.array(force_mesh.faces, dtype=np.uint32),
                color=self._force_color,
                opacity=1.0,
                position=(self._force_x, 0.0, z),
            )

            torque_mesh = build_mesh(
                supports, self._data.torque_basis, self._data.torque_faces
            )
            self._mesh_handles[f"{side}_torque"] = self._server.scene.add_mesh_simple(
                name=f"{self._prefix}/{side}/torque",
                vertices=np.array(torque_mesh.vertices, dtype=np.float32),
                faces=np.array(torque_mesh.faces, dtype=np.uint32),
                color=self._torque_color,
                opacity=1.0,
                position=(self._torque_x, 0.0, z),
            )

    # -- GUI ------------------------------------------------------------------

    def _add_gui(self) -> None:
        """Wire up the timestep slider and play/pause button."""
        h = self._data.num_timesteps
        self._time_slider = self._server.gui.add_slider(
            label="Timestep", min=0, max=h - 1, step=1, initial_value=0
        )

        @self._time_slider.on_update
        def _on_slider_update(_: Any) -> None:
            self.update_frame(int(self._time_slider.value))

        play_button = self._server.gui.add_button(label="Play")

        @play_button.on_click
        def _on_play_click(_: Any) -> None:
            if self._playing.is_set():
                self._playing.clear()
            else:
                self._playing.set()

        threading.Thread(target=self._playback_loop, daemon=True).start()

    def _playback_loop(self) -> None:
        """Advance the timestep slider continuously while playing."""
        h = self._data.num_timesteps
        while True:
            if self._playing.is_set():
                t = (int(self._time_slider.value) + 1) % h
                self._time_slider.value = t
                self.update_frame(t)
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the wrench support visualiser from a ``.pt`` data file."""
    parser = argparse.ArgumentParser(description="Visualise contact wrench supports.")
    parser.add_argument(
        "--data", type=str, default="data.pt", help="Path to the .pt data file."
    )
    args = parser.parse_args()

    server = viser.ViserServer()
    data = WrenchSupportData.from_pt(args.data)
    WrenchSupportVisualizer(server, data)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
