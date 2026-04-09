import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# Entity type constants
ENTITY_BACKGROUND = 0
ENTITY_BODY = 1
ENTITY_LEFT_HAND = 2
ENTITY_RIGHT_HAND = 3
ENTITY_OBJECT_BASE = 4  # entity_id = ENTITY_OBJECT_BASE + rigid_body_id

# Inverse SH zeroth-order coefficient
C0 = 0.28209479177387814


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to SH DC coefficient."""
    return (rgb - 0.5) / C0


def sh_dc_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    """Convert SH DC coefficient to RGB [0,1]."""
    return sh_dc * C0 + 0.5


class GaussianScene(nn.Module):
    """
    Entity-typed 3D Gaussian scene.

    All Gaussians are stored in a flat array. entity_ids (fixed buffer) determines
    which deformation model applies to each Gaussian:
      - ENTITY_BACKGROUND: static (no deformation)
      - ENTITY_BODY: LBS via SMPL
      - ENTITY_OBJECT_BASE+k: SE(3) rigid body for object k
    """

    def __init__(
        self,
        positions: torch.Tensor,       # (N, 3) canonical positions
        colors: torch.Tensor,           # (N, 3) RGB in [0, 1]
        entity_ids: torch.Tensor,       # (N,) int
        skinning_weights: Optional[torch.Tensor] = None,  # (N_body, J)
        smpl_vertex_ids: Optional[torch.Tensor] = None,   # (N_body,) indices into SMPL mesh
    ):
        super().__init__()
        N = positions.shape[0]
        device = positions.device

        self._positions = nn.Parameter(positions.float().clone())

        q = torch.zeros(N, 4, device=device)
        q[:, 0] = 1.0  # w=1, identity quaternion
        self._rotations = nn.Parameter(q)

        # Small initial scale (~0.05 in world units after exp)
        self._log_scales = nn.Parameter(torch.full((N, 3), -3.0, device=device))

        # Low initial opacity (~0.05 after sigmoid)
        self._opacities_raw = nn.Parameter(torch.full((N, 1), -3.0, device=device))

        # SH degree 3: 16 coefficients, DC initialised from input colors, rest zero
        sh_dc = rgb_to_sh_dc(colors.float().clone()).unsqueeze(1)  # (N, 1, 3)
        self._sh_dc = nn.Parameter(sh_dc)
        self._sh_rest = nn.Parameter(torch.zeros(N, 15, 3, device=device))

        # Fixed entity labels
        self.register_buffer('entity_ids', entity_ids.int().clone())

        # Optional body skinning (only allocated when body entity is present)
        if skinning_weights is not None:
            self._skinning_weights_raw = nn.Parameter(skinning_weights.float().clone())
            self.register_buffer('smpl_vertex_ids', smpl_vertex_ids.long().clone())
        else:
            self._skinning_weights_raw = None
            self.register_buffer('smpl_vertex_ids', None)

        # Fixed reference positions for anchor losses (set after construction).
        # {rid: (N_obj, 3) tensor} for objects; body anchor is derived from smpl_vertex_ids.
        self._initial_obj_positions: dict = {}

    # ------------------------------------------------------------------ #
    # Activated properties
    # ------------------------------------------------------------------ #

    @property
    def positions(self) -> torch.Tensor:
        return self._positions

    @property
    def rotations(self) -> torch.Tensor:
        return F.normalize(self._rotations, dim=-1)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self._log_scales)

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._opacities_raw).squeeze(-1)  # (N,)

    @property
    def sh_features(self) -> torch.Tensor:
        return torch.cat([self._sh_dc, self._sh_rest], dim=1)  # (N, 16, 3)

    @property
    def skinning_weights(self) -> Optional[torch.Tensor]:
        if self._skinning_weights_raw is None:
            return None
        return F.softmax(self._skinning_weights_raw, dim=-1)

    # ------------------------------------------------------------------ #
    # Entity masks
    # ------------------------------------------------------------------ #

    def body_mask(self) -> torch.Tensor:
        return self.entity_ids == ENTITY_BODY

    def object_mask(self, rigid_body_id: int) -> torch.Tensor:
        return self.entity_ids == (ENTITY_OBJECT_BASE + rigid_body_id)

    def background_mask(self) -> torch.Tensor:
        return self.entity_ids == ENTITY_BACKGROUND

    def n_objects(self) -> int:
        max_id = int(self.entity_ids.max().item())
        return max(0, max_id - ENTITY_OBJECT_BASE + 1) if max_id >= ENTITY_OBJECT_BASE else 0

    @property
    def num_gaussians(self) -> int:
        return self._positions.shape[0]

    # ------------------------------------------------------------------ #
    # Densification helpers
    # ------------------------------------------------------------------ #

    def get_param_tensors_for_entity(self, mask: torch.Tensor) -> dict:
        """Gather all raw parameter tensors for a boolean mask (used in clone/split)."""
        return {
            'positions': self._positions[mask],
            'rotations': self._rotations[mask],
            'log_scales': self._log_scales[mask],
            'opacities_raw': self._opacities_raw[mask],
            'sh_dc': self._sh_dc[mask],
            'sh_rest': self._sh_rest[mask],
            'entity_ids': self.entity_ids[mask],
        }

    @classmethod
    def concat(cls, scenes: list) -> 'GaussianScene':
        """Concatenate multiple GaussianScene instances into one."""
        positions = torch.cat([s._positions.data for s in scenes])
        colors = torch.cat([sh_dc_to_rgb(s._sh_dc.data[:, 0, :]) for s in scenes])
        entity_ids = torch.cat([s.entity_ids for s in scenes])

        has_skinning = any(s._skinning_weights_raw is not None for s in scenes)
        sw = None
        vid = None
        if has_skinning:
            device = scenes[0]._positions.device
            n_joints = next(s._skinning_weights_raw.shape[1] for s in scenes if s._skinning_weights_raw is not None)
            sw_parts = [s._skinning_weights_raw.data if s._skinning_weights_raw is not None
                        else torch.zeros(s.body_mask().sum(), n_joints, device=device) for s in scenes]
            vid_parts = [s.smpl_vertex_ids if s.smpl_vertex_ids is not None
                         else torch.zeros(s.body_mask().sum(), dtype=torch.long, device=device) for s in scenes]
            sw = torch.cat(sw_parts)
            vid = torch.cat(vid_parts)

        new_scene = cls(positions, colors, entity_ids, sw, vid)

        # Copy all raw param data (not just DC color)
        new_scene._rotations.data = torch.cat([s._rotations.data for s in scenes])
        new_scene._log_scales.data = torch.cat([s._log_scales.data for s in scenes])
        new_scene._opacities_raw.data = torch.cat([s._opacities_raw.data for s in scenes])
        new_scene._sh_dc.data = torch.cat([s._sh_dc.data for s in scenes])
        new_scene._sh_rest.data = torch.cat([s._sh_rest.data for s in scenes])

        # Carry over initial object positions (used for anchor loss)
        for s in scenes:
            if s._initial_obj_positions:
                new_scene._initial_obj_positions.update(s._initial_obj_positions)

        return new_scene
