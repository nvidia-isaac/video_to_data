"""
Hierarchical visual feature extractor for pose alignment.

Wraps DINOv2 to extract dense per-patch features from intermediate transformer
layers. These are more viewpoint-invariant than raw RGB and are used to
supervise FeatureGaussians during pose optimisation.

Features are extracted once per frame and cached — no gradient flows through
the encoder during optimisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FeatureExtractor(nn.Module):
    """
    Extract dense hierarchical features using DINOv2.

    encoder:  'dinov2_vits14' | 'dinov2_vitb14' | 'dinov2_vitl14'
    proj_dim: output dim per patch after linear projection (0 = raw encoder dim)
    levels:   list of ViT block indices (0-indexed) to extract and concatenate;
              defaults to [3, 6, 9, 11] for a 12-block ViT-S/14
    """

    def __init__(
        self,
        encoder: str = 'dinov2_vits14',
        proj_dim: int = 64,
        levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.encoder_name = encoder
        self.levels = levels or [3, 6, 9, 11]

        # Load frozen DINOv2 backbone.
        # skip_validation=True avoids the GitHub API call that hits rate limits
        # in network-restricted container environments.
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', encoder, pretrained=True, verbose=False,
            skip_validation=True,
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.patch_size: int = self.backbone.patch_size   # 14 for all vit*14 models
        raw_dim = self.backbone.embed_dim * len(self.levels)

        # Fixed frozen linear projection to proj_dim
        if proj_dim > 0 and proj_dim != raw_dim:
            proj = nn.Linear(raw_dim, proj_dim, bias=False)
            nn.init.orthogonal_(proj.weight)
            for p in proj.parameters():
                p.requires_grad_(False)
            self.proj = proj
            self._feature_dim = proj_dim
        else:
            self.proj = None
            self._feature_dim = raw_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """
        image:   (H, W, 3) float32 in [0, 1], any device
        returns: (h, w, D) where h = H // patch_size, w = W // patch_size
        """
        device = image.device
        p = self.patch_size
        H, W = image.shape[:2]

        # Crop to nearest multiple of patch_size
        H_c = (H // p) * p
        W_c = (W // p) * p
        img = image[:H_c, :W_c]

        # ImageNet normalisation expected by DINOv2
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)
        x = ((img - mean) / std).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H_c, W_c)

        backbone_device = next(self.backbone.parameters()).device
        feats = self.backbone.get_intermediate_layers(
            x.to(backbone_device),
            n=self.levels,
            return_class_token=False,
        )  # tuple of (1, N_patches, D_enc)

        h, w = H_c // p, W_c // p
        parts = [f[0].reshape(h, w, -1) for f in feats]   # each (h, w, D_enc)
        feat = torch.cat(parts, dim=-1)                    # (h, w, D_enc * n_levels)

        if self.proj is not None:
            feat = self.proj.to(device)(feat)              # (h, w, proj_dim)

        return feat.to(device)
