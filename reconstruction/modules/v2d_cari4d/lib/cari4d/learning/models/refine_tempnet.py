# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from einops import rearrange, repeat
from pytorch3d.transforms import so3_exp_map, so3_log_map
import torch.nn as nn

from .util import AlphaBlender
from .network_modules import *
from Utils import *
from .feat_model import Encoder16x16
from .mhr_heads import enabled_mhr_head_specs
from .prediction_head import forward_contact_head_fp32
from learning.training.cuda_event_timing import cuda_timing_section
from learning.training.runtime_optimization import bounded_batch_slices
from learning.datasets.mhr_input_materialization import normalize_mhr_translation
from torchvision.transforms import functional as TVF

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def conv_config(conv: torch.nn.Conv2d) -> dict:
    """Return a dict with everything needed to rebuild `conv`."""
    return dict(
        in_channels   = conv.in_channels,
        out_channels  = conv.out_channels,
        kernel_size   = conv.kernel_size,
        stride        = conv.stride,
        padding       = conv.padding,
        dilation      = conv.dilation,
        groups        = conv.groups,
        bias          = conv.bias is not None,
        padding_mode  = conv.padding_mode,
        device        = conv.weight.device,
        dtype         = conv.weight.dtype,
    )


class SpatialTemporalBlock(nn.Module):
    "one spatial layer, one temporal layer"
    def __init__(self, embed_dim, num_heads=4, merge_factor=0.5, merge_strategy: str = "fixed",
                 spatial_posi_len=400, time_posi_len=180, dropout=0.1):
        super(SpatialTemporalBlock, self).__init__()
        self.temporal_attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
                                                        batch_first=True, dropout=dropout)
        self.spatial_attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
                                                       batch_first=True, dropout=dropout)
        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )
        self.spatial_posi = PositionalEmbedding(d_model=embed_dim, max_len=spatial_posi_len)
        self.time_posi = PositionalEmbedding(d_model=embed_dim, max_len=time_posi_len)
        print(f'Using {merge_strategy} blending strategy, default alpha={merge_factor}')

    def forward(self, x):
        "input x: (b, t, c, h, w), output (b, t, c, h, w)"
        x_in = x
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c (h w)')
        x = x.permute(0, 2, 1)
        x = self.spatial_posi(x)
        x_spatial = self.spatial_attn(x)
        x_spatial = rearrange(x_spatial, '(b t) c s -> (b c) t s', t=t)
        x = self.time_posi(x_spatial)
        x_time = self.temporal_attn(x) # (bc, t, s)
        x = self.time_mixer(x_spatial, x_time) # (bc, t, s), alpha * x_spatial + (1 - alpha) * x_time 
        x = rearrange(x, '(b c) t s -> b c t s', c=h*w)
        x = rearrange(x, 'b (h w) t c -> b h w t c', h=h)
        return x.permute(0, 3, 4, 1, 2) + x_in

class DINOTempRefineNet(nn.Module):
    ""
    def __init__(self, cfg, c_in=4):
        super(DINOTempRefineNet, self).__init__()
        self.cfg = cfg

        # This disables the specific function that triggers the 403 error, avoid frequent rate limit error.
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

        self.encoder_rgb = torch.hub.load('facebookresearch/dinov2', cfg.dino_model)
        for name, params in self.encoder_rgb.named_parameters():
            if 'feat_fuser' in name:
                params.requires_grad = True
                print(f"Setting {name} to require gradient")
            else:
                params.requires_grad = False
        dino_feat_dim_map = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
        }
        feat_dim_rgb = dino_feat_dim_map[cfg.dino_model]

        # Encoders for xyz+mask maps: DINO with more input channels, the whole model will be fine tuned
        self.encoder_xyzm = self.load_dino_with_more_channels(c_in, cfg.encoder_xyzm)
        feat_dim_xyzm = dino_feat_dim_map[cfg.encoder_xyzm]

        # Encoders for merging AB features
        if self.cfg.use_BN:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = None

        self.encoderAB_rgb = nn.Sequential(
            ResnetBasicBlock(feat_dim_rgb*2, feat_dim_rgb*2, bias=True, norm_layer=norm_layer), # this cannot compress feat dim
            ConvBNReLU(feat_dim_rgb*2, feat_dim_rgb, kernel_size=3, stride=1, norm_layer=norm_layer),
            ResnetBasicBlock(feat_dim_rgb, feat_dim_rgb, bias=True, norm_layer=norm_layer)
        )
        self.encoderAB_xyzm = nn.Sequential(
            ResnetBasicBlock(feat_dim_xyzm * 2, feat_dim_xyzm* 2, bias=True, norm_layer=norm_layer),
            ConvBNReLU(feat_dim_xyzm * 2, feat_dim_xyzm, kernel_size=3, stride=1, norm_layer=norm_layer),
            ResnetBasicBlock(feat_dim_xyzm, feat_dim_xyzm, bias=True, norm_layer=norm_layer)
        )

        # Merge features between RGB and xyz+mask
        embed_dim = cfg.embed_dim_spatial_temporal # 768
        self.init_feat_fuser(embed_dim, feat_dim_rgb, feat_dim_xyzm, norm_layer) # fuse rgb, mask and additional features

        # Spatial temporal attention layers
        merge_strategy: str = "fixed"
        merge_factor: float = 0.5
        num_heads = 4
        self.init_spatial_temporal_module(cfg, embed_dim, num_heads)
        self.time_posi = PositionalEmbedding(d_model=embed_dim, max_len=180)
        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )
        posi_len = 400 if 'posi_len' not in cfg else cfg['posi_len']
        self.spatial_posi = PositionalEmbedding(d_model=embed_dim, max_len=posi_len)

        self.init_pred_heads(embed_dim, num_heads)
        self.init_others()

    def init_others(self):
        pass

    def init_spatial_temporal_module(self, cfg, embed_dim, num_heads):
        if cfg.mixed_spatial_temporal:
            posi_len = 400 if 'posi_len' not in cfg else cfg['posi_len']
            modules = [SpatialTemporalBlock(cfg.embed_dim_spatial_temporal,
                                            spatial_posi_len=posi_len,
                                            dropout=cfg.dropout_1st,
                                            merge_strategy=cfg.merge_strategy,
                                            merge_factor=cfg.merge_factor) for x in range(cfg.num_attn_layers)]
            self.spatial_temp_list = nn.ModuleList(modules)
            print(f"Using mixed spatial temporal block: {len(self.spatial_temp_list)} layers")
        else:
            if cfg.num_attn_layers == 1:
                self.temporal_attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
                                                                batch_first=True, dropout=cfg.dropout_1st)
                self.spatial_attn = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
                                                               batch_first=True, dropout=cfg.dropout_1st)
            else:
                print(f"Using {cfg.num_attn_layers} attention layers")
                self.temporal_attn = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
                                               batch_first=True, dropout=cfg.dropout_1st), num_layers=cfg.num_attn_layers)
                self.spatial_attn = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
                                               batch_first=True, dropout=cfg.dropout_1st), num_layers=cfg.num_attn_layers)

    def init_pred_heads(self, embed_dim, num_heads):
        rot_out_dim = 3
        if self.cfg.pred_head_attn:
            self.trans_head = nn.Sequential(
                nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
                nn.Linear(512, 3 if not self.cfg.pred_uncertainty else 4 ),
            )

            self.rot_head = nn.Sequential(
                nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
                nn.Linear(512, rot_out_dim if not self.cfg.pred_uncertainty else rot_out_dim+1),
            )
            print('Using attention in prediction head!')
        else:
            blocks = [
                nn.Linear(embed_dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 3 if not self.cfg.pred_uncertainty else 4)
            ]
            self.trans_head = nn.Sequential(*blocks)
            blocks = [
                nn.Linear(embed_dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, rot_out_dim if not self.cfg.pred_uncertainty else rot_out_dim+1),
            ]
            self.rot_head = nn.Sequential(*blocks)
            print("Using non-spatial attention head!")


    def init_feat_fuser(self, embed_dim, feat_dim_rgb, feat_dim_xyzm, norm_layer):
        self.encoderAB_rgb_xyzm = nn.Sequential(
            ResnetBasicBlock(feat_dim_rgb + feat_dim_xyzm, feat_dim_rgb + feat_dim_xyzm, bias=True,
                             norm_layer=norm_layer),
            ConvBNReLU(feat_dim_rgb + feat_dim_xyzm, embed_dim, kernel_size=3, stride=1, norm_layer=norm_layer),
            ResnetBasicBlock(embed_dim, embed_dim, bias=True, norm_layer=norm_layer)
        )

    @staticmethod
    def load_dino_with_more_channels(c_in, model_name='dinov2_vits14'):
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        state = model.state_dict()

        # now modify the layers
        old_conv = model.patch_embed.proj  # the original layer
        cfg = conv_config(old_conv)
        cfg['in_channels'] = c_in  # <-- your new channel count
        new_conv = torch.nn.Conv2d(**cfg)
        model.patch_embed.proj = new_conv

        # init the layer
        # ❷ Expand the first conv
        old_w = state["patch_embed.proj.weight"]  # [E, 3, k, k]
        E, _, k, _ = old_w.shape
        new_w = torch.zeros(E, c_in, k, k)

        # good initialisations for the two extra channels --------------------------
        # A) copy the mean of RGB weights  (works well when extra bands look image-like)
        new_w[:, :3] = old_w
        new_w[:, 3:] = old_w.mean(1, keepdim=True).repeat(1, c_in - 3, 1, 1)

        # B) --OR-- random/He initialisation if the new bands are very different
        # --------------------------------------------------------------------------
        state["patch_embed.proj.weight"] = new_w  # bias stays identical

        missing_keys, unexpected_keys = model.load_state_dict(state, strict=True)
        return model

    def forward(self, A, B, obj_pose=None, batch=None):
        """
        concat DINO and our encoder features.
        A: (B, T, 6, H, W), RGB + xyz + HO mask
        B: (B, T, 8, H, W), RGB + xyz + HO mask
        pose: (B, T, 4, 4) absolute object pose
        """
        ab_merge = self.encode_input(A, B, obj_pose)
        bs, t, c, h, w = A.shape
        cuda_event_timer = getattr(self, "cuda_event_timer", None)
        with cuda_timing_section(cuda_event_timer, "temporal_blocks"):
            if self.cfg.mixed_spatial_temporal:
                x = rearrange(ab_merge, '(b t) c h w -> b t c h w', t=t)
                for i, module in enumerate(self.spatial_temp_list):
                    with cuda_timing_section(cuda_event_timer, f"temporal_block_{i}"):
                        x = module(x)
                x = rearrange(x, 'b t d h w -> (b t) (h w) d')
            else:
                bt, D, H, W = ab_merge.shape
                ab = ab_merge.reshape(bt, D, H*W).permute(0, 2, 1) # BT, D, HW -> BT, HW, D

                ############## Start of spatial temporal attention ##############
                ab = self.spatial_posi(ab)
                x_spatial = self.spatial_attn(ab)  # (bt, c, s) -> (bt, c, s) [64, 256, 512]
                c = x_spatial.shape[1]
                x = rearrange(x_spatial, '(b t) c s -> (b c) t s', t=t)  # 1600, 16, 512

                x = self.add_pose_feat_step2(x)
                x_p = self.time_posi(x)  # add temporal positional embedding
                x_time = self.temporal_attn(x_p)  # (bc, t, s)

                # blend spatial and temporal
                x = self.time_mixer(
                    x_spatial=x,
                    x_temporal=x_time
                )
                x = rearrange(x, '(b c) t s -> (b t) c s', c=c)
                ############## End of spatial temporal attention ##############

        # use the original prediction heads
        output = self.pred_head_forward(x, batch, B)

        return output

    def pred_head_forward(self, x, batch, B):
        "x: (BT, HW, D)"
        output = {}
        output['trans'] = self.trans_head(x).mean(dim=1)  # (bt, 3)
        output['rot'] = self.rot_head(x).mean(dim=1)  # (bt, 3)
        return output

    def encode_input(self, A, B, obj_pose):
        "return: (BT, D, H, W)"
        bs, t, c, h, w = A.shape
        A = rearrange(A, "b t c h w -> (b t) c h w")
        B = rearrange(B, "b t c h w -> (b t) c h w")
        bt = len(A)
        # Extract DINO features of RGB
        ab_rgb = self.encode_AB(A[:, :3], B[:, :3], self.encoder_rgb, self.encoderAB_rgb, profile_name="dino_rgb")  # (BT, D, H, W)
        # Extract DINO features of xyz + mask
        ab_xyzm = self.encode_AB(A[:, 3:], B[:, 3:], self.encoder_xyzm, self.encoderAB_xyzm, normalize=False,
                                 no_grad=False, profile_name="dino_xyzm")  # (BT, D, H, W)
        # Now merge RGB and xyzm features
        ab_merge = self.merge_features(ab_rgb, ab_xyzm, obj_pose)
        return ab_merge

    def add_pose_feat_step2(self, x_p):
        return x_p

    def merge_features(self, ab_rgb, ab_xyzm, obj_pose=None):
        "merge rgb, mask and any additional features"
        ab_merge = self.encoderAB_rgb_xyzm(torch.cat([ab_rgb, ab_xyzm], 1))  # (BT, D, H, W)
        return ab_merge

    def encode_AB(self, a, b, encoderA, encoderAB, normalize=True, no_grad=True, profile_name=None):
        bt = len(a)
        x = torch.cat([a, b], dim=0)  # shared encoder for RGB
        if normalize:
            xn = TVF.normalize(x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        else:
            xn = x  # xyz + mask cannot be normalized with imagenet parameters
        cuda_event_timer = getattr(self, "cuda_event_timer", None)
        with cuda_timing_section(cuda_event_timer, profile_name or "dino"):
            if no_grad:
                with torch.no_grad():
                    chunk_slices = bounded_batch_slices(len(xn), getattr(self.cfg, "frozen_dino_chunk_size", None))
                    tokens_all = [encoderA.forward_features(xn[chunk_slice])['x_norm_patchtokens'] for chunk_slice in chunk_slices]
                tokens = tokens_all[0] if len(tokens_all) == 1 else torch.cat(tokens_all, 0)
            else:
                # to avoid OOM
                chunk_size = 96
                tokens_all = []
                for i in range(0, bt*2, chunk_size):
                    out = encoderA.forward_features(xn[i:i+chunk_size])
                    tokens_all.append(out['x_norm_patchtokens'])
                tokens = torch.cat(tokens_all, 0)
        H, D = int(np.sqrt(tokens.shape[1])), tokens.shape[-1]
        a = tokens[:bt].permute(0, 2, 1).reshape(bt, D, H, H)
        b = tokens[bt:].permute(0, 2, 1).reshape(bt, D, H, H)  # (BT, D, H, W)
        ab = torch.cat((a, b), dim=1)
        with cuda_timing_section(cuda_event_timer, f"{profile_name}_pair_fuser" if profile_name else "dino_pair_fuser"):
            ab_merge = encoderAB(ab)
        return ab_merge

class DINOTempHORefineDeltaAndAbs(DINOTempRefineNet):
    def init_obj_pose_encoder(self, feat_dim_6dpose):
        ""
        obj_pose_encoder = nn.Sequential(nn.Linear(self.cfg.obj_pose_dim_input, feat_dim_6dpose // 4),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(feat_dim_6dpose // 4, feat_dim_6dpose//2),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(feat_dim_6dpose // 2, feat_dim_6dpose // 2),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(feat_dim_6dpose // 2, feat_dim_6dpose),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(feat_dim_6dpose, feat_dim_6dpose),
                                         )
        print('Using five layers of embedding')
        return obj_pose_encoder

    def init_others(self):
        "init human prediction head"
        """
                init a small conv2d encoder to extract global feature from grid features
                and a transformer to predict abs pose from rot, object mask info
                """
        embed_dim = self.cfg.embed_dim_spatial_temporal
        if self.is_mhr_body_model():
            self.init_mhr_prediction_heads(embed_dim)
            return

        self.encoder_mask = self.encoder_rgb if self.cfg.mask_dino_type == 'dinov2_vitb14' else torch.hub.load(
            'facebookresearch/dinov2', self.cfg.mask_dino_type)
        embed_dim = self.encoder_mask.embed_dim
        print(f'using {self.cfg.mask_dino_type} for binary mask encoding, embed dim={embed_dim}!')
        c_out = self.cfg.feat_dim_mask  # 768//4
        if c_out >= 1:
            self.mask_feat_fuser = Encoder16x16(embed_dim, c_out, embed_dim)
        else:
            print("Not using any mask information for 2nd transformer!")
        if self.cfg.bnorm_mask:
            self.mask_bnorm = nn.BatchNorm1d(c_out)
            print(f'normalizing mask feature of dimension: {c_out}')

        # transformer for abs pose prediction
        feat_dim_6dpose = self.cfg.feat_dim_6dpose
        obj_pose_encoder = self.init_obj_pose_encoder(feat_dim_6dpose)
        obj_pose_bnorm = nn.BatchNorm1d(feat_dim_6dpose)
        self.obj_pose_encoder = obj_pose_encoder
        self.obj_pose_bnorm = obj_pose_bnorm

        # transformer
        embed_dim = self.get_abspose_feats_dim()
        print(f'dropout={self.cfg.dropout} for abs pose encoding, embed dim={embed_dim}!')
        self.init_2nd_backbone(embed_dim)
        print('2nd transformer activation:', self.cfg.attn_activation, ', layers:', self.cfg.abspose_layers)
        print('Total number of parameters in 2nd network core:',
              sum(p.numel() for p in self.abspose_transformer.parameters() if p.requires_grad))

        out_dim = 3
        head_dims = [embed_dim] + list(self.cfg.pred_head_dims)
        blocks = self.get_pred_head_blocks(head_dims, out_dim)

        self.abshead_trans = nn.Sequential(*blocks)
        if self.cfg['rot_rep'] == 'axis_angle':
            rot_out_dim = 3
        elif self.cfg['rot_rep'] == '6d':
            rot_out_dim = 6
        else:
            raise RuntimeError
        blocks = self.get_pred_head_blocks(head_dims, rot_out_dim)
        self.abshead_rot = nn.Sequential(*blocks)

        if self.cfg.abspose_posi_encode:
            d_model = embed_dim if self.cfg.fp_err_dim < 0 else embed_dim - self.cfg.fp_err_dim
            d_model = d_model if self.cfg.visibility_dim < 0 else embed_dim - self.cfg.visibility_dim  # do not add any positional embedding to visibility info
            self.abspose_posi = PositionalEmbedding(d_model=d_model, max_len=180)  # for transformer

        # now add human cond info
        feat_dim_hum = self.cfg.hum_cond_embed_dim
        # also use 5 layers
        feat_dim_6dpose = feat_dim_hum
        hum_cond_dim = getattr(self.cfg, "mhr_cond_dim", self.cfg.hum_cond_dim) if self.is_mhr_body_model() else self.cfg.hum_cond_dim
        obj_pose_encoder = nn.Sequential(nn.Linear(hum_cond_dim, feat_dim_6dpose // 4),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(feat_dim_6dpose // 4, feat_dim_6dpose // 2),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(feat_dim_6dpose // 2, feat_dim_6dpose // 2),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(feat_dim_6dpose // 2, feat_dim_6dpose),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(feat_dim_6dpose, feat_dim_6dpose),
                                         )
        obj_pose_bnorm = nn.BatchNorm1d(feat_dim_hum)
        self.hum_encoder_pose = obj_pose_encoder
        self.hum_pose_bnorm = obj_pose_bnorm
        print('Adding human information')

        embed_dim = self.cfg.embed_dim_spatial_temporal
        self.init_legacy_human_prediction_heads(embed_dim)

    def is_mhr_body_model(self):
        return getattr(self.cfg, "body_model", "smpl") == "mhr"

    def init_mhr_prediction_heads(self, embed_dim):
        head_dims = [embed_dim] + list(self.cfg.pred_head_hum_pose)
        specs = enabled_mhr_head_specs(
            pred_shape=getattr(self.cfg, "pred_mhr_shape", False),
            pred_scale=getattr(self.cfg, "pred_mhr_scale", False),
            pred_face=getattr(self.cfg, "pred_mhr_face", False),
        )
        self.mhr_head_names = []
        for spec in specs:
            blocks = self.get_pred_head_blocks(head_dims, spec.dim)
            module_name = spec.output_key + "_head"
            setattr(self, module_name, nn.Sequential(*blocks))
            self.mhr_head_names.append((spec.output_key, module_name))
        print("Predicting native MHR residuals:", [name for name, _ in self.mhr_head_names])

        if self.cfg.cont_out_dim > 0:
            head_dims = list(self.cfg.pred_head_contact)
            blocks = self.get_pred_head_blocks(head_dims, self.cfg.cont_out_dim)
            self.contact_head = nn.Sequential(*blocks)
            print('Predicting MHR contact, head initialized!')

    def init_legacy_human_prediction_heads(self, embed_dim):
        # human pose and transl prediction head
        head_dims = [embed_dim] + list(self.cfg.pred_head_hum_pose)
        if self.cfg.rot_rep_hum == '6d':
            rot_dim = 6
        elif self.cfg.rot_rep_hum == 'axis':
            rot_dim = 3
        else:
            raise ValueError("Unknown rot_rep_hum={}".format(self.cfg.rot_rep_hum))
        if self.cfg.pred_uncertainty:
            rot_dim += 1
        blocks = self.get_pred_head_blocks(head_dims, 24 * rot_dim) # use 6d representation
        self.humpose_head = nn.Sequential(*blocks)

        head_dims = [embed_dim] + list(self.cfg.pred_head_hum_trans)
        if self.cfg.pred_uncertainty:
            blocks = self.get_pred_head_blocks(head_dims, 3 + 1)
        else:
            blocks = self.get_pred_head_blocks(head_dims, 3)
        self.humtrans_head = nn.Sequential(*blocks)

        if self.cfg.pred_shape:
            shape_dims = [embed_dim] + list(self.cfg.pred_head_hum_trans)
            blocks = self.get_pred_head_blocks(shape_dims, 10)
            self.hum_shape_head = nn.Sequential(*blocks)
            print('Predicting human shape, head initialized!')

        # prediction contact
        if self.cfg.cont_out_dim > 0:
            head_dims = list(self.cfg.pred_head_contact)
            blocks = self.get_pred_head_blocks(head_dims, self.cfg.cont_out_dim)
            self.contact_head = nn.Sequential(*blocks)
            print('Predicting contact, head initialized!')

    def init_2nd_backbone(self, embed_dim):
        "initialize the main component of the 2nd network"
        self.abspose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=self.cfg.nhead_2nd, dim_feedforward=self.cfg.d_ff_2nd,
                                       batch_first=True, dropout=self.cfg.dropout,
                                       activation=self.cfg.attn_activation), num_layers=self.cfg.abspose_layers)

    def get_pred_head_blocks(self, head_dims, out_dim):
        blocks = []
        for i in range(len(head_dims) - 1):
            blocks.append(nn.Linear(head_dims[i], head_dims[i + 1]))
            blocks.append(nn.LeakyReLU())
        blocks.append(nn.Linear(head_dims[-1], out_dim))
        return blocks

    def get_abspose_feats_dim(self):
        "add human dim"
        c_out = self.cfg.feat_dim_mask
        feat_dim_6dpose = self.cfg.feat_dim_6dpose
        embed_dim = c_out + feat_dim_6dpose
        if self.cfg.fp_err_dim > 0:
            embed_dim += self.cfg.fp_err_dim  # add FP errors to the feature as well
            print(f"Adding FP error as additional feature with dim={self.cfg.fp_err_dim}")
        if self.cfg.visibility_dim > 0:
            embed_dim += self.cfg.visibility_dim
            print(f'Adding object visibility feature of dim={self.cfg.visibility_dim}')
        dim = embed_dim
        dim += self.cfg.hum_cond_embed_dim
        return dim

    def forward(self, A, B, obj_pose=None, batch=None):
        "first module predicts delta, then compute an abs pose, to regress final abs pose "
        if self.cfg.input_gtpose_2nd or self.cfg.input_fp_2nd or self.cfg.pose_pred_dir is not None:
            # simply copy GT for the intermediate prediction
            out_delta  = {}
            b, rot_delta_gt, t, trans_delta_gt = self.comput_gt_obj_delta(B, batch)
            out_delta['rot'] = rot_delta_gt.reshape(b*t, -1)
            out_delta['trans'] = trans_delta_gt.reshape(b*t, -1) # use GT pose
        else:
            if self.cfg.train_stage == '2nd-only':
                # no grad from 1st stage
                with torch.no_grad():
                    self.eval()
                    out_delta = super().forward(A, B, obj_pose, batch) # this predicts relative pose
                    self.train()
            else:
                out_delta = super().forward(A, B, obj_pose, batch)  # this predicts relative pose

        if self.is_mhr_body_model():
            return out_delta
        self.predict_abspose(B, batch, obj_pose, out_delta)
        return out_delta

    def comput_gt_obj_delta(self, B, batch):
        b, t = B.shape[:2]
        trans_delta_gt = batch['delta_transl'].clone()  # (B, T, 3)
        if self.is_mhr_body_model():
            trans_delta_gt = normalize_mhr_translation(trans_delta_gt, batch)
        else:
            mesh_radius = batch['mesh_diameter'] / 2.  # (B, T)
            trans_delta_gt *= 1 / mesh_radius.reshape(len(trans_delta_gt), t, -1)
        rot_delta_mat_gt = batch['delta_rot'].clone()
        rot_delta_gt = so3_log_map(
            rot_delta_mat_gt.reshape(b * t, 3, 3).permute(0, 2, 1))  # permute: pyt3d so3 uses col order
        rot_delta_gt = rot_delta_gt / self.cfg['rot_normalizer']
        return b, rot_delta_gt, t, trans_delta_gt

    def get_body_joints(self, batch):
        if self.is_mhr_body_model():
            cond_key = getattr(self.cfg, "mhr_cond_key", "mhr_coco17_init")
            if cond_key in batch:
                return batch[cond_key]
            if "mhr_keypoints_init" in batch:
                return batch["mhr_keypoints_init"]
            raise KeyError(f"MHR conditioning key {cond_key} is missing from batch")
        joints_body = batch['joints_nlf']  # (B, T, 25, 3)
        return joints_body

    def pred_head_forward(self, x, batch, B):
        # x: (BT, HW, C)
        output = {}
        if self.cfg.train_smpl_only:
            # set the trans and rot to GT
            b, rot_delta_gt, t, trans_delta_gt = self.comput_gt_obj_delta(B, batch)
            output['rot'] = rot_delta_gt.reshape(b * t, -1)
            output['trans'] = trans_delta_gt.reshape(b * t, -1)  # use GT pose
        else:
            trans_out = self.trans_head(x).mean(dim=1)
            rot_out = self.rot_head(x).mean(dim=1)
            if not self.cfg.pred_uncertainty:
                output['trans'] = trans_out  # (bt, 3)
                output['rot'] = rot_out
            else:
                output['trans'] = trans_out[:, :-1]  # (bt, 3)
                output['rot'] = rot_out[:, :-1] # one more variable is the uncertainty
                output['trans_uncertainty'] = trans_out[:, -1:]
                output['rot_uncertainty'] = rot_out[:, -1:]
        if self.is_mhr_body_model():
            for output_key, module_name in self.mhr_head_names:
                output[output_key] = getattr(self, module_name)(x).mean(dim=1)
            if self.cfg.cont_out_dim > 0:
                output['contact'] = forward_contact_head_fp32(self.contact_head, x)
            return output

        hum_pose = self.humpose_head(x).mean(dim=1)
        hum_trans = self.humtrans_head(x).mean(dim=1)
        if not self.cfg.pred_uncertainty:
            output['hum_pose'] = hum_pose
            output['hum_trans'] = hum_trans
        else:
            output['hum_pose'] = hum_pose[:, :-24]
            output['hum_trans'] = hum_trans[:, :-1]
            output['hum_pose_uncertainty'] = hum_pose[:, -24:]
            output['hum_trans_uncertainty'] = hum_trans[:, -1:]
        # predict human shape
        if self.cfg.pred_shape:
            shape_out = self.hum_shape_head(x).mean(dim=1)
            output['hum_shape'] = shape_out
        if self.cfg.cont_out_dim > 0:
            contact_out = forward_contact_head_fp32(self.contact_head, x)
            output['contact'] = contact_out
        return output

    def additional_feats(self, batch, feat_comb):
        "add human info"
        joints_body = self.get_body_joints(batch)
        B, T = joints_body.shape[:2]
        joints_body = joints_body.reshape(B, T, -1)

        feat_hum = self.hum_encoder_pose(joints_body)
        feat_hum = self.hum_pose_bnorm(feat_hum.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, D)
        self.pose_feat_hum = feat_hum  # (B, T, D)
        feat_comb = torch.cat([feat_comb, feat_hum], -1)
        return feat_comb

    def predict_abspose(self, B, batch, obj_pose, out_delta):
        assert not self.cfg.train_smpl_only
        b, t = B.shape[:2]
        if self.cfg.train_stage == '1st-only':
            # train only the 1st stage model
            pose_gt = batch['pose_gt'].clone()
            self.cfg['rot_rep'] == '6d', 'only rot6d for abs pose!'
            trans_ref = pose_gt[:, 0:1, :3, 3].clone()
            out_delta['rot_abs'] = pose_gt[:, :, :3, :2].reshape(b * t, 6)
            out_delta['trans_abs'] = pose_gt[:, :, :3, 3].clone().reshape(-1, 3)
            out_delta['trans_abs_rela'] = (pose_gt[:, :, :3, 3].clone() - trans_ref).reshape(-1,
                                                                                             3)
            return

        B = rearrange(B, "b t c h w -> (b t) c h w")
        masks = B[:, 6:]  # rgb + xyz + 2 channel human + obj mask
        feat_mask = None
        if self.cfg.feat_dim_mask >= 1:
            if self.cfg.no_hum_mask:
                masks_in = TVF.normalize(torch.cat([masks[:, 1:2], torch.zeros_like(masks[:, 0:2])], 1),
                                         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            else:
                if masks.shape[1] == 2:  # add one additional all zero channel
                    if self.cfg.mask_encode_type == 'obj-fullobj':
                        # we just want to know the visibility of this frame.
                        masks = rearrange(batch['render_xyz'], "b t c h w -> (b t) c h w")[:, 3:]
                    masks_in = TVF.normalize(torch.cat([masks, torch.zeros_like(masks[:, 0:1])], 1),
                                             mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
                elif masks.shape[1] == 1:
                    # one channel mask
                    masks_in = TVF.normalize(masks.repeat(1, 3, 1, 1), mean=IMAGENET_DEFAULT_MEAN,
                                             std=IMAGENET_DEFAULT_STD)
                elif masks.shape[1] == 3:
                    # human + obj + obj full render
                    masks_in = TVF.normalize(masks, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
                else:
                    raise ValueError(f"Unknown mask shape {masks.shape}!")
            with torch.no_grad():
                out_mask = self.encoder_mask.forward_features(masks_in)
                tokens = out_mask['x_norm_patchtokens']
                H, D = int(np.sqrt(tokens.shape[1])), tokens.shape[-1]
                tokens = tokens.reshape(b * t, H, H, D).permute(0, 3, 1, 2)  # (BT, D, H, W)
            feat_mask = self.mask_feat_fuser(tokens).reshape(b, t, -1)
            if self.cfg.bnorm_mask:
                feat_mask = self.mask_bnorm(feat_mask.permute(0, 2, 1)).permute(0, 2, 1)
        # compute abs pose from prediction
        import Utils
        from collections.abc import Iterable
        mesh_diameter = batch['mesh_diameter'] if isinstance(batch, Iterable) else batch.mesh_diameters
        poseA = batch['pose_perturbed'] if isinstance(batch, Iterable) else batch.poseA

        trans_delta_final = out_delta['trans'] * mesh_diameter.reshape((-1, 1)) / 2.  # undo normalization
        rot_delta_final = so3_exp_map(out_delta['rot'] * self.cfg['rot_normalizer']).permute(0, 2, 1)
        B_in_cams = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4), trans_delta=trans_delta_final,
                                                        rot_mat_delta=rot_delta_final).reshape(b, t, 4,
                                                                                               4).detach()  # no gradient anymore here
        if self.cfg.input_gtpose_2nd:
            # here it is already different!
            B_in_cams = batch['pose_gt'].clone()
        elif self.cfg.input_fp_2nd:
            B_in_cams = batch['pose_perturbed'].clone()
        elif self.cfg.pose_pred_dir is not None:
            B_in_cams = batch['pose_pred'].clone()
        trans_ref = B_in_cams[:, 0:1, :3, 3].clone()  # (B, 1, 3) make sure to clone it!!!
        if self.cfg.exp_name == 'abs+delta1seq':
            trans_ref = B_in_cams[:, 0:1, :3, 3]  # for debug, and backward compatibility
            print('warning: using wrong translation reference!')

        obj_pose = B_in_cams.clone()
        obj_pose[:, :, :3, 3] = obj_pose[:, :, :3, 3] - trans_ref  # predict relative translation
        B, T = b, t
        if self.cfg.obj_pose_dim_input == 12:
            pose_in = obj_pose[:, :, :3].reshape(B, T, -1)
        elif self.cfg.obj_pose_dim_input == 9:
            pose_in = torch.cat([obj_pose[:, :, :3, 0:2], obj_pose[:, :, :3, 3:4]], -1).reshape(B, T, -1)
        elif self.cfg.obj_pose_dim_input == 3:
            # convert rot as axis angle and encode that only
            rot_axis = so3_log_map(obj_pose[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1))
            pose_in = rot_axis.reshape(B, T, -1)
        elif self.cfg.obj_pose_dim_input == 6:
            if self.cfg['rot_rep'] == '6d':
                pose_in = obj_pose[:, :, :3, 0:2].reshape(B, T, -1)
            else:
                rot_axis = so3_log_map(obj_pose[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1))
                pose_in = torch.cat([rot_axis.reshape(B, T, -1), obj_pose[:, :, :3, 3]], -1)
        else:
            raise NotImplementedError

        if self.cfg.mask_out_thres_1st < 5000:
            err = Utils.geodesic_distance_batch(B_in_cams.reshape(-1, 4, 4)[:, :3, :3],
                                                batch['pose_gt'].reshape(-1, 4, 4)[:, :3, :3])
            err_bt = err.reshape(b, t) / torch.pi  # normalize to 0-1
            mask = err_bt < self.cfg.mask_out_thres_1st  #
            pose_in = pose_in * mask.float()[:, :, None]

        pose_feat = self.obj_pose_encoder(pose_in)
        pose_feat = self.obj_pose_bnorm(pose_feat.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, D)
        self.pose_feat = pose_feat  # (B, T, D)
        feat_comb = torch.cat([feat_mask, pose_feat], -1) if feat_mask is not None else pose_feat
        feat_comb = self.additional_feats(batch, feat_comb)

        if self.cfg.abspose_posi_encode:
            feat_comb = self.abspose_posi(feat_comb)

        if self.cfg.fp_err_dim > 0:
            # do not let positional encoding affect it
            fp_error = batch['fp_error']  # (B, T)
            fp_error = fp_error[:, :, None].repeat(1, 1, self.cfg.fp_err_dim)
            feat_comb = torch.cat([feat_comb, fp_error], -1)
        if self.cfg.visibility_dim > 0:
            visibility = batch['visibility']
            visibility = visibility[:, :, None].repeat(1, 1, self.cfg.visibility_dim)
            feat_comb = torch.cat([feat_comb, visibility], -1)

        # now send to transformer to predict
        x = self.abspose_transformer(feat_comb)
        abs_trans = self.abshead_trans(x)
        abs_rot = self.abshead_rot(x)
        out_delta['rot_abs'] = abs_rot.reshape(b * t, abs_rot.shape[-1])
        out_delta['trans_abs'] = (abs_trans + trans_ref).reshape(-1, 3)
        out_delta['trans_abs_rela'] = abs_trans  # relative abs trans prediction


def additional_feats(self, batch, feat_comb):
        "add human info"
        joints_body = self.get_body_joints(batch)
        B, T = joints_body.shape[:2]
        joints_body = joints_body.reshape(B, T, -1)

        feat_hum = self.hum_encoder_pose(joints_body)
        feat_hum = self.hum_pose_bnorm(feat_hum.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, D)
        self.pose_feat_hum = feat_hum  # (B, T, D)
        feat_comb = torch.cat([feat_comb, feat_hum], -1)
        return feat_comb
