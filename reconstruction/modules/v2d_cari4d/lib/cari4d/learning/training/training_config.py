# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple,Union, Dict, Any
import numpy as np
import omegaconf
import torch


@dataclass
class TrainingConfig(omegaconf.dictconfig.DictConfig):
    input_resize: tuple = (160, 160)
    normalize_xyz:Optional[bool] = True
    use_mask:Optional[bool] = False
    crop_ratio:Optional[float] = None
    split_objects_across_gpus: bool = True
    max_num_key: Optional[int] = None
    use_normal:bool = False
    n_view:int = 1
    zfar:float = np.inf
    c_in:int = 6
    train_num_pair:Optional[int] = None
    make_pair_online:Optional[bool] = False
    render_backend:Optional[str] = 'nvdiffrast'

    # Run management
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    resume_run_id: Optional[str] = None
    save_dir: Optional[str] = None
    batch_size: int = 64
    epoch_size: int = 115200
    val_size: int = 1280
    n_epochs: int = 25
    save_epoch_interval: int = 100
    n_dataloader_workers: int = 20
    n_rendering_workers: int = 1
    gradient_max_norm:float = np.inf
    max_step_per_epoch: Optional[int] = 25000

    # Network
    use_BN:bool = True
    loss_type:Optional[str] = 'pairwise_valid'

    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1   # -1 means disable
    n_epochs_warmup: int = 1

    # Visualization
    vis_interval: Optional[int] = 1000

    debug: Optional[bool] = None



@dataclass
class TrainRefinerConfig:
    # Datasets
    input_resize: tuple = (160, 160)  #(W,H)
    crop_ratio:Optional[float] = None
    max_num_key: Optional[int] = None
    use_normal:bool = False
    use_mask:Optional[bool] = False
    normal_uint8:bool = False
    normalize_xyz:Optional[bool] = True
    mhr_xyz_anchor_type: str = 'root_joint_1' # center MHR XYZ conditioning on decoded kinematic root joint 1
    trans_normalizer:Optional[list] = None
    rot_normalizer:Optional[float] = None
    c_in:int = 6
    n_view:int = 1
    zfar:float = np.inf
    trans_rep:str = 'tracknet'
    rot_rep:Optional[str] = 'axis_angle'  # 6d/axis_angle
    save_dir: Optional[str] = None

    # Run management
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    batch_size: int = 64
    use_BN:bool = True
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1
    loss_type:str = 'l2'   # l1/l2/add

    vis_interval: Optional[int] = 1000
    debug: Optional[bool] = None


@dataclass
class LRSchedulerConfig:
    type: str = 'none'
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class LinearSchedulerConfig(LRSchedulerConfig):
    type: str = 'transformers'

    kwargs: Dict = field(default_factory=lambda: dict(
        name='linear',
        num_warmup_steps=0,
        num_training_steps="${max_steps}",
    ))


@dataclass
class DenoiserConfig:
    out_dim_hum: int = 157 # smpl pose 6d, transl, betas
    out_dim_obj: int = 9
    out_dim_contact: int = 0
    avgbeta: bool = True 
    latent_dim_hum: int = 512
    latent_dim_obj: int = 512   
    latent_dim_xt: int = 256
    latent_dim_contact: int = 0
    latent_dim: int = 1024
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_layers: int = 12
    num_heads: int = 8
    max_len: int = 180 # set this automatically to clip_len 
    layer_norm: bool = False
    bps_repr: str = 'none' # the bps input representation, none for no bps encoding. 
    bps_dim: int = 0 # the dimension of the bps encoding 
    pred_contact_points: bool = False # predict contact points 
    num_contact_points: int = 8 # number of contact points 

@dataclass
class DiffSchedulerInitArgsConfig:
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = "linear"
    prediction_type: str = "sample"  # predict x0

@dataclass
class DiffSchedulerFullConfig:
    init_args: DiffSchedulerInitArgsConfig = field(default_factory=DiffSchedulerInitArgsConfig)
    num_inference_steps: int = 5
    eta: float = 1.0 # with step 5 is the best
    disable_tqdm: bool = False
    contact_guidance: float = 0. # default none 
    cfg_prob: float = 0. # probability to use cfg training: zero out the cond probability
    cfg_weight: float = 0 # weight for the cfg training, zero means only cond, see https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#classifier-free-guidance 
    input_predx0: bool = False # input the predicted x0 as the cond, not the noisy x0 

@dataclass
class ContactOptimConfig:
    # optimization config for contact optimization
    body_model: str = 'smpl'
    contact_dist_thres: float = 0.2 # only compute loses smaller than this 
    opt_rot: bool = False # optimize the rotation of the object
    opt_trans: bool = True # optimize the translation of the object
    opt_betas: bool = False # optimize the betas of the object
    opt_smpl_pose: bool = True # optimize the smpl pose 
    opt_smpl_trans: bool = False # optimize the smpl translation

    # contact prediction config 
    contact_dim: int = 24 # contact dimension
    contact_pred_type: str = 'binary' # binary or distance 
    contact_mask_thres: float = 0.015 # contact mask threshold. 

    # loss weights
    w_contact: float = 100.0 # weight for contact loss
    w_acc_r: float = 100.0 # weight for temporal smoothness loss
    w_acc_t: float = 100.0 # weight for temporal smoothness loss
    w_acc_v: float = 100.0 # weight for temporal smoothness loss of object points 
    w_orig_t: float = 100.0 # weight for original translation loss
    w_cdir: float = 0.0 # weight for contact direction loss

    # input file
    pth_file: str = 'xxx' # path to the pth file, contains input, pr and GT 
    wild_video: bool = False # whether the video is in the wild
    data_source: str = 'behave' # behave or hodome 
    
    # opt configs
    lr: float = 0.001
    num_steps: int = 300
    batch_size: int = 384

    # logging
    no_wandb: bool = True 
    viz_steps: int = 500 
    save_every_n_steps: int = 500 
    save_name: str = 'contact'
    debug: bool = False
    use_gt: bool = False # use GT contacts or not 

    # data paths
    video_root: str = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/videos-demo'
    masks_root: str = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/masks-h5-my'
    packed_root: str = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/behave-packed'
    hy3d_meshes_root: str = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/selected-views/hy3d-aligned-center'
    index: Optional[int] = None # index of the video to use

@dataclass
class RefineOutOptimConfig(ContactOptimConfig):
    save_name: str = 'refineout'
    w_sil: float = 1e-3 # weight for silhouette loss 
    w_pen: float = 10.0 # weight for penetration loss 
    w_j2d: float = 0.02 # weight for 2D joint loss 
    w_contact: float = 0.1 # weight for contact loss 
    w_temp: float = 30.0 # weight for temporal smoothness loss 
    # w_velo: float = 30.0 # weight for velocity loss: object mainly 
    w_velo: float = 0.0 # weight for velocity loss: object mainly 
    w_init_ot: float = 100.0 # weight for initial object translation 
    w_init_ht: float = 8000.0 # weight for initial human translation, do not allow large human translation changes 
    w_pinit: float = 0 # weight for init pose 
    pen_loss_start: float = 0.6 # start of the pen loss, after this step, the pen loss is disabled 
    batch_size: int = 192

    opt_rot: bool = True # optimize the rotation of the object
    opt_trans: bool = True # optimize the translation of the object
    opt_betas: bool = False # optimize the betas of the object
    opt_smpl_pose: bool = True # optimize the smpl pose 
    opt_smpl_trans: bool = False # optimize the smpl translation
    op_thres: float = 0.3 # joint confidence threshold, 0.83 for v2.7 
    # apply oneeuro filter to results first
    oneeuro_type: str = 'none' # none, xyz, or all  

    use_input: bool = False # use the input 2D joints as the initial pose and translation
    outpath: str = 'output/opt'

@dataclass
class TrainTemporalRefinerConfig:
    # Datasets
    rgb_files: Optional[str] = 'xxx' # give the pattern to tar files
    render_files: Optional[str] = 'xxx'  # path to pre-rendered images

    split_file: Optional[str] = 'splits/behave-chairs.json'  # contains train, and val list
    val_split_file: Optional[str] = None # optional validation-only split file; accepts a val or test list
    render_root: Optional[str] = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/foundpose_train/behave-h5' # root to all renderings
    rgb_root: Optional[str] = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/30fps/h5-resized' # root to all rgb files
    packed_root: Optional[str] = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/behave-packed'  # root to packed files
    packed_format: str = 'pickle' # canonical target/init storage format: pickle or h5
    expected_train_sequence_count: Optional[int] = None # launch preflight contract for the configured training split
    expected_val_sequence_count: Optional[int] = None # launch preflight contract for the configured validation split
    fp_root: Optional[str] = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/fp' # root to fp files
    contacts_root: Optional[str] = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/contact-jts' # root to contact files
    body_model: str = 'smpl' # smpl legacy path or mhr native path
    mhr_gt_root: Optional[str] = None # root for native MHR target annotations
    mhr_init_root: Optional[str] = None # root for native MHR initialization annotations
    mhr_pseudogt_root: Optional[str] = None # validation-only official MHR geometry pseudo-ground truth
    mhr_sample_kids: Optional[Any] = None # optional camera ids used to build MHR training samples
    mhr_dataset_index_path: Optional[str] = None # prebuilt train dataset metadata index shared by all distributed ranks
    mhr_interaction_trim_root: Optional[str] = None # optional per-sequence interaction_trim.json root restricting train windows by source-frame range
    mhr_effective_mask_root: Optional[str] = None # canonical effective-mask H5 root used to reject windows containing unusable masks
    mhr_min_canonical_object_nonempty_frame_fraction: float = 0.10 # minimum sampled-frame fraction with a nonempty canonical object mask in every sampled camera
    mhr_required_contact_revision: Optional[str] = None # required semantic schema for native MHR contact supervision
    val_packed_root: Optional[str] = None # validation override for packed_root
    val_packed_format: Optional[str] = None # validation override for packed_format
    val_render_root: Optional[str] = None # validation override for render_root
    val_mhr_gt_root: Optional[str] = None # validation override for mhr_gt_root
    val_mhr_init_root: Optional[str] = None # validation override for mhr_init_root
    val_mhr_pseudogt_root: Optional[str] = None # validation override for mhr_pseudogt_root
    val_mhr_sample_kids: Optional[Any] = None # validation-only camera ids
    val_mhr_dataset_index_path: Optional[str] = None # prebuilt validation dataset metadata index shared by all distributed ranks
    val_mhr_interaction_trim_root: Optional[str] = None # optional validation interaction-trim root; unset prevents train trim metadata from leaking into validation
    val_mhr_effective_mask_root: Optional[str] = None # optional validation canonical effective-mask H5 root
    val_mhr_required_contact_revision: Optional[str] = None # validation override for the required MHR contact schema
    val_clip_len: Optional[int] = None # validation override for clip_len
    val_window: Optional[int] = None # validation override for window
    require_mhr_geometry: bool = False # require MHR joints/keypoints while loading dataset
    mhr_load_init_geometry: bool = False # deprecated; persisted MHR vertices are not supported
    clip_len: int = 96 # temporal length of one clip, 96 & bs=8 maximizes the GPU memory usage
    window: int = 1 # distance between two clip start point
    mhr_window_sampling_mode: str = 'strides_1_10_max_sampled_overlap_v1' # future MHR runs enumerate fixed temporal strides with bounded overlap
    mhr_window_temporal_strides: List[int] = field(default_factory=lambda: list(range(1, 11))) # fixed frame stride within each indexed MHR clip
    mhr_window_max_overlap_frames: int = 10 # maximum shared sampled frames between retained same-stride clips
    mhr_window_sampling_contract_path: Optional[str] = None # run-persistent contract used by index building and checkpoint resume
    min_valid_frame_fraction: float = 0.0 # keep clips only when the valid GT fraction is strictly greater than this value
    mhr_h5_handle_cache_size: int = 32 # maximum packed and render H5 handles retained per DataLoader worker
    mhr_minimal_batch_fields: bool = False # omit inactive targets and diagnostic metadata from production MHR batches
    mhr_rank_local_preprocess: bool = True # defer large render tensors to rank-local workers and reusable pinned buffers
    mhr_rank_local_preprocess_workers: int = 8 # matched eight-GPU benchmark optimum for render preprocessing workers per rank
    val_mhr_rank_local_preprocess_workers: Optional[int] = None # validation render workers per distributed rank; defaults to the training count
    mhr_rank_local_preprocess_slots: int = 2 # fixed reusable render-buffer slots per distributed rank
    mhr_rank_local_preprocess_prefetch_factor: Optional[int] = 2 # reusable slots equal render workers times this factor
    val_mhr_rank_local_preprocess_prefetch_factor: Optional[int] = None # validation override; unset retains fixed slot count
    mhr_rank_local_materialization_mode: str = 'gpu_compact_v1' # compact host transfer followed by GPU training input materialization
    val_mhr_rank_local_materialization_mode: Optional[str] = 'cpu_float32_v1' # validation retains stored float32 XYZ because depth reconstruction is not numerically equivalent
    mhr_input_materialization_upgrade_revision: Optional[str] = None # explicit benchmark-certified upgrade for an existing checkpoint
    mhr_rank_local_preprocess_backend: str = 'process' # thread or spawned process workers writing into fixed shared buffers
    mhr_rank_local_worker_torch_threads: int = 4 # intra-op CPU threads per spawned render worker
    mhr_rank_local_profile_samples_per_worker: int = 2 # detailed CPU-stage timing for only the first samples handled by each persistent worker
    data_name: str = 'video-data'

    # Dataloaders
    batch_size: int = 8
    num_workers: int = 32
    val_batch_size: Optional[int] = None
    val_num_workers: Optional[int] = None

    # Dataset config
    input_resize: tuple = (224, 224)  #(W,H)
    crop_ratio:Optional[float] = None # preprocessing defines the CoCoNet crop; SMPL-H configurations set an explicit ratio when needed
    max_num_key: Optional[int] = None
    use_normal:bool = False
    use_mask:Optional[bool] = False
    normal_uint8:bool = False
    normalize_xyz:Optional[bool] = True
    mhr_xyz_anchor_type: str = 'root_joint_1' # center MHR XYZ conditioning on decoded kinematic root joint 1
    mhr_spatial_normalization_type: str = 'human_height_2m' # scale centered XYZ and both translation residuals so neutral initialized human height is 2
    mhr_spatial_target_height: float = 2.0
    trans_normalizer:Optional[list] = None
    rot_normalizer:Optional[float] = None
    pose_init_type:Optional[str] = 'random-keyframes' # use each frame's stored initialization
    mhr_object_pose_frame: str = 'centered_axis_aligned' # object rotation/translation supervision uses a centered, axis-aligned mesh frame
    mhr_foundationpose_tier_sampling: bool = True # uniformly sample available score-ranked FoundationPose tiers per training frame
    mhr_foundationpose_tier_sampling_revision: str = 'cari4d.mhr_tier_sampling.stateless.v1' # deterministic per-epoch sampling permits exact fast checkpoint resume
    mhr_require_foundationpose_training_tier_schema: bool = False # reject legacy object initialization fields and require tier-only packed/render H5
    val_mhr_require_foundationpose_training_tier_schema: Optional[bool] = None # validation override; keep false for legacy BEHAVE H5
    mhr_input_augmentation_mode: str = 'smplh_rgb_depth_keep50_v2' # public SMPL-H RGB recipe; keep aligned monocular depth for 50% of frames, otherwise apply all four depth corruptions
    add_ho_mask: bool = False
    crop_xyz_3d: bool = False
    subtract_transl: bool = False # subtract xyz map by translation of A
    mask_encode_type: str = 'stack' # stack human + object
    mask_rgb_bkg: bool = False # mask out input rgb image bkg
    occ_drop_thres: float = 2. # when a window contains a frame with occlusion more than this, then drop it
    data_filter_type: str = 'none' # no filter, or drop-occ-<thres>-<probability>, or mixed-<thres> (half heavy occlusion, half simple)
    mask_out_thres_1st: float = 50000 # when the error to the 2nd transformer input is larger than this, zero it out
    input_gtpose_2nd: bool = False # input GT pose to the 2nd transformer
    input_fp_2nd: bool = False  # input directly FP predictions to 2nd transformer
    pose_pred_dir: Optional[str] = None # input pose dir, if given then use this as the predicted pose
    nlf_root: Optional[str] = None
    exclude_frames: Optional[str] = None # file to frames that should not be used for training
    align_poses: bool = False # align the poses to the canonical shape
    trans_ref_type: str = 'frame' # which translation should be used to normalize the input xyz map 

    # motion diffusion
    random_flip: bool = False
    fp_error_thres: float = 500  # mask out object if error is larger than this
    mask_out_contact_in: bool = False # mask out contact input as well 
    fp_vis_thres: float = 2. # threshold for visibility from FP rendering
    body_vis_thres: float = 2.0 # threshold for visibility from NLF rendering

    # Model architecture
    model_name: str = 'HORefine' # decide which model to use
    c_in:int = 6
    n_view:int = 1
    posi_len: int = 256 # positional encoding length, related to input resolution
    time_posi_len: int = 400
    zfar:float = np.inf
    trans_rep:str = 'tracknet'
    rot_rep:Optional[str] = '6d'  # 6d/axis_angle
    rot_rep_hum: str = '6d'
    use_fp_pretrained:bool = True
    use_fp_head: bool = False
    dino_model:Optional[str] = 'dinov2_vitb14' # DINO base
    dino_bnorm: bool = False
    frozen_dino_chunk_size: Optional[int] = None # cap frozen RGB DINO inference images per call; useful for larger validation batches
    encoder_xyzm: str = 'dinov2_vits14' # DINO small
    embed_dim_spatial_temporal: int = 512 # feature dim for performing spatial temporal attention
    mixed_spatial_temporal: bool = True # mix spatial temporal in one transformer layer
    num_attn_layers: int = 2 # spatial attention layers
    feat_dim_6dpose: int = 1024 # object pose feature dimension
    obj_pose_dim_input: int = 6 # object pose feature input dim
    pred_head_attn: bool = False # use attention in prediction head
    hum_cond_dim: int = 75 # human information dimension
    abspose_layers: int = 3 # number of attention layers for abs pose prediction
    hum_cond_embed_dim: int = 128
    abspose_posi_encode: bool = False
    feat_dim_mask: int = 16 # feature dimension of the masks sent to 2nd transformer
    bnorm_mask: bool = True  # batch norm for mask feature
    mask_dino_type: str = 'dinov2_vits14'
    no_hum_mask: bool = False # for 2nd transformer, do not input human mask
    dropout: float = 0.1
    dropout_1st: float = 0.1
    attn_activation: str = 'relu'
    pred_head_dims: tuple = (512, 256, 128)
    pred_head_hum_pose: tuple = (512, 256, 256)
    pred_head_hum_trans: tuple = (512, 256, 128)
    pred_head_contact: tuple = (512, 256, 128)
    cont_out_dim: int = -1 # prediction contacts or not 
    cont_out_type: str = 'binary' # binary or distance 
    cont_mask_thres: float = 0.015 # contact mask threshold. 
    merge_factor: float = 0.5
    merge_strategy: str = 'fixed' # alpha blending strategy
    d_ff_2nd: int = 512 # feedforward model feat dim of 2nd transformer
    nhead_2nd: int = 4
    fp_err_dim: int = -1 # add fp err as feature or not
    visibility_dim: int = -1 # add object visibility value
    pred_uncertainty: bool = False # add uncertainty in output prediction 
    pred_shape: bool = False # add human shape prediction
    pred_mhr_shape: bool = True # predict native MHR shape residuals
    pred_mhr_scale: bool = False # predict native MHR scale residuals
    pred_mhr_face: bool = False # predict native MHR face residuals
    mhr_cond_key: str = 'mhr_coco17_init' # MHR field used as human conditioning
    mhr_cond_dim: int = 51 # flattened COCO17 MHR conditioning, 17 joints * 3
    beta_nll: float = 1.0 # for uncertainty NLL loss
    var_epsilon: float = 0.001 # to avoid negative values, not really helpful
    ## siMLPe additional
    simlpe_hidden_dim: int = 128
    simlpe_num_layers: int = 48 # use default
    ## TCN network
    tcn_hidden_dim: int = 512
    tcn_num_layers: int = 3
    tcn_kernel_size: int = 11
    tcn_dropout: float = 0.1

    # for D-Linear
    individual_chs: bool = False

    # For Decoder like SMPL head
    transformer_decoder_cfg: Dict = field(default_factory=lambda: dict(
        depth=6, # number of layers
        heads=8,
        mlp_dim=1024, # feedforward MLP dim
        dim_head= 64,
        dropout= 0.0,
        emb_dropout= 0.0,
        norm='layer',
        context_dim='${embed_dim_spatial_temporal}'  # the feat dim after spatial temporal transformer, should be the same as embed_dim_spatial_temporal
    ))
    smpl_head_input: str = 'zero'
    num_body_joints: int = 23
    # End of decoder like SMPL head

    # for HOI diffusion
    denoiser_cfg: DenoiserConfig = field(default_factory=DenoiserConfig)
    diff_scheduler_cfg: DiffSchedulerFullConfig = field(default_factory=DiffSchedulerFullConfig)
    contact_dist_thres: float = 10. # only compute loses smaller than this 

    # Run management
    config: Optional[str] = 'learning/configs/cari4d-release.yml'
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = 'e2etracker'
    no_wandb:bool = False
    job: Optional[str] = 'train'
    run_id: Optional[str] = None
    wandb_run_path: Optional[str] = None # explicit entity/project/run_id for legacy inference checkpoints
    exp_name:Optional[str] = None
    save_dir: Optional[str] = 'experiments'
    seed: Optional[int] = None
    vis_every_n_steps: int = 100 # visualize input every n steps
    mhr_input_viz_interval: int = 1000 # MHR W&B input-grid interval in global optimizer steps
    mhr_input_viz_num_frames: int = 10 # evenly spaced frames shown from one temporal window
    train_timing_log_interval: int = 200 # rank timing log interval in global train steps, <=0 disables interval logs
    train_timing_slow_seconds: float = 30.0 # always log phases slower than this many seconds, <=0 disables slow logs
    train_timing_log_dir: Optional[str] = None # optional per-rank timing directory; avoids interleaved distributed stdout
    wandb_log_interval: int = 100 # train scalar W&B log interval in global train steps, <=0 disables train scalar logs
    benchmark_warmup_steps: int = 0 # opt-in speed benchmark warmup; benchmark_steps=0 leaves normal training unchanged
    benchmark_steps: int = 0 # measured speed-benchmark steps; skips validation/checkpointing when complete
    cuda_profile_start_step: int = 0 # first global step instrumented with CUDA events
    cuda_profile_steps: int = 0 # number of CUDA-event-profiled steps; zero disables profiling
    cuda_profile_segment_start_offset: int = 20 # skip compilation/warmup before each segment's bounded CUDA timing window
    cuda_profile_segment_steps: int = 0 # opt-in synchronized CUDA-event samples per segment; zero preserves normal throughput
    ddp_find_unused_parameters: bool = False # measured default: static training graph does not need per-step unused-parameter discovery
    ddp_static_graph: bool = True # measured default for production MHR training
    torch_compile_targets: Optional[Any] = field(default_factory=lambda: ['rgb_dino']) # measured default; null or [] disables compilation
    torch_compile_backend: str = 'inductor'
    torch_compile_mode: str = 'default'
    torch_compile_fullgraph: bool = False
    torch_compile_optimize_ddp: bool = True # disable only when PyTorch DDPOptimizer rejects a compiled higher-order op
    torch_inductor_cache_root: Optional[str] = None # defaults to a code/config/runtime-keyed persistent cache under the experiment directory
    checkpoint_broadcast_chunk_mb: int = 256 # bound temporary device memory while rank 0 distributes one shared checkpoint
    log_errors: bool = False # add errors to the visualization images

    max_step_val: int = 20
    max_steps: int = 200000
    val_epoch_interval: int = 10
    val_step_interval: int = 10000
    ckpt_interval: int = 2000 # save ckpt after this steps
    checkpoint_keep_last: int = 1 # retain this many full resumable checkpoints; zero disables pruning
    val_at_start: bool = False # do one evaluation at the start
    best_validation_semantics_reset_step: Optional[int] = None # archive legacy best artifacts and rebaseline validation before training resumes at or after this step
    use_BN:bool = True
    BN_momentum: float = 0.1
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1

    # Losses
    w_rot: float = 1.0
    w_transl: float = 1.0
    mhr_object_pose_loss_mode: str = "geodesic_radian_vector_mean_equal_weight_v2"
    w_abs_rot: float = 0.1
    w_abs_trans: float = 0.1
    w_hum_rot: float = 0.1
    w_hum_t: float = 0.1
    w_hum_j: float = 0 # joint locations
    w_hum_b: float = 0 # body shape loss
    w_mhr_root_rot: float = 1.0
    w_mhr_trans: float = 1.0
    w_mhr_body_pose: float = 1.0
    w_mhr_hand: float = 0.0
    w_mhr_shape: float = 0.1
    w_mhr_scale: float = 0.0
    w_mhr_face: float = 0.0
    w_mhr_v2v: float = 0.0
    w_mhr_joints: float = 2.0
    mhr_joint_supervision_mode: str = "body12_freeze_hand_face_all_losses"
    w_velo: float = 0.0 # velocity loss for combined motion 
    w_velo_obj: float = 0.0 # velocity loss for object motion  
    w_diff_l2: float = 10.0 
    w_contact: float = 0. # explicit contacts after joints are computed 
    w_heatmap: float = 0.0 # heatmap loss for contact points 
    loss_type:str = 'l2'   # l1/l2/add
    enable_amp: bool = True
    lw_acc: float = 0.0 # acceleration loss
    vis_interval: Optional[int] = 1000
    loss_abs_trans_rela: bool = False # when compute abs trans loss, use relative to first frame or not
    train_stage: str = 'train'
    train_smpl_only: bool = False # do not train other branches, but just SMPL
    symm_loss: bool = True # compute loss with all symmetries

    # scheduler
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig(type='cosine_floor', kwargs=dict(num_training_steps="${max_steps}", end_lr_ratio=0.01)))

    # Inference config
    mesh_file: Optional[str] = None
    test_scene_dir: Optional[str] = None
    track_refine_iter: int = 1
    save_name: Optional[str] = 'test'
    video_only: Optional[bool] = False
    ckpt: Optional[str] = None
    fp_skip: int = 10
    video_out: str = 'output/viz' # path to save visualizations
    save_mhr_geometry: bool = False # save dense MHR vertices/joints/keypoints in forward-viz artifacts
    mhr_save_postopt_observations: bool = False # rebuild full-resolution silhouette observations for SMPL-H-parity MHR post-optimization
    video: str = '' # the path to the video file to be processed 
    debug: Optional[int] = 2
    debug_dir: str = ''
    skip: int = 30 # skip how many frames in sliding window
    redo: bool = False
    test_gt_pose: bool = False
    ckpt_file: Optional[str] = None
    load_training_state: bool = True # load optimizer/scheduler/RNG state when resuming a training checkpoint
    mhr_inference_offline: bool = False # require checkpoint-embedded W&B identity and supervision contract instead of querying W&B
    use_intermediate: bool = False # for abs + delta prediction
    eval_normalize: bool = False # normalize the error
    eval_input: bool = False # evaluate the input pose
    identifier: str = '' # for eval output file
    run_smooth: bool = False
    smooth_smplt: bool = False
    render_video: bool = False
    cam_id: int = 1
    refine_iters: int = 1
    wild_video: bool = False
    masks_root: str = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/masks-h5-my'
    hy3d_meshes_root: str = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/cari4d-demo/behave/meshes'
    fp_root: str = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/fp'
    outpath: str = '/lustre/fs12/portfolios/nvr/projects/nvr_lpr_dataefficientml/xianghuix/behave/foundpose-input/e2etracker/results'

    prev_pred_root: Optional[str] = None # path to previous predicted results, for motion diffusion 
    viz_input: bool = False
    use_sel_view: bool = True # use the selected view for testing
    full_seq: bool = False # use the full sequence for testing, i.e. run sliding window and averaging 
    inf_only: bool = False # only inference, no metrics computation
    opt_name: Optional[str] = None # name of the optimization method

    # evaluation config
    align2gt: bool = True  # align the predicted pose to the GT pose
    result_dir: str = 'none' # path to pth files
    use_hy3d: bool = True # the reconstruction was done using HY3D reconstructed mesh
