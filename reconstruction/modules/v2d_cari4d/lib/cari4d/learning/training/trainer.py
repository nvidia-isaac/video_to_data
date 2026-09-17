# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
trainer
"""
import sys, os
import hashlib
import json
import time

import cv2
import trimesh
from PIL import Image

sys.path.append(os.getcwd())
import wandb
import torch
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
from accelerate import Accelerator
from learning.datasets import get_dataset
from learning.datasets.mhr_rank_local import MHRRankLocalDataLoader
from learning.datasets.mhr_tier_sampling import validate_resume_tier_sampling_contract
from learning.training.training_utils import TrainState, get_scheduler
from learning.training.resume_state import capture_rng_state, find_latest_step_checkpoint, load_rank_rng_state, normalize_epoch_step, rank_rng_state_path, restore_rng_state, resume_dataloader_iterator, set_dataloader_epoch_seed
from learning.training.source_snapshot import log_wandb_source_snapshot, save_resolved_config, wandb_launch_metadata_from_env
from learning.training.training_config import TrainTemporalRefinerConfig
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms.so3 import so3_log_map, so3_exp_map
from torch.optim.lr_scheduler import LambdaLR
import logging
import os.path as osp
import Utils
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs
from learning.models import get_model
from learning.training.mhr_losses import compute_mhr_training_loss, compute_mhr_training_loss_with_geometry, gt_mhr_params_from_batch, mhr_training_metrics_for_logging
from learning.training.mhr_object_pose_loss import MHR_OBJECT_POSE_LOSS_GEODESIC_RADIAN_EQUAL_WEIGHT, MHR_OBJECT_POSE_LOSS_LEGACY, build_mhr_object_pose_loss_contract, reduce_mhr_object_rotation_loss, reduce_mhr_object_translation_loss, restore_mhr_object_pose_loss_contract
from learning.training.mhr_supervision import apply_mhr_supervision_contract, build_mhr_supervision_contract, configure_mhr_training_heads, resolve_mhr_inference_provenance, restore_mhr_training_supervision_contract, training_wandb_config_sha256, training_wandb_identity
from learning.training.best_checkpoint import apply_best_validation_metadata, archive_best_validation_artifacts, best_validation_metadata, load_best_validation_metadata, reset_best_validation_for_semantics, save_best_validation_metadata, update_best_validation
from lib_mhr.object_symmetry import OBJECT_SYMMETRY_MODE_FINITE, OBJECT_SYMMETRY_MODE_FULL_SO3
from learning.training.resume_state import prune_step_checkpoints
from lib_mhr import MHRLayer
from lib_mhr.rotations import rotation_geodesic_distance_radians, rot6d_to_rotmat
from lib_mhr.schema import MHR_PARAM_DIMS
from learning.training.cuda_event_timing import CudaEventTimer, cuda_timing_section, should_profile_cuda_step
from learning.training.rank_timing_log import TimingWindow, open_rank_timing_log, write_rank_timing_record
from learning.training.lr_scheduler import rebase_lambda_scheduler_to_step
from learning.training.runtime_optimization import apply_compile_targets, configure_persistent_inductor_cache, ddp_options
from learning.training.distributed_checkpoint import load_checkpoint_on_rank_zero
from learning.training.checkpoint_recovery import publish_checkpoint_recovery_certificate
from learning.training.speed_benchmark import aggregate_speed_benchmark_rank_results, benchmark_complete, benchmark_total_steps, update_benchmark_sample_identity, write_speed_benchmark_rank_result
from learning.training.segment_runtime import SegmentRuntimeController, clear_pending_validation, pending_validation_step, write_pending_validation
from learning.training.nonfinite_loss import LossGuardContext, guard_finite_losses
from learning.datasets.mhr_sample_metadata import MHR_SAMPLE_METADATA_KEY
from learning.datasets.mhr_augmentation import resolve_mhr_input_augmentation_mode, restore_mhr_input_augmentation_mode
from learning.datasets.mhr_input_materialization import build_mhr_spatial_normalization_contract, build_mhr_xyz_anchor_contract, denormalize_mhr_translation, normalize_mhr_translation, prepare_mhr_spatial_batch, resolve_mhr_input_materialization_mode, restore_mhr_input_materialization_mode, restore_mhr_spatial_normalization_contract, restore_mhr_xyz_anchor_contract
from learning.datasets.mhr_window_sampling import apply_mhr_window_sampling_contract_path, build_mhr_window_sampling_contract, restore_mhr_window_sampling_contract
from learning.training.mhr_input_grid import build_mhr_input_grid, evenly_spaced_frame_indices, render_human_object_mesh_tiles


class Trainer(object):
    @staticmethod
    def object_pose_metric_keys():
        return ("obj_trans_err_m", "obj_rot_geodesic_deg")

    @staticmethod
    def accelerator_mixed_precision(cfg):
        return 'fp16' if bool(getattr(cfg, "enable_amp", False)) else 'no'

    def __init__(self, cfg:TrainTemporalRefinerConfig):
        startup_started_at = time.monotonic()
        self.cfg = cfg
        self.mhr_window_sampling_contract = apply_mhr_window_sampling_contract_path(cfg) if getattr(cfg, "body_model", "smpl") == "mhr" and str(getattr(cfg, "job", "train")) == "train" else None
        self.exp_dir = osp.join(cfg.save_dir, cfg.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # --- 1. Initialize Accelerator ---
        # `Accelerator` will automatically handle device placement, gradient scaling, etc.
        accelerator_started_at = time.monotonic()
        ddp_kwargs = DistributedDataParallelKwargs(**ddp_options(cfg))
        accelerator = Accelerator(mixed_precision=self.accelerator_mixed_precision(cfg), kwargs_handlers=[ddp_kwargs], step_scheduler_with_optimizer=False)
        startup_timing = {"accelerator_init_seconds": time.monotonic() - accelerator_started_at}
        configured_timing_dir = getattr(cfg, "train_timing_log_dir", None)
        rank_timing_dir = osp.join(self.exp_dir, "training-timing") if configured_timing_dir is None else configured_timing_dir
        self.rank_timing_log_handle = open_rank_timing_log(rank_timing_dir, accelerator.process_index)
        self.train_timing_window = TimingWindow()

        # --- 2. Create Model, Optimizer, and Loss Function ---
        model_setup_started_at = time.monotonic()
        inductor_cache = configure_persistent_inductor_cache(cfg, self.exp_dir, accelerator, repo_root=os.getcwd())
        if inductor_cache is not None:
            accelerator.print(f"Persistent Inductor cache: {inductor_cache['cache_dir']} identity={inductor_cache['identity'][:20]}")
        model = get_model(cfg)
        compiled_modules = apply_compile_targets(model, cfg)
        if compiled_modules:
            accelerator.print(f"torch.compile modules: {', '.join(compiled_modules)}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.train_state = TrainState()
        self.resume_rng_state = None
        self.resume_ckpt_file = None
        self.resume_data_sampling_contract = None
        self.data_sampling_contract = None
        self.mhr_input_grid_renderer = None
        self.mhr_supervision_contract = build_mhr_supervision_contract(cfg) if getattr(cfg, "body_model", "smpl") == "mhr" else None
        self.mhr_object_pose_loss_contract = None
        self.mhr_xyz_anchor_contract = None
        self.mhr_spatial_normalization_contract = None
        self.mhr_inference_provenance = None
        self.wandb_run = None
        self.wandb_identity = None
        self.wandb_config_sha256 = None
        loaded_checkpoint_payload = None
        self.segment_runtime = SegmentRuntimeController.from_environment()
        if self.segment_runtime is not None:
            accelerator.print(f"Segment runtime tracking enabled: limit={self.segment_runtime.time_limit_seconds}s margin={self.segment_runtime.checkpoint_margin_seconds}s check_interval={self.segment_runtime.check_interval_steps} steps")
        if cfg.lr_scheduler.type == 'none':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            scheduler = get_scheduler(cfg, optimizer)
        if cfg.ckpt_file is not None:
            ckpt_files = [cfg.ckpt_file]
        else:
            # load ckpt
            latest_ckpt = find_latest_step_checkpoint(self.exp_dir)
            ckpt_files = [] if latest_ckpt is None else [str(latest_ckpt)]
        if len(ckpt_files) == 0:
            assert cfg.job == 'train', 'No ckpt found, and job is not train'
            fp_ckpt_file = "experiments/weights/2023-10-28-18-33-37/model_best.pth"
            
            if cfg.use_fp_pretrained:
                ckpt, load_stats = load_checkpoint_on_rank_zero(fp_ckpt_file, accelerator, max_chunk_bytes=int(getattr(cfg, "checkpoint_broadcast_chunk_mb", 256)) * 1024 * 1024)
                accelerator.print(f"Loaded checkpoint once on rank 0 and distributed {load_stats['tensor_count']} tensors ({load_stats['tensor_bytes'] / 1024 ** 3:.2f} GiB) to {load_stats['world_size']} ranks")
                if 'model' in ckpt:
                    ckpt = ckpt['model']
                # TODO: adapt the pose_embed.pe tensor
                ckpt_new = {}
                for k, v in ckpt.items():
                    if k in ['pos_embed.pe', 'time_pose.pe']:
                        if model.pos_embed.pe.shape != ckpt[k].shape:
                            print(f"Warning: {k} in ckpt shape {ckpt[k].shape} != {model.pos_embed.pe.shape}")
                        else:
                            ckpt_new[k] = v
                    else: ckpt_new[k] = v
                missing_keys, unexpected_keys = model.load_state_dict(ckpt_new, strict=False)
                if len(missing_keys):
                    print(f' - Missing_keys: {missing_keys}')
                if len(unexpected_keys):
                    print(f' - Unexpected_keys: {unexpected_keys}')
                print('loaded model from checkpoint', fp_ckpt_file)
            else:
                if cfg.use_fp_head:
                    states_model = model.state_dict()
                    for k, v in states_model.items():
                        if 'rot_head' in k or 'trans_head' in k:
                            states_model[k] = ckpt[k]
                            print(f'reusing the weight of {k} from FP')
                    model.load_state_dict(states_model)
                else:
                    print("Not loading any ckpt, train from scratch!")
        else:
            # load model and optimizer state
            ckpt, load_stats = load_checkpoint_on_rank_zero(ckpt_files[-1], accelerator, max_chunk_bytes=int(getattr(cfg, "checkpoint_broadcast_chunk_mb", 256)) * 1024 * 1024)
            loaded_checkpoint_payload = ckpt
            accelerator.print(f"Loaded checkpoint once on rank 0 and distributed {load_stats['tensor_count']} tensors ({load_stats['tensor_bytes'] / 1024 ** 3:.2f} GiB) to {load_stats['world_size']} ranks")
            ckpt_new = {}
            for k, v in ckpt['model'].items():
                if k in ['pos_embed.pe', 'time_pose.pe']:
                    if model.pos_embed.pe.shape != ckpt['model'][k].shape:
                        print(f"Warning: {k} in ckpt shape {ckpt['model'][k].shape} != {model.pos_embed.pe.shape}")
                    else:
                        ckpt_new[k] = v
                else:
                    ckpt_new[k] = v
            missing_keys, unexpected_keys = model.load_state_dict(ckpt_new, strict=False)
            self.train_state = TrainState(epoch=ckpt['epoch'], step=ckpt['step'], best_val=ckpt.get('best_val'), epoch_step=ckpt.get('epoch_step'), best_step=ckpt.get('best_step'), pending_validation_step=ckpt.get('pending_validation_step'))
            self.resume_ckpt_file = ckpt_files[-1]
            self.resume_data_sampling_contract = ckpt.get('data_sampling_contract')
            fp_ckpt_file = ckpt_files[-1]
            if getattr(cfg, "load_training_state", True):
                if 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
                else:
                    print("Warning: no optimizer states found in the ckpt!")
                self.resume_rng_state = load_rank_rng_state(ckpt_files[-1], accelerator.process_index, ckpt)
                if 'scheduler' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler'])
                else:
                    print('No scheduler states found in the ckpt!')
                if cfg.lr_scheduler.type == 'cosine_floor':
                    rebase_lambda_scheduler_to_step(scheduler, self.train_state.step)
                    print(f"Rebased cosine scheduler to global optimizer step {self.train_state.step}; lr={optimizer.param_groups[0]['lr']:.8g}")
                if 'amp_scaler' in ckpt:
                    if accelerator.scaler is None and ckpt['amp_scaler'] is not None:
                        raise RuntimeError("Checkpoint contains AMP GradScaler state, but mixed precision is disabled")
                    if accelerator.scaler is not None and ckpt['amp_scaler'] is not None:
                        accelerator.scaler.load_state_dict(ckpt['amp_scaler'])
                        print('Loaded AMP GradScaler state from checkpoint')
                elif accelerator.scaler is not None:
                    print('Warning: no AMP GradScaler state found in the checkpoint!')
            else:
                print('Skipping optimizer, scheduler, and RNG state load from checkpoint.')

            print('loaded model from checkpoint', fp_ckpt_file)
        if getattr(cfg, "body_model", "smpl") == "mhr":
            self.mhr_object_pose_loss_contract = build_mhr_object_pose_loss_contract(cfg) if loaded_checkpoint_payload is None else restore_mhr_object_pose_loss_contract(loaded_checkpoint_payload, cfg)
            source = "fresh configuration" if loaded_checkpoint_payload is None else "checkpoint"
            accelerator.print(f"MHR object-pose loss from {source}: mode={self.mhr_object_pose_loss_contract['mode']} w_rot={cfg.w_rot} w_transl={cfg.w_transl}")
            self.mhr_xyz_anchor_contract = build_mhr_xyz_anchor_contract(cfg) if loaded_checkpoint_payload is None else restore_mhr_xyz_anchor_contract(loaded_checkpoint_payload, cfg)
            accelerator.print(f"MHR XYZ anchor from {source}: {self.mhr_xyz_anchor_contract['anchorType']}")
            self.mhr_spatial_normalization_contract = build_mhr_spatial_normalization_contract(cfg) if loaded_checkpoint_payload is None else restore_mhr_spatial_normalization_contract(loaded_checkpoint_payload, cfg)
            accelerator.print(f"MHR spatial normalization from {source}: {self.mhr_spatial_normalization_contract['normalizationType']} target_height={self.mhr_spatial_normalization_contract['targetHumanHeight']}")
        if getattr(cfg, "body_model", "smpl") == "mhr" and str(getattr(cfg, "job", "train")) == "train":
            self.mhr_window_sampling_contract = build_mhr_window_sampling_contract(cfg) if loaded_checkpoint_payload is None else restore_mhr_window_sampling_contract(loaded_checkpoint_payload, cfg)
            source = "fresh configuration" if loaded_checkpoint_payload is None else "checkpoint"
            accelerator.print(f"MHR training windows from {source}: mode={self.mhr_window_sampling_contract['mode']} strides={self.mhr_window_sampling_contract['temporalStrides']} max_overlap={self.mhr_window_sampling_contract['maximumSampledFrameOverlap']}")
        if getattr(cfg, "body_model", "smpl") == "mhr" and str(getattr(cfg, "job", "train")) == "train":
            augmentation_mode = resolve_mhr_input_augmentation_mode(cfg) if loaded_checkpoint_payload is None else restore_mhr_input_augmentation_mode(loaded_checkpoint_payload, cfg)
            accelerator.print(f"MHR training input augmentation: {augmentation_mode} ({'fresh configuration' if loaded_checkpoint_payload is None else 'checkpoint'})")
            materialization_mode = resolve_mhr_input_materialization_mode(cfg) if loaded_checkpoint_payload is None else restore_mhr_input_materialization_mode(loaded_checkpoint_payload, cfg)
            accelerator.print(f"MHR training input materialization: {materialization_mode} ({'fresh configuration' if loaded_checkpoint_payload is None else 'checkpoint'})")
        if getattr(cfg, "body_model", "smpl") == "mhr" and loaded_checkpoint_payload is not None:
            if str(getattr(cfg, "job", "train")) != "train":
                self.mhr_inference_provenance = resolve_mhr_inference_provenance(loaded_checkpoint_payload, fp_ckpt_file, cfg, wandb_run_path=getattr(cfg, "wandb_run_path", None), offline=bool(getattr(cfg, "mhr_inference_offline", False)))
                self.mhr_supervision_contract = self.mhr_inference_provenance["supervisionContract"]
                accelerator.print(f"MHR inference supervision: source={self.mhr_inference_provenance['configSource']} supervised={self.mhr_supervision_contract['supervisedDeltaKeys']} frozen={self.mhr_supervision_contract['frozenDeltaKeys']}")
            else:
                self.mhr_supervision_contract = restore_mhr_training_supervision_contract(loaded_checkpoint_payload, cfg)
                accelerator.print(f"Restored MHR joint supervision from checkpoint: {cfg.mhr_joint_supervision_mode}")
        if getattr(cfg, "body_model", "smpl") == "mhr" and str(getattr(cfg, "job", "train")) == "train":
            head_contract = configure_mhr_training_heads(model, self.mhr_supervision_contract)
            accelerator.print(f"MHR trainable heads: {head_contract['trainableHeadKeys']}; frozen heads: {head_contract['frozenHeadKeys']}")
        stored_best_metadata = load_best_validation_metadata(self.exp_dir)
        if stored_best_metadata is not None and self.resume_ckpt_file is None:
            raise RuntimeError(f"Found {self.exp_dir}/best_validation.json without a resumable step checkpoint")
        semantics_reset_step = getattr(cfg, "best_validation_semantics_reset_step", None)
        if apply_best_validation_metadata(self.train_state, stored_best_metadata, minimum_step=semantics_reset_step):
            print(f"Restored best validation metric={self.train_state.best_val} step={self.train_state.best_step}")
        should_reset_best = semantics_reset_step is not None and int(self.train_state.step) >= int(semantics_reset_step) and (self.train_state.best_step is None or int(self.train_state.best_step) < int(semantics_reset_step))
        if should_reset_best and accelerator.is_main_process:
            archived = archive_best_validation_artifacts(self.exp_dir, int(semantics_reset_step))
            print(f"Archived legacy best-validation artifacts for corrected semantics: {[str(path) for path in archived]}")
        if reset_best_validation_for_semantics(self.train_state, semantics_reset_step):
            if accelerator.is_main_process:
                write_pending_validation(self.exp_dir, int(self.train_state.step), "best_validation_semantics_reset")
            accelerator.print(f"Reset best validation semantics at global_step={self.train_state.step}; corrected validation will run before the next optimizer step")
        accelerator.wait_for_everyone()
        self.reconcile_pending_validation_marker(accelerator)
        self.ckpt_file = fp_ckpt_file
        print('loss type:', cfg.loss_type)
        print("Total number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        startup_timing["model_optimizer_checkpoint_seconds"] = time.monotonic() - model_setup_started_at

        # --- 3. Create the DataLoader ---
        dataset_setup_started_at = time.monotonic()
        dataloader_train, dataloader_val, dataset_test, dataset_train = get_dataset(cfg)
        if self.mhr_window_sampling_contract is not None and getattr(dataset_train, "window_sampling_contract", None) != self.mhr_window_sampling_contract:
            raise ValueError(f"MHR dataset and trainer window sampling contracts differ: dataset={getattr(dataset_train, 'window_sampling_contract', None)} trainer={self.mhr_window_sampling_contract}")
        self.data_sampling_contract = getattr(dataset_train, "data_sampling_contract", None)
        if self.resume_ckpt_file is not None:
            migrated_sampling = validate_resume_tier_sampling_contract(self.resume_data_sampling_contract, self.data_sampling_contract, self.train_state.step)
            if migrated_sampling:
                accelerator.print(f"Migrating checkpoint step {self.train_state.step} to data sampling contract {self.data_sampling_contract}")
        startup_timing["dataset_setup_seconds"] = time.monotonic() - dataset_setup_started_at

        # --- 4. Prepare distributed components and data placement ---
        distributed_prepare_started_at = time.monotonic()
        rank_local_preprocess = bool(getattr(cfg, "mhr_rank_local_preprocess", False))
        if rank_local_preprocess and getattr(cfg, "body_model", "smpl") != "mhr":
            raise ValueError("mhr_rank_local_preprocess requires body_model=mhr")
        if rank_local_preprocess:
            self.model, self.optimizer, self.scheduler = accelerator.prepare(model, optimizer, scheduler)
            prepared_train = accelerator.prepare_data_loader(dataloader_train, device_placement=False)
            prepared_val = accelerator.prepare_data_loader(dataloader_val, device_placement=False)
            worker_count = int(getattr(cfg, "mhr_rank_local_preprocess_workers", 8))
            val_worker_count_cfg = getattr(cfg, "val_mhr_rank_local_preprocess_workers", None)
            val_worker_count = worker_count if val_worker_count_cfg is None else int(val_worker_count_cfg)
            val_batch_size = int(cfg.batch_size) if getattr(cfg, "val_batch_size", None) is None else int(cfg.val_batch_size)
            dataset_test.cfg.mhr_rank_local_preprocess_prefetch_factor = getattr(cfg, "val_mhr_rank_local_preprocess_prefetch_factor", None)
            self.train_dataloader = MHRRankLocalDataLoader(prepared_train, dataset_train, dataset_train.cfg, max_batch_size=int(cfg.batch_size), worker_count=worker_count, device=accelerator.device, pin_memory=torch.cuda.is_available())
            self.val_dataloader = MHRRankLocalDataLoader(prepared_val, dataset_test, dataset_test.cfg, max_batch_size=val_batch_size, worker_count=val_worker_count, device=accelerator.device, pin_memory=torch.cuda.is_available())
        else:
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler = accelerator.prepare(model, optimizer, dataloader_train, dataloader_val, scheduler)
        self.accelerator = accelerator
        startup_timing["distributed_prepare_seconds"] = time.monotonic() - distributed_prepare_started_at
        self.dataset_test = dataset_test
        self.dataset_train = dataset_train
        resolved_config_path = save_resolved_config(cfg, self.exp_dir) if accelerator.is_main_process else None

        # init logging
        wandb_started_at = time.monotonic()
        if not cfg.no_wandb and accelerator.is_main_process:
            run_id = getattr(cfg, "run_id", None)
            if run_id:
                print(f"resume wandb from explicit run_id={run_id}")
                wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.exp_name, job_type=cfg.job,
                           config=OmegaConf.to_container(cfg),
                           id=run_id,
                           dir=self.exp_dir, resume='allow')
            else:
                print("starting a fresh wandb run")
                wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.exp_name, job_type=cfg.job,
                           config=OmegaConf.to_container(cfg),
                           dir=self.exp_dir, resume='never')
            launch_metadata = wandb_launch_metadata_from_env()
            if launch_metadata:
                wandb_run.config.update(launch_metadata, allow_val_change=True)
            if self.train_state.step == 0:
                log_wandb_source_snapshot(os.getcwd(), self.exp_dir, cfg.exp_name, wandb_run, launch_metadata=launch_metadata, resolved_config_path=resolved_config_path)
            self.wandb_run = wandb_run
            self.wandb_identity = training_wandb_identity(wandb_run)
            self.wandb_config_sha256 = training_wandb_config_sha256(wandb_run)
        startup_timing["wandb_init_seconds"] = time.monotonic() - wandb_started_at


        # Init human model
        human_model_started_at = time.monotonic()
        self.mhr_layer = None
        if cfg.nlf_root is not None and getattr(cfg, "body_model", "smpl") != "mhr":
            from lib_smpl import get_smpl

            self.smpl_male = get_smpl('male', True).cuda()
            self.smpl_female = get_smpl('female', True).cuda()
        elif getattr(cfg, "body_model", "smpl") == "mhr":
            self.mhr_layer = MHRLayer.from_mhr_assets(device=str(accelerator.device))
        benchmark_total = benchmark_total_steps(cfg)
        if benchmark_total > 0 and not bool(cfg.no_wandb):
            raise ValueError("Speed benchmarks require no_wandb=true")
        cuda_profile_steps = int(getattr(cfg, "cuda_profile_steps", 0) or 0)
        cuda_profile_segment_steps = int(getattr(cfg, "cuda_profile_segment_steps", 0) or 0)
        if (cuda_profile_steps > 0 or cuda_profile_segment_steps > 0) and accelerator.device.type != "cuda":
            raise RuntimeError("CUDA event profiling requires a CUDA device")
        self.cuda_event_timer = CudaEventTimer(enabled=cuda_profile_steps > 0 or cuda_profile_segment_steps > 0)
        self.active_cuda_event_timer = None
        startup_timing["human_model_init_seconds"] = time.monotonic() - human_model_started_at
        startup_timing["startup_total_seconds"] = time.monotonic() - startup_started_at
        write_rank_timing_record(self.rank_timing_log_handle, "startup_timing", {"rank": accelerator.process_index, "world": accelerator.num_processes, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "global_step": self.train_state.step, "phases": startup_timing})


    @staticmethod
    def scalar_log_dict(values):
        output = {key: value for key, value in values.items() if not torch.is_tensor(value)}
        tensor_items = [(key, value) for key, value in values.items() if torch.is_tensor(value)]
        for key, value in tensor_items:
            if value.numel() != 1:
                raise ValueError(f"W&B scalar {key} must contain one value, got shape {tuple(value.shape)}")
        if tensor_items:
            host_values = torch.stack([value.detach().reshape(()).double() for _key, value in tensor_items]).cpu().tolist()
            output.update({key: value for (key, _tensor), value in zip(tensor_items, host_values)})
        return output

    def reconcile_pending_validation_marker(self, accelerator):
        marker_step = pending_validation_step(self.exp_dir)
        checkpoint_step = self.train_state.pending_validation_step
        if checkpoint_step is not None:
            checkpoint_step = int(checkpoint_step)
            if checkpoint_step != int(self.train_state.step):
                raise ValueError(f"Pending validation step {checkpoint_step} does not match checkpoint global step {self.train_state.step}")
            if marker_step is not None and marker_step != checkpoint_step:
                raise ValueError(f"Pending validation marker step {marker_step} does not match checkpoint value {checkpoint_step}")
            if marker_step is None and accelerator.is_main_process:
                write_pending_validation(self.exp_dir, checkpoint_step, "checkpoint_recovery")
                accelerator.print(f"Recreated pending validation marker at step {checkpoint_step}")
        elif marker_step is not None:
            if self.resume_ckpt_file is None:
                raise ValueError(f"Found pending validation marker at step {marker_step} without a resumable checkpoint")
            if accelerator.is_main_process:
                clear_pending_validation(self.exp_dir)
                accelerator.print(f"Removed stale pending validation marker at step {marker_step}; checkpoint step {self.train_state.step} has no pending validation")
        accelerator.wait_for_everyone()

    def segment_runtime_checkpoint_reason(self, global_step, force=False):
        controller = getattr(self, "segment_runtime", None)
        if controller is None or not controller.should_check(global_step, force=force):
            return None
        local_reason = controller.local_reason(global_step, force=True)
        votes = torch.tensor([int(local_reason == "usr1"), int(local_reason == "deadline")], dtype=torch.int32, device=self.accelerator.device)
        votes = self.accelerator.reduce(votes, reduction="sum")
        usr1_votes, deadline_votes = votes.detach().cpu().tolist()
        if usr1_votes > 0:
            return "usr1"
        if deadline_votes > 0:
            return "deadline"
        return None

    def save_runtime_checkpoint(self, cfg, model, optimizer, scheduler, train_state, reason, pending_validation=None):
        if self.segment_runtime is None:
            raise RuntimeError("Runtime checkpoint requested without segment runtime metadata")
        train_state.pending_validation_step = pending_validation
        metadata = self.segment_runtime.metadata(train_state.step, reason)
        checkpoint_start = time.monotonic()
        ckpt_file = self.save_checkpoint(self.accelerator, cfg, model, optimizer, scheduler, train_state, checkpoint_reason=f"segment_{reason}", segment_runtime=metadata)
        metadata["checkpoint_write_seconds"] = time.monotonic() - checkpoint_start
        metadata["checkpoint_path"] = ckpt_file
        metadata["pending_validation_step"] = pending_validation
        if self.accelerator.is_main_process:
            marker = self.segment_runtime.write_checkpoint_completion(metadata)
            print(f"SEGMENT_RUNTIME_CHECKPOINT reason={reason} step={train_state.step} elapsed_seconds={metadata['elapsed_seconds']:.1f} remaining_seconds={metadata['remaining_seconds']:.1f} write_seconds={metadata['checkpoint_write_seconds']:.1f} checkpoint={ckpt_file} marker={marker}", flush=True)
        self.accelerator.wait_for_everyone()
        return ckpt_file

    def train(self):
        cfg = self.cfg
        accelerator = self.accelerator
        model, optimizer, train_dataloader, val_dataloader = self.model, self.optimizer, self.train_dataloader, self.val_dataloader
        scheduler = self.scheduler
        # --- 5. The Training Loop ---
        train_state = self.train_state
        benchmark_start_step = int(train_state.step)
        benchmark_total = benchmark_total_steps(cfg)
        benchmark_warmup_steps = int(getattr(cfg, "benchmark_warmup_steps", 0) or 0)
        benchmark_measured_steps = int(getattr(cfg, "benchmark_steps", 0) or 0)
        benchmark_timing_window = TimingWindow() if benchmark_total > 0 else None
        benchmark_sample_identity = hashlib.sha256() if benchmark_total > 0 else None
        benchmark_measured_started_at = None
        self.train_timing_window = TimingWindow()
        self.train_timing_steps = 0
        write_rank_timing_record(self.rank_timing_log_handle, "training_segment_start", {"rank": accelerator.process_index, "world": accelerator.num_processes, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "global_step": train_state.step})
        dataloader_len = len(train_dataloader)
        if dataloader_len == 0:
            raise ValueError("Training dataloader is empty; max_steps training cannot make progress.")
        train_state.epoch, train_state.epoch_step = normalize_epoch_step(train_state.epoch, train_state.step, train_state.epoch_step, dataloader_len)
        deferred_validation_completed = False
        if train_state.pending_validation_step is not None:
            if self.resume_rng_state is not None:
                restore_rng_state(self.resume_rng_state)
                self.resume_rng_state = None
                accelerator.print(f"Restored RNG state from {self.resume_ckpt_file} before deferred validation")
            accelerator.print(f"Running deferred validation at global_step={train_state.pending_validation_step} before further training")
            runtime_reason = self.eval_model(cfg, model, train_state, val_dataloader)
            if runtime_reason is not None:
                self.save_runtime_checkpoint(cfg, model, optimizer, scheduler, train_state, runtime_reason, pending_validation=train_state.step)
                return
            train_state.pending_validation_step = None
            self.resume_ckpt_file = self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state, checkpoint_reason="deferred_validation_complete")
            self.resume_rng_state = capture_rng_state()
            deferred_validation_completed = True
            accelerator.print(f"Deferred validation completed and checkpoint updated at global_step={train_state.step}")
        if train_state.step >= cfg.max_steps:
            accelerator.print(f"Loaded step={train_state.step} >= max_steps={cfg.max_steps}, skipping training.")
            return
        if cfg.val_at_start and train_state.step == 0 and not deferred_validation_completed:
            print('Evaluation at the start of training.')
            runtime_reason = self.eval_model(cfg, model, train_state, val_dataloader)
            if runtime_reason is not None:
                self.save_runtime_checkpoint(cfg, model, optimizer, scheduler, train_state, runtime_reason, pending_validation=train_state.step)
                return
        accelerator.print(f"Starting training from global_step={train_state.step}, epoch={train_state.epoch}, epoch_step={train_state.epoch_step}")
        while train_state.step < cfg.max_steps:
            epoch = train_state.epoch
            model.train()
            total_loss = torch.zeros((), dtype=torch.float64, device=accelerator.device)
            steps_this_epoch = 0
            set_dataloader_epoch_seed(train_dataloader, getattr(cfg, "seed", None), epoch)
            start_epoch_step = int(train_state.epoch_step or 0) if epoch == train_state.epoch else 0
            dataloader_iter, resume_skip_mode = resume_dataloader_iterator(accelerator, train_dataloader, start_epoch_step)
            if start_epoch_step > 0:
                accelerator.print(f"Positioned training dataloader at epoch_step={start_epoch_step} mode={resume_skip_mode}")
            if self.resume_rng_state is not None:
                restore_rng_state(self.resume_rng_state)
                self.resume_rng_state = None
                accelerator.print(f"Restored RNG state from {self.resume_ckpt_file}")

            step = start_epoch_step
            while True:
                step_wall_started_at = time.monotonic()
                benchmark_relative_step = train_state.step - benchmark_start_step
                benchmark_measured_this_step = benchmark_total > 0 and benchmark_warmup_steps <= benchmark_relative_step < benchmark_total
                if benchmark_total > 0 and benchmark_relative_step == benchmark_warmup_steps:
                    if hasattr(train_dataloader, "consume_timing_summary"):
                        train_dataloader.consume_timing_summary()
                    benchmark_measured_started_at = step_wall_started_at
                batch_fetch_start = time.monotonic()
                self.log_rank_timing('batch_fetch_start', cfg, train_state, epoch, step)
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                global_cuda_profile = should_profile_cuda_step(getattr(cfg, "cuda_profile_start_step", 0), getattr(cfg, "cuda_profile_steps", 0), train_state.step)
                segment_cuda_profile = should_profile_cuda_step(getattr(cfg, "cuda_profile_segment_start_offset", 20), getattr(cfg, "cuda_profile_segment_steps", 0), train_state.step - benchmark_start_step)
                profile_this_step = global_cuda_profile or segment_cuda_profile
                self.active_cuda_event_timer = self.cuda_event_timer if profile_this_step else None
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.cuda_event_timer = self.active_cuda_event_timer
                if profile_this_step:
                    self.cuda_event_timer.begin_step(train_state.step)
                batch_fetch_seconds = time.monotonic() - batch_fetch_start
                if benchmark_measured_this_step:
                    metadata_rows = batch.get(MHR_SAMPLE_METADATA_KEY)
                    if not isinstance(metadata_rows, list) or len(metadata_rows) != int(cfg.batch_size):
                        raise ValueError(f"Speed benchmark requires {int(cfg.batch_size)} ordered MHR sample metadata rows, got {type(metadata_rows).__name__} length={None if metadata_rows is None else len(metadata_rows)}")
                    update_benchmark_sample_identity(benchmark_sample_identity, metadata_rows)
                self.log_rank_timing('batch_fetch_done', cfg, train_state, epoch, step, batch_fetch_seconds)
                # No need for .to(device), accelerate handles it!
                # Forward pass
                model_forward_start = time.monotonic()
                self.log_rank_timing('model_forward_start', cfg, train_state, epoch, step)
                log_train_wandb = self.should_log_train_wandb(cfg, train_state.step)
                log_train_object_metrics = log_train_wandb and not cfg.no_wandb and accelerator.is_main_process
                train_visualization_frame_indices = None
                train_prediction_geometry = None
                if getattr(cfg, "body_model", "smpl") == "mhr" and accelerator.is_main_process and self.should_run_train_visualization(cfg, train_state.step + 1):
                    train_visualization_frame_indices = evenly_spaced_frame_indices(batch['render_rgbs'].shape[1], int(getattr(cfg, "mhr_input_viz_num_frames", 10)))
                    train_prediction_geometry = {}
                with cuda_timing_section(self.active_cuda_event_timer, "model_forward_and_loss"):
                    loss, loss_r, loss_t, loss_acc, loss_dict = self.forward_step(batch, cfg, model, vis=False, return_loss_dict=True, log_object_pose_metrics=log_train_object_metrics, log_wandb=False, visualization_frame_indices=train_visualization_frame_indices, visualization_geometry=train_prediction_geometry)
                model_forward_seconds = time.monotonic() - model_forward_start
                self.log_rank_timing('model_forward_done', cfg, train_state, epoch, step, model_forward_seconds)
                metrics_logging_started_at = time.monotonic()
                guard_finite_losses(loss, loss_dict, batch, LossGuardContext(exp_dir=self.exp_dir, global_step=train_state.step, epoch=epoch, epoch_batch_index=step, rank=accelerator.process_index, world_size=accelerator.num_processes, experiment_name=cfg.exp_name, seed=getattr(cfg, "seed", None), resume_checkpoint=self.resume_ckpt_file))
                if log_train_wandb and not cfg.no_wandb and accelerator.is_main_process:
                    log_dict = {'train/loss_train': loss.detach(), 'train/loss_train_r': loss_r.detach(),
                                'train/loss_train_t': loss_t.detach(), 'train/loss_train_acc': loss_acc.detach(),
                                'train/lr': optimizer.param_groups[0]['lr']}
                    train_metrics = mhr_training_metrics_for_logging(loss_dict, cfg)
                    log_dict.update({f'train/{name}': value.detach() if torch.is_tensor(value) else value for name, value in train_metrics.items()})
                    log_dict = self.scalar_log_dict(log_dict)
                    wandb.log(log_dict, step=train_state.step)
                metrics_logging_seconds = time.monotonic() - metrics_logging_started_at

                # Backward pass - accelerator handles the backward pass
                backward_start = time.monotonic()
                self.log_rank_timing('backward_start', cfg, train_state, epoch, step)
                with cuda_timing_section(self.active_cuda_event_timer, "backward"):
                    accelerator.backward(loss)
                backward_seconds = time.monotonic() - backward_start
                self.log_rank_timing('backward_done', cfg, train_state, epoch, step, backward_seconds)
                optimizer_start = time.monotonic()
                self.log_rank_timing('optimizer_start', cfg, train_state, epoch, step)
                with cuda_timing_section(self.active_cuda_event_timer, "optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                if getattr(accelerator, "optimizer_step_was_skipped", False):
                    accelerator.print(f"Optimizer step skipped by AMP GradScaler at global_step={train_state.step}")
                optimizer_seconds = time.monotonic() - optimizer_start
                self.log_rank_timing('optimizer_done', cfg, train_state, epoch, step, optimizer_seconds)
                if profile_this_step:
                    cuda_result = self.cuda_event_timer.finish_step()
                    cuda_result.update({"rank": accelerator.process_index, "world": accelerator.num_processes, "segment_step": train_state.step - benchmark_start_step})
                    write_rank_timing_record(self.rank_timing_log_handle, "cuda_profile", cuda_result)
                    unwrapped_model.cuda_event_timer = None
                    self.active_cuda_event_timer = None

                step_wall_seconds = time.monotonic() - step_wall_started_at
                step_timings = {"batch_fetch": batch_fetch_seconds, "model_forward_and_loss": model_forward_seconds, "metrics_and_logging": metrics_logging_seconds, "backward": backward_seconds, "optimizer": optimizer_seconds, "step_wall": step_wall_seconds}
                for phase, seconds in step_timings.items():
                    self.train_timing_window.add(phase, seconds)
                    if benchmark_measured_this_step:
                        benchmark_timing_window.add(phase, seconds)
                self.train_timing_steps += 1
                self.maybe_log_train_timing_summary(cfg, train_state.step + 1, epoch, step, train_dataloader)

                total_loss += loss.detach().to(dtype=torch.float64)
                steps_this_epoch += 1
                train_state.step += 1
                train_state.epoch = epoch
                train_state.epoch_step = step + 1

                if benchmark_complete(cfg, benchmark_start_step, train_state.step):
                    if benchmark_measured_started_at is None or benchmark_sample_identity is None or benchmark_timing_window is None:
                        raise RuntimeError("Speed benchmark completed without initialized measurement state")
                    rank_payload = {"worker_count": int(getattr(cfg, "mhr_rank_local_preprocess_workers", 0)), "world_size": accelerator.num_processes, "batch_size_per_rank": int(cfg.batch_size), "clip_length": int(cfg.clip_len), "warmup_steps": benchmark_warmup_steps, "measured_steps": benchmark_measured_steps, "measured_elapsed_seconds": time.monotonic() - benchmark_measured_started_at, "sample_identity_sha256": benchmark_sample_identity.hexdigest(), "phases": benchmark_timing_window.summary(reset=True), "preprocess": train_dataloader.consume_timing_summary() if hasattr(train_dataloader, "consume_timing_summary") else {}}
                    rank_result_path = write_speed_benchmark_rank_result(self.exp_dir, accelerator.process_index, rank_payload)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        benchmark_result = aggregate_speed_benchmark_rank_results(self.exp_dir, accelerator.num_processes)
                        print("BENCHMARK_RESULT " + json.dumps(benchmark_result, sort_keys=True), flush=True)
                    accelerator.wait_for_everyone()
                    accelerator.print(f"BENCHMARK_COMPLETE warmup_steps={benchmark_warmup_steps} measured_steps={benchmark_measured_steps} start_step={benchmark_start_step} end_step={train_state.step} rank_result={rank_result_path}")
                    return

                validation_due = train_state.step >= cfg.max_steps or train_state.step % cfg.val_step_interval == 0
                runtime_reason = self.segment_runtime_checkpoint_reason(train_state.step, force=validation_due)
                if runtime_reason is not None:
                    pending_validation = train_state.step if validation_due else None
                    self.save_runtime_checkpoint(cfg, model, optimizer, scheduler, train_state, runtime_reason, pending_validation=pending_validation)
                    return

                if train_state.step >= cfg.max_steps:
                    accelerator.print(f"Reached max_steps={cfg.max_steps}, stopping training.")
                    runtime_reason = self.eval_model(cfg, model, train_state, val_dataloader)
                    if runtime_reason is not None:
                        self.save_runtime_checkpoint(cfg, model, optimizer, scheduler, train_state, runtime_reason, pending_validation=train_state.step)
                        return
                    self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state, checkpoint_reason="final")
                    self.run_train_visualization(batch, cfg, model, train_state, prediction_geometry=train_prediction_geometry)
                    return

                # Print progress from the main process only
                if accelerator.is_main_process and step % 20 == 0:
                    accelerator.print(f"Epoch [{epoch + 1}], Step [{step}], Global Step [{train_state.step}/{cfg.max_steps}], LR: {optimizer.param_groups[0]['lr']:.8g}")

                if train_state.step % cfg.val_step_interval == 0:
                    runtime_reason = self.eval_model(cfg, model, train_state, val_dataloader)
                    if runtime_reason is not None:
                        self.save_runtime_checkpoint(cfg, model, optimizer, scheduler, train_state, runtime_reason, pending_validation=train_state.step)
                        return
                if train_state.step % cfg.ckpt_interval == 0:
                    self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state, checkpoint_reason="periodic")

                lr = optimizer.param_groups[0]['lr']
                if self.should_stop_for_small_lr(cfg, accelerator, train_state, lr):
                    print("Learning rate too small, stopping training.")
                    runtime_reason = self.segment_runtime_checkpoint_reason(train_state.step, force=True)
                    if runtime_reason is not None:
                        self.save_runtime_checkpoint(cfg, model, optimizer, scheduler, train_state, runtime_reason, pending_validation=train_state.step)
                        return
                    runtime_reason = self.eval_model(cfg, model, train_state, val_dataloader)
                    if runtime_reason is not None:
                        self.save_runtime_checkpoint(cfg, model, optimizer, scheduler, train_state, runtime_reason, pending_validation=train_state.step)
                        return
                    self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state, checkpoint_reason="small_lr")
                    self.run_train_visualization(batch, cfg, model, train_state, prediction_geometry=train_prediction_geometry)
                    return
                self.run_train_visualization(batch, cfg, model, train_state, prediction_geometry=train_prediction_geometry)
                step += 1

            # Log average loss for the epoch from the main process
            if accelerator.is_main_process and steps_this_epoch > 0:
                avg_loss = float((total_loss / steps_this_epoch).detach().cpu())
                accelerator.print(f"--- End of Epoch [{epoch + 1}], Average Loss: {avg_loss:.4f} ---")
            train_state.epoch = epoch + 1
            train_state.epoch_step = 0
        accelerator.print("Training complete!")

    def close(self):
        for dataloader in (self.train_dataloader, self.val_dataloader):
            close = getattr(dataloader, "close", None)
            if callable(close):
                close()

    @staticmethod
    def should_log_train_timing(cfg, global_step, seconds=None):
        interval = int(getattr(cfg, "train_timing_log_interval", 0) or 0)
        slow_seconds = float(getattr(cfg, "train_timing_slow_seconds", 0.0) or 0.0)
        if seconds is not None and slow_seconds > 0 and seconds >= slow_seconds:
            return True
        return interval > 0 and global_step >= 0 and global_step % interval == 0

    def maybe_log_train_timing_summary(self, cfg, completed_global_step, epoch, epoch_step, train_dataloader):
        interval = int(getattr(cfg, "train_timing_log_interval", 0) or 0)
        if interval <= 0 or self.train_timing_steps < interval:
            return
        phases = self.train_timing_window.summary(reset=True)
        preprocess = train_dataloader.consume_timing_summary() if hasattr(train_dataloader, "consume_timing_summary") else {}
        payload = {"rank": self.accelerator.process_index, "world": self.accelerator.num_processes, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "global_step": int(completed_global_step), "epoch": int(epoch), "epoch_step": int(epoch_step), "window_steps": int(self.train_timing_steps), "phases": phases, "preprocess": preprocess}
        write_rank_timing_record(self.rank_timing_log_handle, "train_timing_summary", payload)
        if self.accelerator.is_main_process and not bool(cfg.no_wandb):
            wandb_values = {}
            for scope, summary in (("step", phases), ("preprocess", preprocess)):
                for phase, stats in summary.items():
                    for statistic in ("mean_seconds", "p95_seconds", "max_seconds"):
                        wandb_values[f"timing/{scope}/{phase}_{statistic.removesuffix('_seconds')}_s"] = stats[statistic]
            step_wall = phases.get("step_wall", {}).get("mean_seconds")
            if step_wall:
                global_batch = int(cfg.batch_size) * int(self.accelerator.num_processes)
                wandb_values["timing/throughput_clips_per_second"] = global_batch / step_wall
                wandb_values["timing/throughput_frames_per_second"] = global_batch * int(cfg.clip_len) / step_wall
            wandb.log(wandb_values, step=max(0, int(completed_global_step) - 1))
        self.train_timing_steps = 0

    @staticmethod
    def should_log_train_wandb(cfg, global_step):
        interval = int(getattr(cfg, "wandb_log_interval", 0) or 0)
        return interval > 0 and global_step >= 0 and global_step % interval == 0

    @staticmethod
    def scheduler_warmup_steps(cfg):
        scheduler_cfg = getattr(cfg, "lr_scheduler", None)
        scheduler_kwargs = getattr(scheduler_cfg, "kwargs", {}) if scheduler_cfg is not None else {}
        if hasattr(scheduler_kwargs, "get"):
            return int(scheduler_kwargs.get("num_warmup_steps", 0) or 0)
        return int(getattr(scheduler_kwargs, "num_warmup_steps", 0) or 0)

    def should_stop_for_small_lr(self, cfg, accelerator, train_state, lr):
        if getattr(accelerator, "optimizer_step_was_skipped", False):
            return False
        warmup_steps = self.scheduler_warmup_steps(cfg)
        if warmup_steps > 0 and train_state.step <= warmup_steps:
            return False
        return lr < 1e-7

    def log_rank_timing(self, phase, cfg, train_state, epoch, epoch_step, seconds=None):
        if not self.should_log_train_timing(cfg, train_state.step, seconds):
            return
        rank = getattr(self.accelerator, "process_index", 0)
        world = getattr(self.accelerator, "num_processes", 1)
        seconds_text = "" if seconds is None else f" seconds={seconds:.6f}"
        line = f"[rank_timing] wall_time={time.strftime('%Y-%m-%dT%H:%M:%S')} rank={rank} world={world} phase={phase} global_step={train_state.step} epoch={epoch} epoch_step={epoch_step}{seconds_text}"
        if self.rank_timing_log_handle is None:
            print(line, flush=True)
        else:
            self.rank_timing_log_handle.write(line + "\n")

    @staticmethod
    def should_run_train_visualization(cfg, global_step):
        is_mhr = getattr(cfg, "body_model", "smpl") == "mhr"
        if is_mhr and bool(getattr(cfg, "no_wandb", False)):
            return False
        interval = int(getattr(cfg, "mhr_input_viz_interval", 0) or 0) if is_mhr else int(getattr(cfg, "vis_every_n_steps", 0) or 0)
        if interval <= 0 or global_step <= 0:
            return False
        return global_step % interval == 0

    def run_train_visualization(self, batch, cfg, model, train_state, prediction_geometry=None):
        if not self.should_run_train_visualization(cfg, train_state.step):
            return
        accelerator = self.accelerator
        accelerator.wait_for_everyone()
        if getattr(cfg, "body_model", "smpl") == "mhr":
            if accelerator.is_main_process:
                with torch.no_grad():
                    self.log_mhr_input_viz(batch, cfg, model, key='train', prediction_geometry=prediction_geometry)
        else:
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    self.forward_batch(batch, cfg, model, vis=accelerator.is_main_process, ret_dict=True, vis_key='train')
            finally:
                if was_training:
                    model.train()
                else:
                    model.eval()
        accelerator.wait_for_everyone()

    def save_checkpoint(self, accelerator, cfg, model, optimizer, scheduler, train_state, checkpoint_reason="periodic", segment_runtime=None):
        ckpt_file = osp.join(self.exp_dir, f'step{train_state.step:06d}.pth')
        rng_state = capture_rng_state()
        rng_file = rank_rng_state_path(ckpt_file, accelerator.process_index)
        rng_tmp_file = rng_file.with_name(f".{rng_file.name}.tmp-{os.getpid()}-{time.time_ns()}")
        torch.save(rng_state, rng_tmp_file)
        os.replace(rng_tmp_file, rng_file)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if train_state.pending_validation_step is not None:
                write_pending_validation(self.exp_dir, int(train_state.pending_validation_step), checkpoint_reason)
            print(f"Training state: epoch={train_state.epoch}, step={train_state.step}, epoch_step={train_state.epoch_step}")
            best_metadata = best_validation_metadata(cfg, train_state) if train_state.best_val is not None and train_state.best_step is not None else None
            checkpoint_dict = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': train_state.epoch,
                'step': train_state.step,
                'epoch_step': train_state.epoch_step,
                'best_val': train_state.best_val,
                'best_step': train_state.best_step,
                'best_validation': best_metadata,
                'amp_scaler': accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                'rng_state': rng_state,
                'pending_validation_step': train_state.pending_validation_step,
                'checkpoint_reason': checkpoint_reason,
                'segment_runtime': segment_runtime,
                'data_sampling_contract': getattr(self, 'data_sampling_contract', None),
                'mhr_window_sampling_contract': getattr(self, 'mhr_window_sampling_contract', None),
                'mhr_supervision_contract': getattr(self, 'mhr_supervision_contract', None),
                'mhr_object_pose_loss_contract': getattr(self, 'mhr_object_pose_loss_contract', None),
                'mhr_xyz_anchor_contract': getattr(self, 'mhr_xyz_anchor_contract', None),
                'mhr_spatial_normalization_contract': getattr(self, 'mhr_spatial_normalization_contract', None),
                'wandb_identity': getattr(self, 'wandb_identity', None),
                'wandb_config_sha256': getattr(self, 'wandb_config_sha256', None),
                'cfg': cfg
            }
            ckpt_tmp_file = osp.join(self.exp_dir, f'.{osp.basename(ckpt_file)}.tmp-{os.getpid()}-{time.time_ns()}')
            accelerator.save(checkpoint_dict, ckpt_tmp_file)
            os.replace(ckpt_tmp_file, ckpt_file)
            print(f"ckpt saved to {ckpt_file} reason={checkpoint_reason}")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            certificate = publish_checkpoint_recovery_certificate(ckpt_file, checkpoint_dict, accelerator.num_processes)
            print(f"checkpoint recovery certificate saved to {certificate['certificatePath']}")
            removed = prune_step_checkpoints(self.exp_dir, getattr(cfg, "checkpoint_keep_last", 0))
            if removed:
                print(f"Pruned {len(removed)} old checkpoint files: {[str(path) for path in removed]}")
            if train_state.pending_validation_step is None:
                clear_pending_validation(self.exp_dir)
        accelerator.wait_for_everyone()
        return ckpt_file

    def save_best_checkpoint(self, accelerator, cfg, model, train_state):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metadata = best_validation_metadata(cfg, train_state)
            ckpt_file = osp.join(self.exp_dir, 'model_best.pth')
            tmp_file = osp.join(self.exp_dir, f'.model_best.pth.tmp-{os.getpid()}')
            checkpoint_dict = {'model': accelerator.unwrap_model(model).state_dict(), 'cfg': cfg, 'best_val': train_state.best_val, 'best_step': train_state.best_step, 'mhr_window_sampling_contract': getattr(self, 'mhr_window_sampling_contract', None), 'mhr_supervision_contract': getattr(self, 'mhr_supervision_contract', None), 'mhr_object_pose_loss_contract': getattr(self, 'mhr_object_pose_loss_contract', None), 'mhr_xyz_anchor_contract': getattr(self, 'mhr_xyz_anchor_contract', None), 'mhr_spatial_normalization_contract': getattr(self, 'mhr_spatial_normalization_contract', None), 'wandb_identity': getattr(self, 'wandb_identity', None), 'wandb_config_sha256': getattr(self, 'wandb_config_sha256', None), **metadata}
            accelerator.save(checkpoint_dict, tmp_file)
            os.replace(tmp_file, ckpt_file)
            metadata_file = save_best_validation_metadata(self.exp_dir, metadata)
            print(f"best checkpoint saved to {ckpt_file} metric={train_state.best_val} step={train_state.best_step} metadata={metadata_file}")
        accelerator.wait_for_everyone()

    def eval_model(self, cfg, model, train_state, val_dataloader):
        model.eval()
        val_seed = None if getattr(cfg, "seed", None) is None else int(cfg.seed) + 1000003
        set_dataloader_epoch_seed(val_dataloader, val_seed, train_state.step)
        validation_totals = torch.zeros(5, dtype=torch.float64, device=self.accelerator.device)
        validation_metric_sums = {}
        validation_metric_names = None
        validation_input_grid = None
        for step_val, batch in enumerate(tqdm(val_dataloader)):
            runtime_reason = self.segment_runtime_checkpoint_reason(train_state.step, force=True)
            if runtime_reason is not None:
                self.accelerator.print(f"Deferring incomplete validation at global_step={train_state.step} reason={runtime_reason} completed_batches={step_val}")
                model.train()
                return runtime_reason
            validation_visualization_frame_indices = None
            validation_prediction_geometry = None
            if step_val == 0 and getattr(cfg, "body_model", "smpl") == "mhr" and not cfg.no_wandb and self.accelerator.is_main_process:
                validation_visualization_frame_indices = evenly_spaced_frame_indices(batch['render_rgbs'].shape[1], int(getattr(cfg, "mhr_input_viz_num_frames", 10)))
                validation_prediction_geometry = {}
            with torch.no_grad():
                if getattr(cfg, "body_model", "smpl") == "mhr":
                    loss, loss_r, loss_t, loss_acc, loss_dict = self.forward_step(batch, cfg, model, vis=step_val==0, return_loss_dict=True, log_wandb=False, visualization_frame_indices=validation_visualization_frame_indices, visualization_geometry=validation_prediction_geometry)
                else:
                    loss, loss_r, loss_t, loss_acc = self.forward_step(batch, cfg, model, vis=step_val==0)
            if step_val == 0 and getattr(cfg, "body_model", "smpl") == "mhr" and not cfg.no_wandb:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    validation_input_grid = self.log_mhr_input_viz(batch, cfg, model, key='val', log_wandb=False, prediction_geometry=validation_prediction_geometry)
                self.accelerator.wait_for_everyone()
            loss_weight = self.validation_loss_weight(batch, loss.device)
            validation_totals[0] += loss.detach().double() * loss_weight
            validation_totals[1] += loss_r.detach().double() * loss_weight
            validation_totals[2] += loss_t.detach().double() * loss_weight
            validation_totals[3] += loss_acc.detach().double() * loss_weight
            validation_totals[4] += loss_weight
            if getattr(cfg, "body_model", "smpl") == "mhr":
                metric_names = tuple(sorted(loss_dict))
                if validation_metric_names is None:
                    validation_metric_names = metric_names
                    validation_metric_sums = {name: torch.zeros((), dtype=torch.float64, device=loss.device) for name in metric_names}
                elif metric_names != validation_metric_names:
                    raise ValueError(f"MHR validation metric keys changed between batches: {validation_metric_names} != {metric_names}")
                for name in validation_metric_names:
                    value = loss_dict[name]
                    if not torch.is_tensor(value):
                        value = torch.as_tensor(value, device=loss.device)
                    if value.numel() != 1:
                        raise ValueError(f"MHR validation metric {name} must be scalar, got shape {tuple(value.shape)}")
                    validation_metric_sums[name] += value.detach().double().reshape(()) * loss_weight
        if validation_metric_names:
            validation_totals = torch.cat([validation_totals, torch.stack([validation_metric_sums[name] for name in validation_metric_names])])
        validation_totals = self.accelerator.reduce(validation_totals, reduction="sum")
        validation_values = validation_totals.detach().cpu()
        loss_weight = validation_values[4]
        if float(loss_weight) <= 0:
            raise ValueError("validation requires at least one valid frame across all ranks")
        loss_val = float(validation_values[0] / loss_weight)
        loss_val_r = float(validation_values[1] / loss_weight)
        loss_val_t = float(validation_values[2] / loss_weight)
        loss_val_acc = float(validation_values[3] / loss_weight)
        if update_best_validation(train_state, loss_val, train_state.step):
            self.save_best_checkpoint(self.accelerator, cfg, model, train_state)
        if not cfg.no_wandb and self.accelerator.is_main_process:
            log_dict = {'val/loss_val': loss_val, 'val/loss_val_r': loss_val_r,
                       'val/loss_val_t': loss_val_t, 'val/loss_val_acc': loss_val_acc}
            if validation_metric_names:
                for index, name in enumerate(validation_metric_names, start=5):
                    if name in ("loss_mhr_v2v", "loss_mhr_joints"):
                        continue
                    log_dict[f"val/{name}"] = float(validation_values[index] / loss_weight)
            if validation_input_grid is not None:
                log_dict['val/input_grid'] = validation_input_grid
            wandb.log(log_dict, step=train_state.step)
        print(f'--- Eval at step {train_state.step}, loss: {loss_val:.4f} lr: {self.optimizer.param_groups[0]["lr"]:.5f} ---')
        model.train()
        return None

    @staticmethod
    def validation_loss_weight(batch, device):
        frame_mask = batch.get("frame_mask")
        if frame_mask is None:
            raise KeyError("validation requires frame_mask in every batch")
        frame_mask = frame_mask.detach().to(device=device, dtype=torch.float64)
        if frame_mask.device.type == "cpu" and not bool(torch.any(frame_mask > 0)):
            raise ValueError("validation loss requires at least one valid frame per batch")
        return frame_mask.sum()

    def forward_step(self, batch, cfg, model, vis=False, return_loss_dict=False, log_object_pose_metrics=True, log_wandb=True, visualization_frame_indices=None, visualization_geometry=None):
        "one model forward and return loss"
        if getattr(cfg, "body_model", "smpl") == "mhr":
            return self.forward_step_mhr(batch, cfg, model, vis=vis, return_loss_dict=return_loss_dict, log_object_pose_metrics=log_object_pose_metrics, log_wandb=log_wandb, visualization_frame_indices=visualization_frame_indices, visualization_geometry=visualization_geometry)
        if visualization_frame_indices is not None or visualization_geometry is not None:
            raise ValueError("prediction geometry capture is only supported for MHR training")

        # pre-trained model: A is the rendered, B is the input
        rot_delta_pred, rot_delta_gt, trans_delta_gt, trans_delta_pred, out_dict = self.forward_batch(batch, cfg, model, vis, ret_dict=True)
        loss_acc = torch.tensor(0, device=rot_delta_gt.device)
        if cfg['loss_type'] == 'l1':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss =  loss_r + loss_t
        elif cfg['loss_type'] == 'l1+self-acc':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            poseA = batch['pose_perturbed']
            B_in_cams, B_in_cams_gt = self.compute_abspose(poseA.shape[0], batch, cfg, poseA, rot_delta_pred, rot_delta_gt,
                                                           trans_delta_gt, trans_delta_pred)
            d1 = rotation_geodesic_distance_radians(B_in_cams[:, 1:-1, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :-2, :3, :3].reshape(-1, 3, 3))
            d2 = rotation_geodesic_distance_radians(B_in_cams[:, 1:-1, :3, :3].reshape(-1, 3, 3),
                                   B_in_cams[:, 2:, :3, :3].reshape(-1, 3, 3)) # (B, t-2, )
            loss_acc = torch.abs(d1 - d2).mean() * self.cfg.lw_acc
            loss = loss + loss_acc
        elif cfg['loss_type'] == 'l1-abs':
            # predicting absolute pose
            pose_gt = batch['pose_gt'] # (B, T, 4, 4)
            rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1))
            loss_r = torch.abs(rot_delta_pred - rot_gt_axis).mean() * cfg['w_rot']
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            loss_t = torch.abs(trans_delta_pred - trans_gt).mean()
            loss = loss_r + loss_t
            # it is not able to predict depth?
        elif cfg['loss_type'] == 'l1-abs-delta':
            # model predicts both delta and abs pose
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']

            # abs pose error
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            if self.cfg.rot_rep == 'axis_angle':
                rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1)) # BT, 3
                loss_r_abs = torch.abs(out_dict['rot_abs'] - rot_gt_axis).mean() * cfg.w_abs_rot
            elif self.cfg.rot_rep == '6d':
                rot_gt_axis = pose_gt[:, :, :3, 0:3].reshape(-1, 3, 3).view(-1, 6)
                loss_r_abs = torch.abs(out_dict['rot_abs'] - rot_gt_axis).mean() * cfg.w_abs_rot
            else:
                raise NotImplementedError
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            if cfg.loss_abs_trans_rela:
                trans_gt_rela = pose_gt[:, :, :3, 3].clone() - pose_gt[:, 0:1, :3, 3] # relative to 1st frame
                loss_t_abs = torch.abs(out_dict['trans_abs_rela'] - trans_gt_rela).mean() * cfg.w_abs_trans
            else:
                print('loss directly to final GT translation')
                loss_t_abs = torch.abs(out_dict['trans_abs'] - trans_gt).mean() * cfg.w_abs_trans
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs

            if not self.cfg.no_wandb and self.accelerator.is_main_process:
                key = 'train' if self.model.training else 'val'
                wandb.log({f'loss_t_abs_{key}': loss_t_abs, f'loss_r_abs_{key}': loss_r_abs,}, step=self.train_state.step)
        elif cfg['loss_type'] == 'l2-abs-delta':
            # model predicts both delta and abs pose
            loss_t = ((trans_delta_pred - trans_delta_gt)**2).mean()
            loss_r = ((rot_delta_pred - rot_delta_gt)**2).mean() * cfg['w_rot']

            # abs pose error
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1)) # BT, 3
            loss_r_abs = ((out_dict['rot_abs'] - rot_gt_axis)**2).mean() * cfg.w_abs_rot
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            if cfg.loss_abs_trans_rela:
                trans_gt_rela = pose_gt[:, :, :3, 3].clone() - pose_gt[:, 0:1, :3, 3] # relative to 1st frame
                loss_t_abs = ((out_dict['trans_abs_rela'] - trans_gt_rela)**2).mean() * cfg.w_abs_trans
            else:
                loss_t_abs = ((out_dict['trans_abs'] - trans_gt)**2).mean() * cfg.w_abs_trans
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs

            if not self.cfg.no_wandb and self.accelerator.is_main_process:
                key = 'train' if self.model.training else 'val'
                wandb.log({f'loss_t_abs_{key}': loss_t_abs, f'loss_r_abs_{key}': loss_r_abs,}, step=self.train_state.step)
        elif cfg['loss_type'] in ['l1-absrot-delta', 'l1-absrot-delta-hum', 'l2-absrot-delta-humabs', 'l1-absrot-delta-humabs', 'l2-absrot-delta-hum']:
            assert cfg.obj_pose_dim_input in [3, 6], 'must encode object rotation only!'
            loss_func = F.l1_loss if 'l1' in cfg['loss_type'] else F.mse_loss
            key = 'train' if self.model.training else 'val'
            loss_dict = {}

            # model predicts both delta and abs pose, abs pose contains rotation only
            frame_mask = batch['frame_mask'].unsqueeze(-1) # (B, T, 1)
            bs, t = frame_mask.shape[:2]
            FIX_LOSS = 0
            if self.cfg.pred_uncertainty:
                # see https://github.com/martius-lab/beta-nll/blob/master/depth_estimation/models/unet_adaptive_bins.py#L189
                uncert_t = F.softplus(out_dict['trans_uncertainty']) + self.cfg.var_epsilon # (BT, 1) eps to protect
                uncert_r = F.softplus(out_dict['rot_uncertainty']) + self.cfg.var_epsilon
                FIX_LOSS = 20 
                loss_t = 0.5 *(loss_func(trans_delta_pred, trans_delta_gt, reduction='none')/uncert_t + uncert_t.log()+ FIX_LOSS) # * cfg['w_transl']
                loss_t = ((loss_t * (uncert_t.detach() ** cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean()* cfg['w_transl']
                # do the same for r
                loss_r = 0.5 * (loss_func(rot_delta_pred, rot_delta_gt, reduction='none')/uncert_r + uncert_r.log()+ FIX_LOSS)
                loss_r = ((loss_r * (uncert_r.detach()**cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * cfg['w_rot']

                # keep track of the classic loss
                with torch.no_grad():
                    loss_t_raw = (loss_func(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs, t,
                                                                                                    -1) * frame_mask).mean() * cfg['w_transl']
                    loss_r_raw = (loss_func(rot_delta_pred, rot_delta_gt, reduction='none').reshape(bs, t,
                                                                                                -1) * frame_mask).mean() * cfg['w_rot']
                    loss_dict[f'{key}/loss_t_raw'] = loss_t_raw
                    loss_dict[f'{key}/loss_r_raw'] = loss_r_raw

            else:
                if self.cfg.symm_loss:
                    # consider symmetries. 
                    B_in_cams_interm = self.abspose_from_relative(batch, cfg, batch['pose_perturbed'], out_dict['rot'], out_dict['trans']) # (B, T, 4, 4) 
                    pose_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4) 
                    loss_r, loss_t = self.symmetry_aware_object_loss(
                        B_in_cams_interm,
                        pose_gt_symm,
                        frame_mask,
                        loss_func,
                        cfg['w_rot'],
                        cfg['w_transl'],
                        batch.get("obj_symmetry_mode"),
                        batch.get("obj_symmetry_center"),
                    )
                else:
                    loss_t = (loss_func(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean()* cfg['w_transl']
                    loss_r = (loss_func(rot_delta_pred, rot_delta_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg['w_rot']


            # abs pose error, TODO: also add symmetry loss here 
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            if self.cfg.rot_rep == 'axis_angle':
                rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1))  # BT, 3
                loss_r_abs = (loss_func(out_dict['rot_abs'], rot_gt_axis, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg.w_abs_rot
            elif self.cfg.rot_rep == '6d':
                if self.cfg.symm_loss:
                    pose_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4) 
                    loss_r_abs = self.symmetry_aware_rot6d_abs_loss(
                        out_dict['rot_abs'],
                        pose_gt_symm,
                        frame_mask,
                        loss_func,
                        cfg.w_abs_rot,
                        batch.get("obj_symmetry_mode"),
                    )
                else:
                    rot_gt_axis = pose_gt[:, :, :3, 0:2].reshape(-1, 3, 2).reshape(-1, 6)
                    loss_r_abs = (loss_func(out_dict['rot_abs'], rot_gt_axis, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg.w_abs_rot
            else:
                raise NotImplementedError

            loss_t_abs = torch.tensor(0, device=rot_delta_gt.device) # abs do not correct translation
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs
            loss_dict.update(**{f'loss_t_abs_{key}': loss_t_abs, f'loss_r_abs_{key}': loss_r_abs})

            # velocity of the abs object pose 

            # compute additional human pose loss
            if self.cfg.nlf_root is not None:
                if cfg.loss_type in ['l1-absrot-delta-hum', 'l2-absrot-delta-hum']:
                    smpl_delta_r = out_dict['hum_pose']
                    smpl_delta_t = out_dict['hum_trans']

                    if self.cfg.rot_rep_hum == '6d':
                        gt_delta_r = batch['delta_smpl_rot'][:, :, :, :, :2].reshape(-1, 24*6)
                        gt_delta_t = batch['delta_smpl_trans'].reshape(-1, 3)
                        if not self.cfg.pred_uncertainty:
                            loss_hum_t = (loss_func(smpl_delta_t, gt_delta_t, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                            loss_hum_r = (loss_func(smpl_delta_r, gt_delta_r, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                        else:
                            uncert_pose = F.softplus(out_dict['hum_pose_uncertainty']).unsqueeze(-1) + self.cfg.var_epsilon# (BT, 24, 1)
                            uncert_smpl_t = F.softplus(out_dict['hum_trans_uncertainty']) + self.cfg.var_epsilon
                            smpl_delta_r = smpl_delta_r.reshape(-1, 24, 6)
                            gt_delta_r = gt_delta_r.reshape(-1, 24, 6)

                            loss_hum_t = 0.5 * (loss_func(smpl_delta_t, gt_delta_t, reduction='none')/uncert_smpl_t + uncert_smpl_t.log()+ FIX_LOSS)
                            loss_hum_t = ((loss_hum_t * (uncert_smpl_t.detach()**self.cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                            loss_hum_r = 0.5 * (loss_func(smpl_delta_r, gt_delta_r, reduction='none')/uncert_pose + uncert_pose.log()+ FIX_LOSS)
                            loss_hum_r = ((loss_hum_r * (uncert_pose.detach()**self.cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                            # keep track of the classic loss
                            with torch.no_grad():
                                loss_hum_t_raw = (loss_func(smpl_delta_t, gt_delta_t, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_t
                                loss_hum_r_raw = (loss_func(smpl_delta_r, gt_delta_r, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_rot
                                loss_dict[f'{key}/loss_hum_r_raw'] = loss_hum_r_raw
                                loss_dict[f'{key}/loss_hum_t_raw'] = loss_hum_t_raw
                        loss_dict[f'loss_hum_r_{key}'] = loss_hum_r
                        loss_dict[f'loss_hum_t_{key}'] = loss_hum_t

                        loss_hum_j = 0. # joints position loss
                        if self.cfg.w_hum_j > 0.:
                            from lib_smpl import pose72to156

                            betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = self.smpl_params_from_pred(batch, out_dict)

                            male_mask = batch['is_male'].reshape(-1).bool() # B*L, same shape as pred_smpl_pose
                            assert len(male_mask) == len(pred_smpl_pose)
                            idx_m, idx_f = male_mask.nonzero(as_tuple=True)[0], (~male_mask).nonzero(as_tuple=True)[0]
                            idx_list, jtrs_pr_list = [], []
                            J = 24 
                            if idx_m.numel() > 0:
                                idx_list.append(idx_m)
                                # use male smpl model to get joints 
                                jts_pr_m = self.smpl_male.get_joints(pose72to156(pred_smpl_pose[idx_m]), betas[idx_m], pred_smpl_t[idx_m]) # [:, :J] # take only the first 23 joints without wrists 
                                jtrs_pr_list.append(jts_pr_m)
                            if idx_f.numel() > 0:
                                idx_list.append(idx_f)
                                jts_pr_f = self.smpl_female.get_joints(pose72to156(pred_smpl_pose[idx_f]), betas[idx_f], pred_smpl_t[idx_f]) # [:, :J] # take only the first 23 joints without wrists 
                                jtrs_pr_list.append(jts_pr_f)
                            jts_pr_list = torch.cat(jtrs_pr_list, dim=0)
                            perm = torch.cat(idx_list, dim=0)    # original positions of each sub-batch
                            jts_pr = jts_pr_list[torch.argsort(perm)]        # (B, ...), restored to original order

                            jts_gt = batch['smpl_jtrs_gt'].reshape(-1, jts_pr.shape[-2], 3) # .reshape(-1, J, 3)
                            if not self.cfg.pred_uncertainty:
                                loss_hum_j = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_j
                            else:
                                loss_hum_j = 0.5 * (loss_func(jts_pr, jts_gt, reduction='none') / uncert_pose + uncert_pose.log() + FIX_LOSS)
                                loss_hum_j = ((loss_hum_j * (uncert_pose.detach() ** self.cfg.beta_nll)).reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_j
                                with torch.no_grad():
                                    loss_hum_j_raw = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t,-1) * frame_mask).mean() * self.cfg.w_hum_j
                                    loss_dict[f'{key}/loss_hum_j_raw'] = loss_hum_j_raw

                        loss_hum_b = 0.
                        if self.cfg.w_hum_b > 0:
                            loss_hum_b = (loss_func(betas, batch['betas_gt'].reshape(-1, 10), reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_b
                            loss_dict[f'{key}/loss_hum_b'] = loss_hum_b
                        loss_dict[f'loss_hum_j_{key}'] = loss_hum_j

                        # add velocity loss
                        loss_velo = 0.
                        if self.cfg.w_velo > 0:
                            # Oct29 midnight: v2: for human use the joint loss and object use the translation loss
                            jts_pr = jts_pr.reshape(bs, t, -1, 3)
                            jts_gt = batch['smpl_jtrs_gt'] 
                            velo_pr_hj = jts_pr[:, 1:] - jts_pr[:, :-1]
                            velo_gt_hj = jts_gt[:, 1:] - jts_gt[:, :-1]
                            velo_pr_ot = B_in_cams_interm[:, 1:, :3, 3] - B_in_cams_interm[:, :-1, :3, 3]
                            velo_gt_ot = pose_gt[:, 1:, :3, 3] - pose_gt[:, :-1, :3, 3]
                            loss_velo_hj = F.mse_loss(velo_pr_hj, velo_gt_hj, reduction='none').sum(-1).mean() 
                            loss_velo_ot = F.mse_loss(velo_pr_ot, velo_gt_ot, reduction='none').sum(-1).mean()  
                            loss_velo = (loss_velo_hj + loss_velo_ot) * self.cfg.w_velo
                            loss_dict[f'{key}/loss_velo'] = loss_velo
                        # contact prediction
                        loss_contact = 0.
                        contact_result = self.contact_loss_from_output(out_dict, batch, self.cfg, frame_mask=frame_mask)
                        if contact_result is not None:
                            loss_contact, _, _ = contact_result
                            loss_dict[f'{key}/loss_contact'] = loss_contact
                        
                        loss += loss_hum_t + loss_hum_r + loss_hum_j + loss_hum_b + loss_velo + loss_contact
                        print(f'step {self.train_state.step} hum_r:{loss_hum_r:.3f}, hum_t:{loss_hum_t:.3f}, hum_j: {loss_hum_j:.3f}, hum_b: {loss_hum_b:.3f}, velo: {loss_velo:.3f}, r_abs:{loss_r_abs:.3f}, t_abs:{loss_t_abs:.3f}, r: {loss_r:.3f}, t: {loss_t:.3f}, contact: {loss_contact:.3f}, tot: {loss:.3f}')
                    else:
                        raise NotImplementedError
                elif cfg.loss_type in ['l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
                    assert self.cfg.w_hum_j > 0.
                    # now predicting abs
                    trans_pr = out_dict['body_transl'] # (BT, 3)
                    trans_gt = batch['smpl_transl_gt'].reshape(-1, 3)
                    # compute loss on parameters, joints
                    loss_hum_t = (loss_func(trans_pr, trans_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                    rotmat_gt = batch['smpl_rotmat_gt'].reshape(-1, 24, 3, 3)
                    rotmat_pr = out_dict['body_rotmat'] # (BT, 24, 3, 3)
                    loss_hum_r = (loss_func(rotmat_pr, rotmat_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                    bs = batch['smpl_transl_gt'].shape[0]
                    betas = batch['betas_gt'].reshape(self.cfg.clip_len * bs, 10)
                    jts_pr = self.smpl_model.get_joints(rotmat_pr.reshape(self.cfg.clip_len * bs, 24*9), betas, trans_pr, axis2rot=False)  # (BT, J, 3)
                    jts_gt = batch['smpl_jtrs_gt'].reshape(-1, 24, 3)
                    loss_hum_j = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean().item() * self.cfg.w_hum_j
                    loss += loss_hum_t + loss_hum_r + loss_hum_j

                    loss_dict[f'loss_hum_r_{key}'] = loss_hum_r
                    loss_dict[f'loss_hum_t_{key}'] = loss_hum_t
                    loss_dict[f'loss_hum_j_{key}'] = loss_hum_j


            if not self.cfg.no_wandb and self.accelerator.is_main_process:
                wandb.log(loss_dict, step=self.train_state.step)

        elif cfg['loss_type'] == 'l1+geo-acc':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            poseA = batch['pose_perturbed']
            B_in_cams, B_in_cams_gt = self.compute_abspose(poseA.shape[0], batch, cfg, poseA, rot_delta_pred, rot_delta_gt,
                                                           trans_delta_gt, trans_delta_pred)
            d1 = rotation_geodesic_distance_radians(B_in_cams[:, 1:, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :-1, :3, :3].reshape(-1, 3, 3))
            d2 = rotation_geodesic_distance_radians(B_in_cams_gt[:, 1:, :3, :3].reshape(-1, 3, 3),
                                   B_in_cams_gt[:, :-1, :3, :3].reshape(-1, 3, 3))
            loss_acc = torch.abs(d1 - d2).mean() * self.cfg.lw_acc
            print(f'loss_acc: {loss_acc:.4f}, loss: {loss:.4f}')
            loss = loss + loss_acc

        elif cfg['loss_type'] == 'l2': # default L2
            loss_t = ((trans_delta_pred - trans_delta_gt) ** 2).mean()
            loss_r = ((rot_delta_pred - rot_delta_gt) ** 2).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            # add acceleration loss
            if self.cfg.lw_acc>0:
                trans_delta_uno = trans_delta_pred * batch['mesh_diameter'].reshape(len(trans_delta_pred), -1) / 2.
                rot_delta_uno = so3_exp_map(rot_delta_pred * self.cfg['rot_normalizer']).permute(0, 2, 1)
                # now convert to (B, T...)
                poseA = batch['pose_perturbed'] # (B, T, 4, 4)
                poseB = batch['pose_gt']
                B, T = poseA.shape[:2]
                pose_corrected = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4), trans_delta=trans_delta_uno,
                                                          rot_mat_delta=rot_delta_uno)
                pose_corrected = pose_corrected.reshape(B, T, 4, 4)

                acc_t_gt = poseB[:, :-2, :3, 3] - 2 * poseB[:, 1:-1, :3, 3] + poseB[:, 2:, :3, 3]
                acc_t_pred = pose_corrected[:, :-2, :3, 3] - 2 * pose_corrected[:, 1:-1, :3, 3] + pose_corrected[:, 2:, :3, 3]

                axis_gt = so3_log_map(poseB[:, :, :3, :3].reshape(-1, 3, 3)).reshape(B, T, -1)
                axis_pr = so3_log_map(pose_corrected[:, :, :3, :3].reshape(-1, 3, 3)).reshape(B, T, -1)
                acc_r_gt = axis_gt[:, :-2] - 2 * axis_gt[:, 1:-1] + axis_gt[:, 2:]
                acc_r_pr = axis_pr[:, :-2] - 2 * axis_pr[:, 1:-1] + axis_pr[:, 2:]
                la_t = F.l1_loss(acc_t_pred, acc_t_gt).mean()
                la_r = F.l1_loss(acc_r_gt, acc_r_pr).mean()
                loss_acc = (la_r + la_t) * self.cfg.lw_acc
            loss = loss + loss_acc

            # this needs to be computed in the original pose space.
        else:
            raise RuntimeError

        if return_loss_dict:
            return loss, loss_r, loss_t, loss_acc, {}
        return loss, loss_r, loss_t, loss_acc

    def forward_step_mhr(self, batch, cfg, model, vis=False, return_loss_dict=False, log_object_pose_metrics=True, log_wandb=True, visualization_frame_indices=None, visualization_geometry=None):
        if (visualization_frame_indices is None) != (visualization_geometry is None):
            raise ValueError("visualization_frame_indices and visualization_geometry must be provided together")
        if visualization_geometry is not None and len(visualization_geometry) != 0:
            raise ValueError("visualization_geometry collector must be empty")
        out_dict = self.forward_batch_mhr(batch, cfg, model, vis=vis)
        if visualization_frame_indices is None:
            loss, loss_dict = compute_mhr_training_loss(out_dict, batch, cfg, mhr_layer=self.mhr_layer, include_geometry_metrics=not model.training, cuda_event_timer=getattr(self, "active_cuda_event_timer", None))
            visualization_vertices = None
        else:
            loss, loss_dict, visualization_vertices = compute_mhr_training_loss_with_geometry(out_dict, batch, cfg, mhr_layer=self.mhr_layer, visualization_batch_index=0, visualization_frame_indices=visualization_frame_indices, include_geometry_metrics=not model.training, cuda_event_timer=getattr(self, "active_cuda_event_timer", None))
        object_loss = self.object_pose_loss_from_output(batch, cfg, out_dict, self.cfg.symm_loss)
        if object_loss is not None:
            loss_obj, object_loss_dict = object_loss
            loss = loss + loss_obj
            loss_dict.update(object_loss_dict)
            if not model.training or log_object_pose_metrics:
                with torch.no_grad():
                    loss_dict.update(self.final_object_pose_metrics_from_output(batch, cfg, out_dict))
        contact_result = self.contact_loss_from_output(out_dict, batch, cfg)
        if contact_result is not None:
            loss_contact, contact_loss_dict, _ = contact_result
            loss = loss + loss_contact
            loss_dict.update(contact_loss_dict)
        log_loss_dict = loss_dict
        if not log_object_pose_metrics:
            object_pose_metric_keys = set(self.object_pose_metric_keys())
            log_loss_dict = {name: value for name, value in loss_dict.items() if name not in object_pose_metric_keys}
        if log_wandb and not self.cfg.no_wandb and self.accelerator.is_main_process and log_loss_dict:
            key = 'train' if model.training else 'val'
            wandb.log(self.scalar_log_dict({f'{key}/{name}': value.detach() for name, value in log_loss_dict.items()}), step=self.train_state.step)
        if visualization_geometry is not None:
            indices = [int(index) for index in visualization_frame_indices]
            with torch.no_grad():
                visualization_object_poses = self.final_object_pose_from_output(batch, cfg, out_dict)[0, indices].detach().clone()
            visualization_values = {"batch_index": 0, "frame_indices": tuple(indices), "human_vertices": visualization_vertices, "object_poses": visualization_object_poses}
            if self._cfg_value(cfg, "cont_out_dim", -1) > 0:
                if "contact" not in out_dict or "contact_dist_gt" not in batch:
                    raise KeyError("Contact-enabled MHR input-grid visualization requires contact output and contact_dist_gt")
                batch_size, frame_count = batch["contact_dist_gt"].shape[:2]
                visualization_values["contact_output"] = out_dict["contact"].reshape(batch_size, frame_count, -1)[0, indices].detach().clone()
            visualization_geometry.update(visualization_values)
        zero = loss * 0
        loss_r = loss_dict.get("loss_obj_rot", loss_dict.get("loss_delta_mhr_global_rot6d", zero))
        loss_t = loss_dict.get("loss_obj_trans", loss_dict.get("loss_delta_mhr_trans", zero))
        loss_acc = loss_dict.get("loss_mhr_v2v", zero)
        if return_loss_dict:
            return loss, loss_r, loss_t, loss_acc, loss_dict
        return loss, loss_r, loss_t, loss_acc

    @staticmethod
    def _cfg_value(cfg, key, default=None):
        try:
            return cfg[key]
        except Exception:
            return getattr(cfg, key, default)

    @staticmethod
    def _flatten_bt(value, b, t):
        return value.reshape(b * t, -1)

    @staticmethod
    def _masked_frame_mean(per_frame, frame_mask, label):
        if per_frame.ndim != 2:
            raise ValueError(f"{label} must have shape [B,T], got {tuple(per_frame.shape)}")
        mask = frame_mask.to(device=per_frame.device, dtype=per_frame.dtype)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        if tuple(mask.shape) != tuple(per_frame.shape):
            raise ValueError(f"{label} shape {tuple(per_frame.shape)} does not match frame_mask {tuple(mask.shape)}")
        valid_count = mask.sum()
        return (per_frame * mask).sum() / valid_count

    @staticmethod
    def final_object_pose_from_output(batch, cfg, out_dict):
        if "rot" not in out_dict or "trans" not in out_dict:
            raise KeyError("final object pose metrics require rot and trans in model output")
        pose_a = batch["pose_perturbed"]
        b, t = pose_a.shape[:2]
        rot = Trainer._flatten_bt(out_dict["rot"].float(), b, t)
        trans = Trainer._flatten_bt(out_dict["trans"].float(), b, t)
        pose_pred = Trainer.abspose_from_relative_static(batch, cfg, pose_a.to(device=rot.device, dtype=rot.dtype), rot, trans)
        return pose_pred

    @staticmethod
    def final_object_pose_metrics_from_output(batch, cfg, out_dict):
        pose_pred = Trainer.final_object_pose_from_output(batch, cfg, out_dict)
        pose_gt = batch["pose_gt"].to(device=pose_pred.device, dtype=pose_pred.dtype)
        b, t = pose_gt.shape[:2]
        frame_mask = batch.get("object_frame_mask", batch.get("frame_mask"))
        if frame_mask is None:
            frame_mask = torch.ones((b, t), dtype=pose_pred.dtype, device=pose_pred.device)
        frame_mask = frame_mask.to(device=pose_pred.device, dtype=pose_pred.dtype).reshape(b, t)
        valid_count = frame_mask.sum()

        trans_err_m = torch.linalg.norm(pose_pred[:, :, :3, 3] - pose_gt[:, :, :3, 3], dim=-1)
        rot_err_rad = rotation_geodesic_distance_radians(pose_pred[:, :, :3, :3].reshape(b * t, 3, 3), pose_gt[:, :, :3, :3].reshape(b * t, 3, 3)).reshape(b, t)
        rot_err_deg = ((rot_err_rad * 180.0 / torch.pi) * frame_mask).sum() / valid_count
        metrics = {
            "obj_trans_err_m": (trans_err_m * frame_mask).sum() / valid_count,
            "obj_rot_geodesic_deg": rot_err_deg,
        }
        if "pose_gt_symm" in batch:
            pose_gt_symm = batch["pose_gt_symm"].to(device=pose_pred.device, dtype=pose_pred.dtype)
            loss_type = Trainer._cfg_value(cfg, "loss_type", "l1")
            loss_func = F.l1_loss if "l1" in loss_type else F.mse_loss
            rot_symm_err_rad, trans_symm_err_m = Trainer.symmetry_pose_errors(pose_pred, pose_gt_symm, loss_func, Trainer._cfg_value(cfg, "w_rot", 1.0), Trainer._cfg_value(cfg, "w_transl", 1.0), batch.get("obj_symmetry_mode"), batch.get("obj_symmetry_center"), Trainer._cfg_value(cfg, "mhr_object_pose_loss_mode", MHR_OBJECT_POSE_LOSS_LEGACY))
            metrics["obj_trans_err_m"] = (trans_symm_err_m * frame_mask).sum() / valid_count
            rot_symm_err_deg = ((rot_symm_err_rad * 180.0 / torch.pi) * frame_mask).sum() / valid_count
            metrics["obj_rot_geodesic_deg"] = rot_symm_err_deg
        return metrics

    @staticmethod
    def symmetry_descriptor_tensors(pose_pred, symmetry_mode=None, symmetry_center=None):
        batch_size = pose_pred.shape[0]
        if symmetry_mode is None:
            mode = torch.full((batch_size,), OBJECT_SYMMETRY_MODE_FINITE, dtype=torch.long, device=pose_pred.device)
        else:
            mode = torch.as_tensor(symmetry_mode, dtype=torch.long, device=pose_pred.device).reshape(-1)
            if mode.numel() == 1 and batch_size != 1:
                mode = mode.expand(batch_size)
        if tuple(mode.shape) != (batch_size,):
            raise ValueError(f"obj_symmetry_mode must have shape [{batch_size}] with supported values, got {tuple(mode.shape)}")
        if symmetry_center is None:
            center = torch.zeros((batch_size, 3), dtype=pose_pred.dtype, device=pose_pred.device)
        else:
            center = torch.as_tensor(symmetry_center, dtype=pose_pred.dtype, device=pose_pred.device)
            if tuple(center.shape) == (3,) and batch_size == 1:
                center = center.reshape(1, 3)
        if tuple(center.shape) != (batch_size, 3):
            raise ValueError(f"obj_symmetry_center must have shape [{batch_size},3], got {tuple(center.shape)}")
        return mode, center

    @staticmethod
    def symmetry_candidate_indices(pose_pred, pose_gt_symm, loss_func, w_rot, w_transl, symmetry_mode=None, symmetry_center=None, object_pose_loss_mode=MHR_OBJECT_POSE_LOSS_LEGACY):
        pose_gt_symm = pose_gt_symm.to(device=pose_pred.device, dtype=pose_pred.dtype)
        pose_pred_symm = pose_pred[:, :, None].expand_as(pose_gt_symm)
        loss_r_candidates = reduce_mhr_object_rotation_loss(pose_pred_symm[:, :, :, :3, :3], pose_gt_symm[:, :, :, :3, :3], loss_func, object_pose_loss_mode)
        loss_t_candidates = reduce_mhr_object_translation_loss(loss_func(pose_pred_symm[:, :, :, :3, 3], pose_gt_symm[:, :, :, :3, 3], reduction='none'), object_pose_loss_mode)
        mode, center = Trainer.symmetry_descriptor_tensors(pose_pred, symmetry_mode, symmetry_center)
        full_so3 = mode == OBJECT_SYMMETRY_MODE_FULL_SO3
        canonical_gt = pose_gt_symm[:, :, 0]
        predicted_center = torch.matmul(pose_pred[:, :, :3, :3], center[:, None, :, None]).squeeze(-1) + pose_pred[:, :, :3, 3]
        ground_truth_center = torch.matmul(canonical_gt[:, :, :3, :3], center[:, None, :, None]).squeeze(-1) + canonical_gt[:, :, :3, 3]
        center_loss = reduce_mhr_object_translation_loss(loss_func(predicted_center, ground_truth_center, reduction='none'), object_pose_loss_mode)
        full_mask = full_so3[:, None, None]
        loss_r_candidates = torch.where(full_mask, torch.zeros_like(loss_r_candidates), loss_r_candidates)
        loss_t_candidates = torch.where(full_mask, center_loss[:, :, None].expand_as(loss_t_candidates), loss_t_candidates)
        candidate_indices = (loss_r_candidates * w_rot + loss_t_candidates * w_transl).argmin(-1)
        return candidate_indices, loss_r_candidates, loss_t_candidates

    @staticmethod
    def gather_symmetry_candidates(pose_gt_symm, candidate_indices):
        gather_indices = candidate_indices[:, :, None, None, None].expand(-1, -1, 1, 4, 4)
        return torch.gather(pose_gt_symm, 2, gather_indices).squeeze(2)

    @staticmethod
    def symmetry_pose_errors(pose_pred, pose_gt_symm, loss_func, w_rot, w_transl, symmetry_mode=None, symmetry_center=None, object_pose_loss_mode=MHR_OBJECT_POSE_LOSS_LEGACY):
        b, t = pose_pred.shape[:2]
        candidate_indices, _, _ = Trainer.symmetry_candidate_indices(pose_pred, pose_gt_symm, loss_func, w_rot, w_transl, symmetry_mode, symmetry_center, object_pose_loss_mode)
        selected_gt = Trainer.gather_symmetry_candidates(pose_gt_symm, candidate_indices)
        translation_error = torch.linalg.norm(pose_pred[:, :, :3, 3] - selected_gt[:, :, :3, 3], dim=-1)
        rotation_error = rotation_geodesic_distance_radians(pose_pred[:, :, :3, :3].reshape(b * t, 3, 3), selected_gt[:, :, :3, :3].reshape(b * t, 3, 3)).reshape(b, t)
        mode, center = Trainer.symmetry_descriptor_tensors(pose_pred, symmetry_mode, symmetry_center)
        full_so3 = mode == OBJECT_SYMMETRY_MODE_FULL_SO3
        canonical_gt = pose_gt_symm[:, :, 0]
        predicted_center = torch.matmul(pose_pred[:, :, :3, :3], center[:, None, :, None]).squeeze(-1) + pose_pred[:, :, :3, 3]
        ground_truth_center = torch.matmul(canonical_gt[:, :, :3, :3], center[:, None, :, None]).squeeze(-1) + canonical_gt[:, :, :3, 3]
        translation_error = torch.where(full_so3[:, None], torch.linalg.norm(predicted_center - ground_truth_center, dim=-1), translation_error)
        rotation_error = torch.where(full_so3[:, None], torch.zeros_like(rotation_error), rotation_error)
        return rotation_error, translation_error

    @staticmethod
    def symmetry_aware_rot6d_abs_loss(rot_abs, pose_gt_symm, frame_mask, loss_func, w_abs_rot, symmetry_mode=None):
        """Min-over-symmetry absolute 6D rotation loss."""

        b, t = frame_mask.shape[:2]
        rot_abs = rot_abs.reshape(b * t, -1).float()
        if rot_abs.shape[-1] != 6:
            raise ValueError(f"rot_abs must have 6D rotations, got {rot_abs.shape}")
        pose_gt_symm = pose_gt_symm.to(device=rot_abs.device, dtype=rot_abs.dtype)
        frame_mask = frame_mask.to(device=rot_abs.device, dtype=rot_abs.dtype).reshape(b, t, 1)

        n_symm = pose_gt_symm.shape[2]
        pose_gt_symm6d = pose_gt_symm[..., :3, 0:2].reshape(b * t, n_symm, 6)
        rot_abs_symm = rot_abs[:, None].expand_as(pose_gt_symm6d)
        per_sym_loss = loss_func(rot_abs_symm, pose_gt_symm6d, reduction='none').mean(-1)
        loss_r_abs = per_sym_loss.min(-1)[0]
        if symmetry_mode is not None:
            mode = torch.as_tensor(symmetry_mode, dtype=torch.long, device=rot_abs.device).reshape(-1)
            if mode.numel() == 1 and b != 1:
                mode = mode.expand(b)
            if tuple(mode.shape) != (b,):
                raise ValueError(f"obj_symmetry_mode must have shape [{b}], got {tuple(mode.shape)}")
            loss_r_abs = torch.where((mode == OBJECT_SYMMETRY_MODE_FULL_SO3)[:, None].expand(b, t).reshape(-1), torch.zeros_like(loss_r_abs), loss_r_abs)
        return Trainer._masked_frame_mean(loss_r_abs.reshape(b, t), frame_mask, "absolute object rotation loss") * w_abs_rot

    @staticmethod
    def contact_targets_from_distances(contact_dist_gt):
        if contact_dist_gt.shape[-1] == 2:
            return contact_dist_gt
        if contact_dist_gt.shape[-1] > 38:
            return contact_dist_gt[:, :, [22, 23 + 15]]
        raise ValueError(f"contact_dist_gt needs 2 MHR hand-surface channels or legacy 52-D wrist-joint contacts, got {contact_dist_gt.shape}")

    @staticmethod
    def contact_visualization_values(contact_output, contact_dist_gt, cfg):
        contact_dist_gt = Trainer.contact_targets_from_distances(contact_dist_gt).float()
        contact_output = contact_output.float()
        if contact_output.shape != contact_dist_gt.shape or contact_output.shape[-1] != 2:
            raise ValueError(f"Contact visualization requires matching left/right [T,2] predictions and targets, got {tuple(contact_output.shape)} and {tuple(contact_dist_gt.shape)}")
        threshold_m = float(Trainer._cfg_value(cfg, "cont_mask_thres", 0.015))
        if not np.isfinite(threshold_m) or threshold_m <= 0:
            raise ValueError(f"cont_mask_thres must be positive and finite, got {threshold_m}")
        ground_truth_contacts = contact_dist_gt < threshold_m
        contact_output_type = Trainer._cfg_value(cfg, "cont_out_type", "binary")
        if contact_output_type == "binary":
            prediction_contact_values = torch.sigmoid(contact_output)
            prediction_contact_value_type = "probability"
        elif contact_output_type == "distance":
            prediction_contact_values = contact_output < threshold_m
            prediction_contact_value_type = "binary"
        else:
            raise ValueError(f"Unknown cont_out_type={contact_output_type}")
        return prediction_contact_values, prediction_contact_value_type, ground_truth_contacts

    @staticmethod
    def contact_loss_from_output(out_dict, batch, cfg, frame_mask=None):
        if Trainer._cfg_value(cfg, "cont_out_dim", -1) <= 0:
            return None
        if "contact" not in out_dict:
            raise KeyError("cont_out_dim > 0 but model output has no contact head")
        if "contact_dist_gt" not in batch:
            raise KeyError("cont_out_dim > 0 requires contact_dist_gt in the batch")

        cont_pred = out_dict["contact"].float()
        contact_dist = batch["contact_dist_gt"].to(device=cont_pred.device, dtype=cont_pred.dtype)
        cont_gt_hands = Trainer.contact_targets_from_distances(contact_dist)
        b, t = cont_gt_hands.shape[:2]
        cont_pred = cont_pred.reshape(b, t, -1)
        if cont_pred.shape[-1] != cont_gt_hands.shape[-1]:
            raise ValueError(f"contact head output dim {cont_pred.shape[-1]} != target dim {cont_gt_hands.shape[-1]}")

        if frame_mask is None:
            frame_mask = batch.get("frame_mask")
        if frame_mask is None:
            frame_mask = torch.ones((b, t), dtype=cont_pred.dtype, device=cont_pred.device)
        frame_mask = frame_mask.to(device=cont_pred.device, dtype=cont_pred.dtype)
        if frame_mask.ndim == 2:
            frame_mask = frame_mask.reshape(b, t, 1)

        if Trainer._cfg_value(cfg, "cont_out_type", "binary") == 'binary':
            contact_target = (cont_gt_hands < Trainer._cfg_value(cfg, "cont_mask_thres", 0.015)).to(cont_pred.dtype)
            loss_contact = F.binary_cross_entropy_with_logits(
                cont_pred,
                contact_target,
                reduction='none',
            )
        elif Trainer._cfg_value(cfg, "cont_out_type", "binary") == 'distance':
            loss_contact = F.mse_loss(cont_pred, cont_gt_hands, reduction='none')
        else:
            raise ValueError(f"Unknown cont_out_type={Trainer._cfg_value(cfg, 'cont_out_type')}")

        loss_contact = Trainer._masked_frame_mean(loss_contact.mean(dim=-1), frame_mask, "contact loss") * Trainer._cfg_value(cfg, "w_contact", 0.0)
        return loss_contact, {"loss_contact": loss_contact}, cont_gt_hands

    @staticmethod
    def object_pose_loss_from_output(batch, cfg, out_dict, symm_loss):
        if "rot" not in out_dict or "trans" not in out_dict:
            return None

        loss_type = Trainer._cfg_value(cfg, "loss_type")
        loss_func = F.l1_loss if 'l1' in loss_type else F.mse_loss
        rot_pred = out_dict["rot"].float()
        trans_pred = out_dict["trans"].float()
        device = rot_pred.device
        dtype = rot_pred.dtype

        trans_delta_gt = batch["delta_transl"].to(device=device, dtype=dtype).clone()
        b, t = trans_delta_gt.shape[:2]
        frame_mask = batch.get("object_frame_mask", batch.get("frame_mask"))
        if frame_mask is None:
            frame_mask = torch.ones((b, t), dtype=dtype, device=device)
        frame_mask = frame_mask.to(device=device, dtype=dtype).reshape(b, t, 1)

        if Trainer._cfg_value(cfg, "normalize_xyz", True):
            trans_delta_gt = normalize_mhr_translation(trans_delta_gt, batch)
        else:
            trans_normalizer = batch["trans_normalizer"].to(device=device, dtype=dtype)
            trans_delta_gt = trans_delta_gt / trans_normalizer
        trans_delta_gt = trans_delta_gt.reshape(b * t, 3)

        rot_delta_mat_gt = batch["delta_rot"].to(device=device, dtype=dtype)
        rot_delta_gt = so3_log_map(rot_delta_mat_gt.reshape(b * t, 3, 3).permute(0, 2, 1))
        rot_delta_gt = rot_delta_gt / float(Trainer._cfg_value(cfg, "rot_normalizer"))

        rot_pred = Trainer._flatten_bt(rot_pred, b, t)
        trans_pred = Trainer._flatten_bt(trans_pred, b, t)

        if symm_loss:
            if "pose_gt_symm" not in batch:
                raise KeyError("symm_loss=True requires pose_gt_symm in the MHR batch")
            # out_dict["rot"] / ["trans"] are first-stage egocentric deltas from
            # pose_perturbed.  For symmetric objects, a delta target is ambiguous:
            # several absolute object poses are equivalent.  Compose the predicted
            # delta into an absolute pose, then select over pose_gt @ symmetry_tf.
            pose_pred = Trainer.abspose_from_relative_static(
                batch,
                cfg,
                batch["pose_perturbed"].to(device=device, dtype=dtype),
                rot_pred,
                trans_pred,
            )
            loss_r, loss_t = Trainer.symmetry_aware_object_loss(
                pose_pred,
                batch["pose_gt_symm"],
                frame_mask,
                loss_func,
                Trainer._cfg_value(cfg, "w_rot"),
                Trainer._cfg_value(cfg, "w_transl"),
                batch.get("obj_symmetry_mode"),
                batch.get("obj_symmetry_center"),
                Trainer._cfg_value(cfg, "mhr_object_pose_loss_mode", MHR_OBJECT_POSE_LOSS_LEGACY),
            )
        else:
            # Non-symmetric translation retains normalized delta supervision.
            # The current rotation contract compares unnormalized delta matrices
            # geodesically; recorded legacy modes retain coordinate-space loss.
            loss_t = Trainer._masked_frame_mean(loss_func(trans_pred, trans_delta_gt, reduction='none').reshape(b, t, -1).mean(dim=-1), frame_mask, "object translation loss")
            loss_t = loss_t * Trainer._cfg_value(cfg, "w_transl")
            object_pose_loss_mode = Trainer._cfg_value(cfg, "mhr_object_pose_loss_mode", MHR_OBJECT_POSE_LOSS_LEGACY)
            if object_pose_loss_mode == MHR_OBJECT_POSE_LOSS_GEODESIC_RADIAN_EQUAL_WEIGHT:
                rot_pred_matrix = so3_exp_map(rot_pred * float(Trainer._cfg_value(cfg, "rot_normalizer"))).permute(0, 2, 1)
                loss_r_frames = reduce_mhr_object_rotation_loss(rot_pred_matrix, rot_delta_mat_gt.reshape(b * t, 3, 3), loss_func, object_pose_loss_mode).reshape(b, t)
            else:
                loss_r_frames = loss_func(rot_pred, rot_delta_gt, reduction='none').reshape(b, t, -1).mean(dim=-1)
            loss_r = Trainer._masked_frame_mean(loss_r_frames, frame_mask, "object rotation loss")
            loss_r = loss_r * Trainer._cfg_value(cfg, "w_rot")

        loss_obj = loss_r + loss_t
        loss_dict = {
            "loss_obj_total": loss_obj,
            "loss_obj_rot": loss_r,
            "loss_obj_trans": loss_t,
            "loss_obj_rot_delta": loss_r,
            "loss_obj_trans_delta": loss_t,
        }
        return loss_obj, loss_dict

    @staticmethod
    def symmetry_aware_object_loss(pose_pred, pose_gt_symm, frame_mask, loss_func, w_rot, w_transl, symmetry_mode=None, symmetry_center=None, object_pose_loss_mode=MHR_OBJECT_POSE_LOSS_LEGACY):
        """Select one equivalent pose, then supervise its rotation and translation."""

        pose_gt_symm = pose_gt_symm.to(device=pose_pred.device, dtype=pose_pred.dtype)
        frame_mask = frame_mask.to(device=pose_pred.device, dtype=pose_pred.dtype)
        candidate_indices, loss_r_candidates, loss_t_candidates = Trainer.symmetry_candidate_indices(pose_pred, pose_gt_symm, loss_func, w_rot, w_transl, symmetry_mode, symmetry_center, object_pose_loss_mode)
        loss_r = torch.gather(loss_r_candidates, -1, candidate_indices.unsqueeze(-1)).squeeze(-1)
        loss_t = torch.gather(loss_t_candidates, -1, candidate_indices.unsqueeze(-1)).squeeze(-1)
        loss_r = Trainer._masked_frame_mean(loss_r, frame_mask, "symmetry-aware object rotation loss") * w_rot
        loss_t = Trainer._masked_frame_mean(loss_t, frame_mask, "symmetry-aware object translation loss") * w_transl
        return loss_r, loss_t

    @staticmethod
    def abspose_from_relative_static(batch, cfg, poseA, rot, trans_delta_pred):
        b, t = poseA.shape[:2]
        trans_delta_final = denormalize_mhr_translation(trans_delta_pred.reshape(b, t, 3), batch).reshape(-1, 3)
        rot_delta_final = so3_exp_map(rot * Trainer._cfg_value(cfg, "rot_normalizer")).permute(0, 2, 1)
        B_in_cams = Utils.egocentric_delta_pose_to_pose(
            poseA.reshape(-1, 4, 4),
            trans_delta=trans_delta_final,
            rot_mat_delta=rot_delta_final,
        ).reshape(b, t, 4, 4)
        return B_in_cams

    def forward_batch_mhr(self, batch, cfg, model, vis=False):
        prepare_mhr_spatial_batch(batch, cfg, self.mhr_layer)
        if "mhr_model_output" in batch:
            output = batch["mhr_model_output"]
        else:
            required = ["input_rgbs", "render_rgbs", "input_xyz", "render_xyz"]
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError(f"MHR training batch is missing rendered input fields: {missing}")
            imgsB, imgsA = batch['input_rgbs'], batch['render_rgbs']
            xyzB, xyzA = batch['input_xyz'], batch['render_xyz']
            obj_pose = batch.get('poseA_norm', batch.get('pose_perturbed'))
            output = model(torch.cat([imgsA, xyzA], 2), torch.cat([imgsB, xyzB], 2), obj_pose, batch)

        if "mhr_trans_init" in batch:
            B, T = batch["mhr_trans_init"].shape[:2]
            for key, value in list(output.items()):
                if key.startswith("delta_mhr_") and len(value.shape) == 2 and value.shape[0] == B * T:
                    output[key] = value.reshape(B, T, -1)
        if model is not None and not model.training:
            if self.mhr_supervision_contract is None:
                raise RuntimeError("MHR inference requires a resolved training supervision contract")
            output = apply_mhr_supervision_contract(output, self.mhr_supervision_contract)
        return output

    def log_mhr_input_viz(self, batch, cfg, model, key=None, log_wandb=True, prediction_geometry=None):
        start = time.time()
        bid = 0
        key = key or ('train' if model.training else 'val')
        clip_len = batch['render_rgbs'].shape[1]
        frame_indices = evenly_spaced_frame_indices(clip_len, int(getattr(cfg, "mhr_input_viz_num_frames", 10)))
        if prediction_geometry is None:
            raise ValueError("MHR input-grid visualization requires prediction geometry from the current forward pass")
        prediction_frame_indices = [int(index) for index in prediction_geometry.get("frame_indices", ())]
        if prediction_frame_indices != frame_indices or int(prediction_geometry.get("batch_index", -1)) != bid:
            raise ValueError(f"MHR prediction geometry selection batch={prediction_geometry.get('batch_index')} frames={prediction_frame_indices} does not match grid batch={bid} frames={frame_indices}")
        prediction_fields = ["human_vertices", "object_poses"]
        if self._cfg_value(cfg, "cont_out_dim", -1) > 0:
            prediction_fields.append("contact_output")
        missing_prediction = [field for field in prediction_fields if field not in prediction_geometry]
        if missing_prediction:
            raise KeyError(f"MHR input-grid prediction overlay requires fields: {missing_prediction}")
        maskA, maskB, rgbsA, rgbsB, xyzA, xyzB = self.prepare_input_viz(batch, cfg, batch_index=bid, frame_indices=frame_indices)
        metadata_rows = batch.get(MHR_SAMPLE_METADATA_KEY)
        metadata = metadata_rows[bid] if isinstance(metadata_rows, list) and len(metadata_rows) > bid else None
        if metadata is None or metadata.get("frame_names") is None:
            raise ValueError("MHR input-grid visualization requires ordered sample metadata with frame_names")
        metadata_frame_names = [str(value) for value in metadata["frame_names"]]
        if max(frame_indices, default=-1) >= len(metadata_frame_names):
            raise ValueError(f"MHR input-grid frame indices {frame_indices} exceed metadata frame count {len(metadata_frame_names)}")
        selected_frame_names = [metadata_frame_names[index] for index in frame_indices]
        viz_dataset = self.dataset_train if key == 'train' else self.dataset_test
        background_rgbs = viz_dataset.load_input_viz_background_rgbs(str(metadata["sequence_name"]), int(metadata["camera_id"]), selected_frame_names)
        missing_ground_truth = [field for field in ("pose_gt", "K_rois") if field not in batch]
        if missing_ground_truth:
            raise KeyError(f"MHR input-grid ground-truth overlay requires batch fields: {missing_ground_truth}")
        human_faces, object_mesh_source, object_mesh_to_pose_transform = viz_dataset.load_input_viz_mesh_geometry(str(metadata["sequence_name"]))
        if self.mhr_input_grid_renderer is None:
            from tools.mhr_mesh_renderer import NvdiffMeshRenderer

            self.mhr_input_grid_renderer = NvdiffMeshRenderer(str(self.accelerator.device))
        gt_params = gt_mhr_params_from_batch(batch)
        missing_params = sorted(set(MHR_PARAM_DIMS) - set(gt_params))
        if missing_params:
            loaded_gt_params = viz_dataset.load_input_viz_gt_mhr_params(str(metadata["sequence_name"]), int(metadata["camera_id"]), selected_frame_names)
            selected_gt_params = {name: torch.as_tensor(value, dtype=torch.float32, device=self.accelerator.device).unsqueeze(0) for name, value in loaded_gt_params.items()}
        else:
            selected_gt_params = {name: value[bid:bid + 1, frame_indices] for name, value in gt_params.items()}
        with torch.no_grad():
            human_vertices_gt = self.mhr_layer.mhr_forward_vertices(selected_gt_params)[0].detach().cpu().numpy()
        object_poses_gt = batch["pose_gt"][bid, frame_indices].detach().cpu().numpy()
        intrinsics = batch["K_rois"][bid, frame_indices].detach().cpu().numpy()
        human_vertices_pred = prediction_geometry["human_vertices"].detach().cpu().numpy()
        object_poses_pred = prediction_geometry["object_poses"].detach().cpu().numpy()
        prediction_rgbs, prediction_masks = render_human_object_mesh_tiles(self.mhr_input_grid_renderer, human_vertices_pred, human_faces, object_mesh_source, object_mesh_to_pose_transform, object_poses_pred, intrinsics, background_rgbs.shape[-2:])
        ground_truth_rgbs, ground_truth_masks = render_human_object_mesh_tiles(self.mhr_input_grid_renderer, human_vertices_gt, human_faces, object_mesh_source, object_mesh_to_pose_transform, object_poses_gt, intrinsics, background_rgbs.shape[-2:])
        frame_valid_mask = batch.get('frame_mask')
        frame_valid_mask = None if frame_valid_mask is None else frame_valid_mask[bid, frame_indices].detach().cpu().numpy()
        prediction_contact_values = prediction_contact_value_type = ground_truth_contacts = None
        contact_threshold_m = float(self._cfg_value(cfg, "cont_mask_thres", 0.015))
        if self._cfg_value(cfg, "cont_out_dim", -1) > 0:
            if "contact_dist_gt" not in batch:
                raise KeyError("Contact-enabled MHR input-grid visualization requires contact_dist_gt")
            contact_dist_gt = self.contact_targets_from_distances(batch["contact_dist_gt"])[bid, frame_indices]
            prediction_contact_values, prediction_contact_value_type, ground_truth_contacts = self.contact_visualization_values(prediction_geometry["contact_output"], contact_dist_gt, cfg)
            prediction_contact_values = prediction_contact_values.detach().cpu().numpy()
            ground_truth_contacts = ground_truth_contacts.detach().cpu().numpy()
        grid_xyzA = xyzA if maskA is None else np.concatenate([xyzA, maskA.astype(np.float32) / 255.0], axis=2)
        grid_xyzB = xyzB if maskB is None else np.concatenate([xyzB, maskB.astype(np.float32) / 255.0], axis=2)
        grid, caption = build_mhr_input_grid(rgbsA[0], rgbsB[0], grid_xyzA[0], grid_xyzB[0], frame_indices, background_rgbs=background_rgbs, prediction_rgbs=prediction_rgbs, prediction_masks=prediction_masks, ground_truth_rgbs=ground_truth_rgbs, ground_truth_masks=ground_truth_masks, prediction_contact_values=prediction_contact_values, prediction_contact_value_type=prediction_contact_value_type, ground_truth_contacts=ground_truth_contacts, contact_threshold_m=contact_threshold_m, metadata=metadata, frame_valid_mask=frame_valid_mask, global_step=self.train_state.step, split=key)
        outfile = osp.join(self.exp_dir, f'vis/{key}_mhr_input_grid_latest.png')
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        Image.fromarray(grid).save(outfile)
        wandb_image = wandb.Image(grid, caption=caption)
        if log_wandb:
            wandb.log({f'{key}/input_grid': wandb_image}, step=self.train_state.step)
        end = time.time()
        action = "uploading" if log_wandb else "generation"
        print(f'Step {self.train_state.step} {key} MHR vis {action} finished after {end - start} seconds')
        return wandb_image

    def forward_batch(self, batch, cfg, model, vis=False, ret_dict=False, vis_key=None):
        "forward one batch"
        if getattr(cfg, "body_model", "smpl") == "mhr":
            out_dict = self.forward_batch_mhr(batch, cfg, model, vis=vis)
            if ret_dict:
                return None, None, None, None, out_dict
            return None, None, None, None

        imgsB, imgsA = batch['input_rgbs'], batch['render_rgbs']
        xyzB, xyzA = batch['input_xyz'], batch['render_xyz']
        pose_perturbed = batch['poseA_norm']
        output = model(torch.cat([imgsA, xyzA], 2), torch.cat([imgsB, xyzB], 2), pose_perturbed, batch)
        trans_delta_gt = batch['delta_transl']  # (B, T, 3)
        mesh_radius = batch['mesh_diameter'] / 2.  # (B, T)
        trans_normalizer = batch['trans_normalizer']  # (B, T, 3)
        B, T = trans_delta_gt.shape[:2]
        # use diameter: the xyz map is normalized by object diameter
        if cfg['normalize_xyz']:
            trans_delta_gt *= 1 / mesh_radius.reshape(len(trans_delta_gt), T, -1)
        else:
            trans_delta_gt = trans_delta_gt / trans_normalizer
            if not (torch.abs(trans_delta_gt) <= 1 + 1e-3).all():
                logging.info("ERROR label")
        rot_delta_mat_gt = batch['delta_rot']
        rot_delta_gt = so3_log_map(rot_delta_mat_gt.reshape(B * T, 3, 3).permute(0, 2, 1))  # permute: pyt3d so3 uses col order
        rot_delta_gt = rot_delta_gt / cfg['rot_normalizer']  # random noise sample range.
        trans = output['trans'].float()  # (BT,3)
        rot = output['rot'].float()  # BT, 3
        trans_delta_pred = trans
        trans_delta_gt = trans_delta_gt.reshape(B * T, 3)  # FP was trained to predict

        # log error
        if self.accelerator.is_main_process and not cfg.no_wandb:
            with torch.no_grad():
                log_dict = {}
                poseA = batch['pose_perturbed']
                B_in_cams, B_in_cams_gt = self.compute_abspose(B, batch, cfg, poseA, rot, rot_delta_gt,
                                                               trans_delta_gt, trans_delta_pred, output)
                err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams[:, :, :3, 3]) ** 2, -1).sqrt().mean()
                err_r = rotation_geodesic_distance_radians(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :, :3, :3].reshape(-1, 3, 3)).mean()
                key = vis_key or ('train' if model.training else 'val')
                log_dict[f'{key}/err_t'] = err_t
                log_dict[f'{key}/err_r'] = err_r
                log_dict[f'{key}/err_r_deg'] = err_r * 180/torch.pi

                # log contact accuracy 
                if self.cfg.cont_out_dim > 0:
                    cont_pred = output['contact'].reshape(B, T, -1)
                    cont_gt_hands = self.contact_targets_from_distances(batch['contact_dist_gt']).to(
                        device=cont_pred.device,
                        dtype=cont_pred.dtype,
                    )
                    if self.cfg.cont_out_type == 'binary':
                        cont_acc = (cont_pred > 0).float() == (cont_gt_hands < self.cfg.cont_mask_thres).float()
                    elif self.cfg.cont_out_type == 'distance':
                        cont_acc = (cont_pred < self.cfg.cont_mask_thres).float() == (cont_gt_hands < self.cfg.cont_mask_thres).float()
                    log_dict[f'{key}/cont_acc'] = cont_acc.float().mean()
                # log error of intermediate predictions
                if self.cfg['loss_type'] in ['l1-abs-delta', 'l1-absrot-delta', 'l1-absrot-delta-hum', 'l1-absrot-delta-humabs', 'l2-absrot-delta-humabs']:
                    B_in_cams_interm = self.abspose_from_relative(batch, cfg, poseA, output['rot'], output['trans'])
                    # also compute symmetries 
                    if self.cfg.symm_loss:
                        B_in_cams_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4)
                        loss_func = F.l1_loss if "l1" in self.cfg["loss_type"] else F.mse_loss
                        err_r_frames, err_t_frames = self.symmetry_pose_errors(B_in_cams_interm, B_in_cams_gt_symm, loss_func, self.cfg["w_rot"], self.cfg["w_transl"], batch.get("obj_symmetry_mode"), batch.get("obj_symmetry_center"))
                        err_r = err_r_frames.mean()
                        err_t = err_t_frames.mean()
                    else:
                        err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams_interm[:, :, :3, 3]) ** 2, -1).sqrt().mean()
                        err_r = rotation_geodesic_distance_radians(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3),
                                                B_in_cams_interm[:, :, :3, :3].reshape(-1, 3, 3)).mean()

                    key = vis_key or ('train' if model.training else 'val')
                    log_dict[f'{key}/err_interm_t'] = err_t
                    log_dict[f'{key}/err_interm_r'] = err_r
                wandb.log(log_dict, step=self.train_state.step)

        # visualize input and output predictions
        if vis and self.accelerator.is_main_process:
            start = time.time()
            bid = 0  # batch id
            skip = 16  # log every N frame

            key = vis_key or ('train' if self.model.training else 'val')
            log_dict = {}
            maskA, maskB, rgbsA, rgbsB, xyzA, xyzB = self.prepare_input_viz(batch, cfg)
            poseB = batch['pose_gt'] # this does not match B_in_cams_gt!
            # TODO: replace poseB with poseA + delta GT

            poseA = batch['pose_perturbed']
            to_origin = batch['to_origin'][bid].cpu().numpy()
            bbox = batch['obj_bbox_3d'][bid].cpu().numpy()
            clip_len = rgbsA.shape[1]

            # log error
            with torch.no_grad():
                B_in_cams, B_in_cams_gt = self.compute_abspose(B, batch, cfg, poseA, rot, rot_delta_gt,
                                                               trans_delta_gt, trans_delta_pred, output)
                err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams[:, :, :3, 3])**2, -1).sqrt().mean()
                err_r = rotation_geodesic_distance_radians(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3), B_in_cams[:, :, :3, :3].reshape(-1, 3, 3)).mean()
                key = vis_key or ('train' if model.training else 'val')
                log_dict[f'{key}/err_t'] = err_t
                log_dict[f'{key}/err_r'] = err_r

                # log SMPL evaluation
                NJ = 24
                if cfg.nlf_root is not None:
                    verts_smpl_gt, verts_smpl, jtrs_gt, jtrs_pr, pred_smpl_r, pred_smpl_t, jts_rot_pr, jts_rot_gt = self.compute_smpl_verts(
                        batch, output)
                    v2v = torch.sum((verts_smpl_gt - verts_smpl) ** 2, -1).sqrt().mean()
                    mpjpe = torch.sum((jtrs_pr - jtrs_gt) ** 2, -1).sqrt().mean()
                    mpjae = Utils.geodesic_distance_batch(jts_rot_pr, jts_rot_gt).mean()
                    ste = torch.sum((batch['smpl_transl_gt'].reshape(-1, 3) - pred_smpl_t) ** 2).sqrt().mean()
                    log_dict[f'{key}/v2v'] = v2v
                    log_dict[f'{key}/mpjpe'] = mpjpe
                    log_dict[f'{key}/mpjae'] = mpjae
                    log_dict[f'{key}/smpl_t'] = ste

            for i in range(0, clip_len, skip):
                comb, rgba, rgbb = self.visualize_rgbm(batch, bid, i, maskA, maskB, rgbsA, rgbsB)
                # add xyz as well
                xyza_vis = (np.clip(xyzA[bid, i].transpose(1, 2, 0)+0.5, 0, 1.)* 255).astype(np.uint8)
                xyzb_vis = (np.clip(xyzB[bid, i].transpose(1, 2, 0)+0.5, 0, 1.)* 255).astype(np.uint8)
                comb = np.concatenate([comb, np.concatenate([xyza_vis, xyzb_vis], 0)], axis=1)
                # Visualize pose predictions as well
                K = batch['K_rois'][bid, i].cpu().numpy()
                center_pose = B_in_cams_gt[bid, i].cpu().numpy() @ np.linalg.inv(to_origin)
                vis_gt, vis_input, vis_pred = rgbb.copy(), rgba.copy(), rgbb.copy()
                vis_gt = Utils.draw_posed_3d_box(K, img=vis_gt, ob_in_cam=center_pose, bbox=bbox, line_color=(0, 255, 0))
                vis_gt = Utils.draw_xyz_axis(vis_gt, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                             transparency=0, is_input_rgb=True)
                center_pose = poseA[0, i].cpu().numpy() @ np.linalg.inv(to_origin)
                vis_input = Utils.draw_posed_3d_box(K, img=vis_input, ob_in_cam=center_pose, bbox=bbox, line_color=(255, 0, 0))
                vis_input = Utils.draw_xyz_axis(vis_input, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                             transparency=0, is_input_rgb=True)

                pose = B_in_cams[bid, i].detach().cpu().numpy()
                center_pose = pose @ np.linalg.inv(to_origin)
                vis_pred = Utils.draw_posed_3d_box(K, img=vis_pred, ob_in_cam=center_pose, bbox=bbox, line_color=(0, 255, 255))
                vis_pred = Utils.draw_xyz_axis(vis_pred, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                                transparency=0, is_input_rgb=True)

                # now show an overlap
                vis_comb = vis_gt.copy()
                vis_comb = Utils.draw_posed_3d_box(K, img=vis_comb, ob_in_cam=center_pose, bbox=bbox,
                                                   line_color=(0, 255, 255))
                vis_comb = Utils.draw_xyz_axis(vis_comb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                               transparency=0, is_input_rgb=True)

                # add contact text 
                if self.cfg.cont_out_dim > 0:
                    cont_gt = self.contact_targets_from_distances(batch['contact_dist_gt'])[bid, i]
                    cont_text = f'lh: {cont_gt[0]:.3f}, rh: {cont_gt[1]:.3f}'
                    cv2.putText(vis_gt, cont_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                    # add to vis_pred as well 
                    cont_pred = output['contact'].reshape(B, T, -1)[bid, i]
                    cont_text = f'lh: {cont_pred[0]:.3f}, rh: {cont_pred[1]:.3f}'
                    cv2.putText(vis_pred, cont_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                pose_comb = np.concatenate([np.concatenate([vis_input, vis_pred], 1),
                                            np.concatenate([vis_gt, vis_comb], 1)], axis=0)
                comb = np.concatenate([comb, pose_comb], axis=1)

                # visualize PC
                pc_ab, pc_colors = self.visualize_xyz_map(bid, i, xyzA, xyzB)
                log_dict[f'{key}_xyz_{i}'] = wandb.Object3D(np.concatenate([pc_ab, pc_colors], 1),
                                                            caption=f'xyz: red-A green-B ')

                log_dict[f'{key}_all_{i}'] = wandb.Image(comb, caption='top-A bottom-B')
                outfile = osp.join(self.exp_dir, f'vis/{key}_xyz_{i}.ply')
                os.makedirs(osp.dirname(outfile), exist_ok=True)
                trimesh.PointCloud(pc_ab, colors=pc_colors).export(outfile)
                outfile = osp.join(self.exp_dir, f'vis/{key}_all_{i}.png')
                Image.fromarray(comb).save(outfile)

            if not self.cfg.no_wandb:
                wandb.log(log_dict, step=self.train_state.step)
            end = time.time()
            print(f'Step {self.train_state.step} {key} vis uploading finished after {end - start} seconds')

        if ret_dict:
            return rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, output
        return rot, rot_delta_gt, trans_delta_gt, trans_delta_pred

    def prepare_input_viz(self, batch, cfg, batch_index=None, frame_indices=None):
        if (batch_index is None) != (frame_indices is None):
            raise ValueError("batch_index and frame_indices must be provided together")

        def select(tensor):
            if batch_index is None:
                return tensor
            return tensor[batch_index:batch_index + 1, frame_indices]

        render_rgbs = select(batch['render_rgbs'])
        input_rgbs = select(batch['input_rgbs'])
        render_xyz = select(batch['render_xyz'])
        input_xyz = select(batch['input_xyz'])
        channel_counts = [render_rgbs.shape[2], input_rgbs.shape[2], render_xyz.shape[2], input_xyz.shape[2]]
        packed = torch.cat([render_rgbs, input_rgbs, render_xyz, input_xyz], dim=2).detach().cpu().numpy()
        offsets = np.cumsum([0] + channel_counts)
        rgbsA = (np.clip(packed[:, :, offsets[0]:offsets[1]], 0, 1) * 255).astype(np.uint8)
        rgbsB = (np.clip(packed[:, :, offsets[1]:offsets[2]], 0, 1) * 255).astype(np.uint8)
        packed_xyzA = packed[:, :, offsets[2]:offsets[3]]
        packed_xyzB = packed[:, :, offsets[3]:offsets[4]]
        xyzA, xyzB = packed_xyzA[:, :, :3], packed_xyzB[:, :, :3]
        if input_xyz.shape[2] in [5, 6]:
            maskA = (packed_xyzA[:, :, 3:] * 255).astype(np.uint8)
            maskB = (packed_xyzB[:, :, 3:] * 255).astype(np.uint8)
        elif cfg.mask_encode_type == 'one-channel':
            maskA = ((packed_xyzA[:, :, 3:] + 1) / 2 * 255).astype(np.uint8)
            maskB = ((packed_xyzB[:, :, 3:] + 1) / 2 * 255).astype(np.uint8)
        else:
            maskA, maskB = None, None
        return maskA, maskB, rgbsA, rgbsB, xyzA, xyzB

    def compute_abspose(self, B, batch, cfg, poseA, rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, out_dict=None):
        b, t = poseA.shape[:2]
        if self.cfg['loss_type'] == 'l1-abs':
            rot_pred = so3_exp_map(rot).permute(0, 2, 1)
            trans_pred = trans_delta_pred
            B_in_cams = torch.zeros_like(poseA)
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
            B_in_cams[:, :, :3, 3] = trans_pred.reshape(b, t, 3)
        elif self.cfg['loss_type'] in ['l1-abs-delta', 'l2-abs-delta']:
            if self.cfg.rot_rep == 'axis_angle':
                rot_pred = so3_exp_map(out_dict['rot_abs']).permute(0, 2, 1)
            elif self.cfg.rot_rep == '6d':
                rot_pred = rot6d_to_rotmat(out_dict['rot_abs'])
            else:
                raise NotImplementedError
            trans_pred = out_dict['trans_abs']
            B_in_cams = torch.zeros_like(poseA)
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
            B_in_cams[:, :, :3, 3] = trans_pred.reshape(b, t, 3)
        elif cfg['loss_type'] in ['l1-absrot-delta', 'l1-absrot-delta-hum', 'l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
            # rot from abs, trans from delta
            B_in_cams = self.abspose_from_relative(batch, cfg, poseA, rot, trans_delta_pred)
            if self.cfg.rot_rep == 'axis_angle':
                rot_pred = so3_exp_map(out_dict['rot_abs']).permute(0, 2, 1)
            elif self.cfg.rot_rep == '6d':
                rot_pred = rot6d_to_rotmat(out_dict['rot_abs'])
            else:
                raise NotImplementedError
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
        else:
            B_in_cams = self.abspose_from_relative(batch, cfg, poseA, rot, trans_delta_pred)
        rot_delta_gt_rot = so3_exp_map(rot_delta_gt * cfg['rot_normalizer']).permute(0, 2, 1)
        B_in_cams_gt = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4),
                                                           trans_delta=trans_delta_gt * batch['mesh_diameter'].reshape((-1, 1)) / 2.,
                                                           rot_mat_delta=rot_delta_gt_rot).reshape(B, t, 4, 4)  # (BT, 4, 4)

        return B_in_cams, B_in_cams_gt

    def abspose_from_relative(self, batch, cfg, poseA, rot, trans_delta_pred):
        "compute abs pose from relative pose prediction"
        return self.abspose_from_relative_static(batch, cfg, poseA, rot, trans_delta_pred)

    def compute_smpl_verts(self, batch, out_dict):
        "compute smpl verts for GT and prediction"
        from lib_smpl import pose72to156

        betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = self.smpl_params_from_pred(batch, out_dict)

        verts_smpl, jtrs_pr, _, _, jts_rot_pr = self.smpl_male(pose72to156(pred_smpl_pose), betas, pred_smpl_t, ret_glb_rot=True)
        verts_smpl_gt, jtrs_gt, _, _, jts_rot_gt = self.smpl_male(batch['smpl_poses_gt'].reshape(-1, 156), betas,
                                                                   batch['smpl_transl_gt'].reshape(-1, 3), ret_glb_rot=True)
        return verts_smpl_gt, verts_smpl, jtrs_gt, jtrs_pr, pred_smpl_r, pred_smpl_t, jts_rot_pr, jts_rot_gt

    def smpl_params_from_pred(self, batch, out_dict):
        "compute SMPL parameters from prediction, return in shape (BT, ...)"
        J, bid = 24, 0
        clip_len = self.cfg.clip_len
        bs = len(batch['nlf_transl'])
        if self.cfg.loss_type in ['l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
            # predict abs pose already
            pred_smpl_r = out_dict['body_rotmat']
            pred_smpl_t = out_dict['body_transl']
        else:
            # additional visualization for human as well
            nlf_poses = batch['nlf_rotmat'].reshape(-1, J, 3, 3)  # B, T, J, 3, 3,
            pred_smpl_t = batch['nlf_transl'].reshape(-1, 3) + out_dict['hum_trans']
            delta_pr_r = rot6d_to_rotmat(out_dict['hum_pose'].reshape(-1, 6)).reshape(-1, J, 3, 3)

            pred_smpl_r = delta_pr_r @ nlf_poses
        pred_smpl_pose = matrix_to_axis_angle(pred_smpl_r.reshape(-1, 3, 3)).reshape(-1, J * 3)
        if 'hum_shape' in out_dict:
            betas = out_dict['hum_shape'] + batch['betas_nlf'].reshape(-1, 10)
        else:
            betas = batch['betas_nlf'].reshape(-1, 10) # use predicted NLF betas
        return betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t

    @staticmethod
    def visualize_rgbm(batch, bid, i, maskA, maskB, rgbsA, rgbsB):
        rgba = rgbsA[bid, i].transpose(1, 2, 0)
        rgbb = rgbsB[bid, i].transpose(1, 2, 0)
        ab = np.concatenate([rgba, rgbb], axis=0)
        # log mask as well
        if batch['input_xyz'].shape[2] == 5:
            maska = maskA[bid, i].transpose(1, 2, 0)  # already (H, W, 2)
            maskb = maskB[bid, i].transpose(1, 2, 0)
            comb = np.concatenate([np.concatenate([maska, np.zeros_like(maska[:, :, 0:1])], -1),
                                   np.concatenate([maskb, np.zeros_like(maskb[:, :, 0:1])], -1)], 0)
            comb = np.concatenate([ab, comb], axis=1)
        elif batch['input_xyz'].shape[2] == 4:
            # one channel
            maska_h = maskA[bid, i, 0][:, :, None].repeat(3, -1)
            maska_o = maskA[bid, i, 0][:, :, None].repeat(3, -1)
            maskb_h = maskB[bid, i, 0][:, :, None].repeat(3, -1)
            maskb_o = maskB[bid, i, 0][:, :, None].repeat(3, -1)
            comb = np.concatenate([maska_h, maskb_h], axis=0)
            comb = np.concatenate([ab, comb], axis=1)
        elif batch['input_xyz'].shape[2] == 6:
            # do nothing
            maska = maskA[bid, i].transpose(1, 2, 0) # already (H, W, 3)
            maskb = maskB[bid, i].transpose(1, 2, 0)
            comb = np.concatenate([maska, maskb], axis=0)
            comb = np.concatenate([ab, comb], axis=1)
        else:
            comb = ab
        return comb, rgba, rgbb

    @staticmethod
    def visualize_xyz_map(bid, i, xyzA, xyzB):
        mask_xyza = np.abs(xyzA[bid, i, 2]) > 0.001  # avoid all zeros
        pc_a = xyzA[bid, i].transpose(1, 2, 0)[mask_xyza].reshape((-1, 3))
        mask_xyzb = np.abs(xyzB[bid, i, 2]) > 0.001
        pc_b = xyzB[bid, i].transpose(1, 2, 0)[mask_xyzb].reshape((-1, 3))
        pc_ab = np.concatenate([pc_a, pc_b], axis=0)
        red = np.array([[255, 0, 0]]).repeat(len(pc_a), 0)
        green = np.array([[0, 255, 0]]).repeat(len(pc_b), 0)
        pc_colors = np.concatenate([red, green], 0)
        return pc_ab, pc_colors

def main2():
    # 1. Create the base config from the structured dataclass
    # This holds all the defaults.
    cfg = get_config()

    trainer = Trainer(cfg)
    trainer.train()
    trainer.close()


def get_config():
    base_conf = OmegaConf.structured(TrainTemporalRefinerConfig)
    cfg_cli = OmegaConf.from_cli()
    # 2. Load the config from the YAML file
    # This holds our overrides.
    if 'config' in cfg_cli:
        file_conf = OmegaConf.load(cfg_cli.config)
        # 3. Merge the two configurations.
        # The values in `file_conf` will overwrite the defaults in `base_conf`.
        cfg: TrainTemporalRefinerConfig = OmegaConf.merge(base_conf, file_conf)
        print("Overriding config from file", cfg_cli.config)
    else:
        cfg = base_conf
    # merge with command line args
    cfg = OmegaConf.merge(cfg, cfg_cli)
    return cfg


if __name__ == "__main__":
    main2()
