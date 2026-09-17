# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from learning.training.training_config import TrainTemporalRefinerConfig


def _dataloader_worker_kwargs(num_workers: int, pin_memory: bool):
    kwargs = {"num_workers": int(num_workers), "pin_memory": pin_memory}
    if int(num_workers) > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return kwargs


def _pad_symmetry_array(value, target_count, axis, label):
    value = np.asarray(value)
    if value.shape[axis] < 1:
        raise ValueError(f"{label} must contain at least one valid symmetry")
    if value.shape[axis] == target_count:
        return value
    first = np.take(value, [0], axis=axis)
    return np.concatenate([value, np.repeat(first, target_count - value.shape[axis], axis=axis)], axis=axis)


def mhr_collate(samples):
    from torch.utils.data import default_collate
    from learning.datasets.mhr_input_materialization import MHR_NEUTRAL_HEIGHT_KEY
    from learning.datasets.mhr_rank_local import MHR_RENDER_REQUEST_KEY

    request_presence = [MHR_RENDER_REQUEST_KEY in sample for sample in samples]
    if any(request_presence) and not all(request_presence):
        raise ValueError("MHR batch cannot mix deferred and materialized render samples")
    render_requests = None
    if all(request_presence):
        render_requests = [sample[MHR_RENDER_REQUEST_KEY] for sample in samples]
        samples = [{key: value for key, value in sample.items() if key != MHR_RENDER_REQUEST_KEY} for sample in samples]

    neutral_height_presence = [MHR_NEUTRAL_HEIGHT_KEY in sample for sample in samples]
    if any(neutral_height_presence) and not all(neutral_height_presence):
        samples = [{key: value for key, value in sample.items() if key != MHR_NEUTRAL_HEIGHT_KEY} for sample in samples]

    symmetry_counts = []
    for index, sample in enumerate(samples):
        counts = []
        if "obj_symmetry_tfs" in sample:
            counts.append(np.asarray(sample["obj_symmetry_tfs"]).shape[0])
        if "pose_gt_symm" in sample:
            counts.append(np.asarray(sample["pose_gt_symm"]).shape[1])
        if len(set(counts)) > 1:
            raise ValueError(f"MHR sample {index} has inconsistent symmetry counts: {counts}")
        symmetry_counts.extend(counts[:1])
    if symmetry_counts:
        target_count = max(symmetry_counts)
        padded = []
        for sample in samples:
            item = dict(sample)
            if "obj_symmetry_tfs" in item:
                item["obj_symmetry_tfs"] = _pad_symmetry_array(item["obj_symmetry_tfs"], target_count, 0, "obj_symmetry_tfs")
            if "pose_gt_symm" in item:
                item["pose_gt_symm"] = _pad_symmetry_array(item["pose_gt_symm"], target_count, 1, "pose_gt_symm")
            padded.append(item)
        samples = padded
    batch = default_collate(samples)
    if render_requests is not None:
        batch[MHR_RENDER_REQUEST_KEY] = render_requests
    return batch


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_set(cfg, key, value):
    if isinstance(cfg, Mapping):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _validation_sequences(split):
    if "val" in split:
        return split["val"]
    if "test" in split:
        return split["test"]
    raise KeyError("validation split file must contain a 'val' or 'test' list")


def _load_dataset_sequences(cfg):
    train_split = _load_json(cfg.split_file)
    if "train" not in train_split:
        raise KeyError("training split file must contain a 'train' list")
    val_split_file = _cfg_get(cfg, "val_split_file") or cfg.split_file
    val_split = train_split if val_split_file == cfg.split_file else _load_json(val_split_file)
    return list(train_split["train"]), list(_validation_sequences(val_split))


def _dataset_config_for_split(cfg, split):
    if split not in {"train", "val"}:
        raise ValueError(f"dataset split must be train or val, got {split!r}")
    split_cfg = copy.deepcopy(cfg)
    if split == "train":
        return split_cfg
    _cfg_set(split_cfg, "mhr_interaction_trim_root", _cfg_get(cfg, "val_mhr_interaction_trim_root"))
    _cfg_set(split_cfg, "mhr_effective_mask_root", _cfg_get(cfg, "val_mhr_effective_mask_root"))
    overrides = {
        "val_packed_root": "packed_root",
        "val_packed_format": "packed_format",
        "val_render_root": "render_root",
        "val_mhr_gt_root": "mhr_gt_root",
        "val_mhr_init_root": "mhr_init_root",
        "val_mhr_pseudogt_root": "mhr_pseudogt_root",
        "val_mhr_sample_kids": "mhr_sample_kids",
        "val_mhr_dataset_index_path": "mhr_dataset_index_path",
        "val_mhr_require_foundationpose_training_tier_schema": "mhr_require_foundationpose_training_tier_schema",
        "val_mhr_required_contact_revision": "mhr_required_contact_revision",
        "val_mhr_rank_local_materialization_mode": "mhr_rank_local_materialization_mode",
        "val_clip_len": "clip_len",
        "val_window": "window",
    }
    for source, target in overrides.items():
        value = _cfg_get(cfg, source)
        if value is not None:
            _cfg_set(split_cfg, target, value)
    return split_cfg


def get_dataset(cfg: "TrainTemporalRefinerConfig"):
    ""
    import torch
    from torch.utils.data import DataLoader

    seqs_train, seqs_test = _load_dataset_sequences(cfg)
    train_cfg = _dataset_config_for_split(cfg, "train")
    val_cfg = _dataset_config_for_split(cfg, "val")

    if getattr(cfg, "body_model", "smpl") == "mhr":
        from learning.datasets.mhr_video_data import MHRVideoDataset

        data_class = MHRVideoDataset
    elif cfg.data_name == 'video-data':
        from learning.datasets.video_data import VideoDataset

        data_class = VideoDataset
    elif cfg.data_name == 'behave-fullseq':
        from learning.datasets.behave_fullseq import BehaveFullSeqTestDataset

        data_class = BehaveFullSeqTestDataset # this is for testing
    elif cfg.data_name == 'test-only':
        from learning.datasets.video_data import VideoDataProcessor

        data_class = VideoDataProcessor
    else:
        raise ValueError('Unknown dataset: {}'.format(cfg.data_name))
    dataset_train, dataset_test = data_class(train_cfg, seqs_train, 'train'), data_class(val_cfg, seqs_test, 'val')

    train_shuffle = cfg.job == 'train'
    print(f"In total {len(dataset_train)} training and {len(dataset_test)} test samples, train shuffle? {train_shuffle}, val shuffle? False")
    train_generator, val_generator = None, None
    if getattr(cfg, "seed", None) is not None:
        train_generator = torch.Generator()
        val_generator = torch.Generator()
        train_generator.manual_seed(int(cfg.seed))
        val_generator.manual_seed(int(cfg.seed) + 1000003)
    train_workers = int(cfg.num_workers)
    val_workers_cfg = _cfg_get(cfg, "val_num_workers")
    val_batch_size_cfg = _cfg_get(cfg, "val_batch_size")
    val_workers = max(0, train_workers // 2) if val_workers_cfg is None else int(val_workers_cfg)
    val_batch_size = int(cfg.batch_size) if val_batch_size_cfg is None else int(val_batch_size_cfg)
    pin_memory = torch.cuda.is_available()
    collate_fn = mhr_collate if getattr(cfg, "body_model", "smpl") == "mhr" else None
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size,
                                shuffle=train_shuffle, generator=train_generator, collate_fn=collate_fn, **_dataloader_worker_kwargs(train_workers, pin_memory))
    dataloader_test = DataLoader(dataset_test, batch_size=val_batch_size,
                                shuffle=False, generator=val_generator, collate_fn=collate_fn, **_dataloader_worker_kwargs(val_workers, pin_memory))

    return dataloader_train, dataloader_test, dataset_test, dataset_train
