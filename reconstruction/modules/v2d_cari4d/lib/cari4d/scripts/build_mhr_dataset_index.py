from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from learning.datasets import _dataset_config_for_split, _load_dataset_sequences
from learning.datasets.mhr_dataset_index import dataset_index_is_current, dataset_source_identity, load_dataset_index, write_dataset_index
from learning.datasets.mhr_video_data import MHRVideoDataset
from learning.datasets.mhr_window_sampling import apply_mhr_window_sampling_contract, load_mhr_window_sampling_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable MHR dataset metadata indexes before distributed training starts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--window-sampling-contract", type=Path)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


def _scan_sequence_chunk(cfg: dict, seqs: list[str], split: str, chunk_index: int) -> tuple[int, list[dict], dict]:
    dataset = MHRVideoDataset(cfg, seqs, split)
    try:
        return chunk_index, dataset.sequence_data, dataset.render_metadata
    finally:
        dataset.close()


def _scan_dataset_metadata(split_cfg, seqs: list[str], split: str, workers: int) -> SimpleNamespace:
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    worker_count = min(workers, len(seqs))
    chunk_size = max(1, (len(seqs) + worker_count * 8 - 1) // (worker_count * 8))
    chunks = [(index, seqs[start:start + chunk_size]) for index, start in enumerate(range(0, len(seqs), chunk_size))]
    cfg_dict = OmegaConf.to_container(split_cfg, resolve=True)
    chunk_results: dict[int, tuple[list[dict], dict]] = {}
    completed_sequences = 0
    report_interval = max(1, len(seqs) // 20)
    next_report = report_interval
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_scan_sequence_chunk, cfg_dict, chunk_seqs, split, chunk_index): len(chunk_seqs) for chunk_index, chunk_seqs in chunks}
        for future in as_completed(futures):
            chunk_index, sequence_data, render_metadata = future.result()
            chunk_results[chunk_index] = (sequence_data, render_metadata)
            completed_sequences += futures[future]
            if completed_sequences >= next_report or completed_sequences == len(seqs):
                print(f"MHR_DATASET_INDEX_SCAN_PROGRESS split={split} sequences={completed_sequences}/{len(seqs)}", flush=True)
                while next_report <= completed_sequences:
                    next_report += report_interval
    all_sequence_data: list[dict] = []
    all_render_metadata: dict = {}
    for chunk_index in range(len(chunks)):
        sequence_data, render_metadata = chunk_results[chunk_index]
        all_sequence_data.extend(sequence_data)
        all_render_metadata.update(render_metadata)
    if any(len(data["sample_starts"]) != len(data["sample_strides"]) for data in all_sequence_data):
        raise ValueError("MHR sample starts and temporal strides must have identical lengths")
    sample_counts = [len(data["sample_starts"]) * len(data["sample_kids"]) for data in all_sequence_data]
    sample_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sample_counts, dtype=np.int64)))
    return SimpleNamespace(sequence_data=all_sequence_data, render_metadata=all_render_metadata, sample_offsets=sample_offsets, _sample_count=int(sample_offsets[-1]))


def build_split(cfg, seqs: list[str], split: str, output: Path, workers: int) -> None:
    split_cfg = _dataset_config_for_split(cfg, split)
    split_cfg.mhr_dataset_index_path = None
    started = time.perf_counter()
    print(f"MHR_DATASET_INDEX_SOURCE_SCAN_BEGIN split={split} sequences={len(seqs)}", flush=True)
    source_identity = dataset_source_identity(split_cfg, seqs)
    print(f"MHR_DATASET_INDEX_SOURCE_SCAN_COMPLETE split={split} files={source_identity['file_count']} elapsed_seconds={time.perf_counter() - started:.3f}", flush=True)
    if dataset_index_is_current(output, split_cfg, seqs, split, source_identity):
        payload = load_dataset_index(output, split_cfg, seqs, split, source_identity)
        elapsed = time.perf_counter() - started
        print(f"MHR_DATASET_INDEX_REUSED split={split} sequences={len(seqs)} samples={payload['sample_count']} files={source_identity['file_count']} bytes={output.stat().st_size} elapsed_seconds={elapsed:.3f} path={output}", flush=True)
        return
    dataset = _scan_dataset_metadata(split_cfg, seqs, split, workers)
    payload = write_dataset_index(output, dataset, split_cfg, seqs, split, source_identity)
    elapsed = time.perf_counter() - started
    print(f"MHR_DATASET_INDEX_BUILT split={split} sequences={len(seqs)} samples={payload['sample_count']} files={source_identity['file_count']} bytes={output.stat().st_size} elapsed_seconds={elapsed:.3f} path={output}", flush=True)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.window_sampling_contract is not None:
        apply_mhr_window_sampling_contract(cfg, load_mhr_window_sampling_contract(args.window_sampling_contract))
    if str(cfg.body_model) != "mhr" or str(cfg.packed_format) != "h5" or str(cfg.val_packed_format) != "h5":
        raise ValueError("MHR dataset indexes require body_model=mhr and H5 train/validation inputs")
    train_seqs, val_seqs = _load_dataset_sequences(cfg)
    build_split(cfg, train_seqs, "train", args.train_output, args.workers)
    build_split(cfg, val_seqs, "val", args.val_output, args.workers)


if __name__ == "__main__":
    main()
