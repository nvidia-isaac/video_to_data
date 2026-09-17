import random
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


STEP_CKPT_RE = re.compile(r"^step(\d+)\.pth$")


def checkpoint_step(path) -> Optional[int]:
    match = STEP_CKPT_RE.match(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def find_latest_step_checkpoint(exp_dir) -> Optional[Path]:
    candidates = []
    for path in Path(exp_dir).glob("step*.pth"):
        step = checkpoint_step(path)
        if step is not None:
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def prune_step_checkpoints(exp_dir, keep: int):
    keep = int(keep)
    if keep <= 0:
        return []
    checkpoints = []
    for path in Path(exp_dir).glob("step*.pth"):
        step = checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))
    checkpoints.sort()
    removed = []
    for _, path in checkpoints[:-keep]:
        sidecars = sorted(path.parent.glob(f"{path.stem}.rank*.rng"))
        certificate = path.with_name(f"{path.stem}.recovery.json")
        path.unlink()
        removed.append(path)
        for sidecar in sidecars:
            sidecar.unlink()
            removed.append(sidecar)
        if certificate.is_file():
            certificate.unlink()
            removed.append(certificate)
    return removed


def normalize_epoch_step(saved_epoch: int, saved_step: int, saved_epoch_step: Optional[int], dataloader_len: int) -> Tuple[int, int]:
    if saved_epoch_step is not None and int(saved_epoch_step) >= 0:
        return int(saved_epoch), int(saved_epoch_step)
    if int(dataloader_len) <= 0:
        raise ValueError(f"dataloader_len must be positive, got {dataloader_len}")
    return int(saved_epoch), int(saved_step) % int(dataloader_len)


def rank_rng_state_path(ckpt_file, rank: int) -> Path:
    path = Path(ckpt_file)
    return path.with_name(f"{path.stem}.rank{int(rank):02d}.rng")


def capture_rng_state():
    import torch

    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state) -> None:
    if state is None:
        return
    import torch

    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def load_rank_rng_state(ckpt_file, rank: int, checkpoint_dict=None):
    import torch

    rng_file = rank_rng_state_path(ckpt_file, rank)
    if rng_file.is_file():
        return torch.load(rng_file, map_location="cpu", weights_only=False)
    if checkpoint_dict is not None:
        return checkpoint_dict.get("rng_state")
    return None


def set_dataloader_epoch_seed(dataloader, seed, epoch: int) -> None:
    if seed is None:
        return
    import torch

    generator = torch.Generator()
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(int(epoch))
    generator.manual_seed(int(seed) + int(epoch))
    targets = [dataloader]
    for attr in ("base_dataloader", "dataloader"):
        target = getattr(dataloader, attr, None)
        if target is not None and target not in targets:
            targets.append(target)
    for target in targets:
        if hasattr(target, "generator"):
            target.generator = generator
        sampler = getattr(target, "sampler", None)
        if sampler is not None and hasattr(sampler, "generator"):
            sampler.generator = generator
        batch_sampler = getattr(target, "batch_sampler", None)
        nested_sampler = getattr(batch_sampler, "sampler", None)
        if nested_sampler is not None and hasattr(nested_sampler, "generator"):
            nested_sampler.generator = generator


def resume_dataloader_iterator(accelerator, dataloader, start_epoch_step: int):
    start_epoch_step = int(start_epoch_step)
    if start_epoch_step < 0:
        raise ValueError(f"start_epoch_step must be nonnegative, got {start_epoch_step}")
    if start_epoch_step == 0:
        return iter(dataloader), "none"
    if bool(getattr(dataloader, "supports_sampler_level_resume_skip", False)):
        base_dataloader = getattr(dataloader, "base_dataloader", None)
        iterator_factory = getattr(dataloader, "iter_from_base_dataloader", None)
        if base_dataloader is None or not callable(iterator_factory):
            raise RuntimeError("Sampler-level resume skipping requires a base dataloader and iter_from_base_dataloader")
        skipped_dataloader = accelerator.skip_first_batches(base_dataloader, start_epoch_step)
        return iterator_factory(skipped_dataloader), "sampler"
    dataloader_iter = iter(dataloader)
    for _ in zip(range(start_epoch_step), dataloader_iter):
        pass
    return dataloader_iter, "replay"
