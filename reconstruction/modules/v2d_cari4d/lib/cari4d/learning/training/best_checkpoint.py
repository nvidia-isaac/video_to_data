import json
import os
from pathlib import Path


BEST_VALIDATION_FILENAME = "best_validation.json"
BEST_VALIDATION_METRIC = "val/loss_val"
BEST_MODEL_FILENAME = "model_best.pth"


def update_best_validation(train_state, loss_val: float, step: int) -> bool:
    loss_val = float(loss_val)
    if train_state.best_val is not None and loss_val >= float(train_state.best_val):
        return False
    train_state.best_val = loss_val
    train_state.best_step = int(step)
    return True


def best_validation_metadata(cfg, train_state):
    if train_state.best_val is None or train_state.best_step is None:
        raise ValueError("Best-validation metadata requires best_val and best_step")
    return {
        "metric_name": BEST_VALIDATION_METRIC,
        "best_metric": float(train_state.best_val),
        "best_step": int(train_state.best_step),
        "greater_is_better": False,
        "run_name": str(cfg.exp_name),
        "wandb_run_id": None if getattr(cfg, "run_id", None) is None else str(cfg.run_id),
        "source_checkpoint_step": int(train_state.best_step),
    }


def save_best_validation_metadata(exp_dir, metadata) -> Path:
    path = Path(exp_dir) / BEST_VALIDATION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)
    return path


def load_best_validation_metadata(exp_dir):
    path = Path(exp_dir) / BEST_VALIDATION_FILENAME
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text())
    required = {"metric_name", "best_metric", "best_step", "greater_is_better", "run_name", "wandb_run_id", "source_checkpoint_step"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Best-validation metadata is missing keys: {missing}")
    if metadata["metric_name"] != BEST_VALIDATION_METRIC or metadata["greater_is_better"] is not False:
        raise ValueError(f"Unsupported best-validation policy: {metadata}")
    return metadata


def apply_best_validation_metadata(train_state, metadata, minimum_step=None) -> bool:
    if metadata is None:
        return False
    best_metric = float(metadata["best_metric"])
    best_step = int(metadata["best_step"])
    force_new_semantics = minimum_step is not None and best_step >= int(minimum_step)
    if not force_new_semantics and train_state.best_val is not None and best_metric >= float(train_state.best_val):
        return False
    train_state.best_val = best_metric
    train_state.best_step = best_step
    return True


def archive_best_validation_artifacts(exp_dir, reset_step: int):
    exp_dir = Path(exp_dir)
    reset_step = int(reset_step)
    if reset_step < 1:
        raise ValueError(f"Best-validation semantics reset step must be positive, got {reset_step}")
    archived = []
    for source_name in (BEST_MODEL_FILENAME, BEST_VALIDATION_FILENAME):
        source = exp_dir / source_name
        stem, suffix = source.stem, source.suffix
        destination = exp_dir / f"{stem}.before_validation_reset_step{reset_step:06d}{suffix}"
        if destination.exists():
            if source.exists():
                raise FileExistsError(f"Refusing to overwrite existing validation archive while current artifact also exists: {source} and {destination}")
        elif source.exists():
            os.replace(source, destination)
        archived.append(destination)
    return archived


def reset_best_validation_for_semantics(train_state, reset_step) -> bool:
    if reset_step is None:
        return False
    reset_step = int(reset_step)
    if reset_step < 1:
        raise ValueError(f"Best-validation semantics reset step must be positive, got {reset_step}")
    current_step = int(train_state.step)
    if current_step < reset_step:
        return False
    if train_state.best_step is not None and int(train_state.best_step) >= reset_step:
        return False
    if train_state.pending_validation_step is not None and int(train_state.pending_validation_step) != current_step:
        raise ValueError(f"Cannot schedule validation semantics reset at step {current_step} with pending validation at step {train_state.pending_validation_step}")
    train_state.best_val = None
    train_state.best_step = None
    train_state.pending_validation_step = current_step
    return True
