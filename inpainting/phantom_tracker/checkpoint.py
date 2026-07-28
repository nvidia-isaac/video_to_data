"""Checkpoint loading helpers kept independent of heavyweight ML imports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def load_state_dict_strict(model: Any, state_dict: Any) -> dict[str, list[str]]:
    """Load an exact state dict and reject every compatibility discrepancy."""

    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Official HaMeR checkpoint state_dict is not a mapping")
    try:
        incompatibility = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Official HaMeR checkpoint is incompatible with the pinned model"
        ) from error
    missing = sorted(getattr(incompatibility, "missing_keys", ()))
    unexpected = sorted(getattr(incompatibility, "unexpected_keys", ()))
    # PyTorch's strict=True already raises for these. Keep the explicit check so
    # a backend that violates that contract cannot silently weaken this guard.
    if missing or unexpected:
        raise RuntimeError(
            "Strict HaMeR checkpoint load reported incompatible keys: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {"missing_keys": missing, "unexpected_keys": unexpected}
