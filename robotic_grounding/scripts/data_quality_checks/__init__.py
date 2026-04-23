# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Auto-discovery for data quality checks.

Each ``.py`` file in this directory that defines a ``check()`` function is
automatically registered as a quality check.  The file name (without extension)
becomes the check name.

Check protocol::

    def check(data: dict, **kwargs) -> dict:
        '''Evaluate one sequence.

Args:
            data: Parquet row as a dict (from ``pq.read_table().to_pydict()``).
                  Each value is a list with one element per row.
            **kwargs: Overridable parameters (e.g. ``threshold``).

Returns:
            {"pass": bool, "score": float, "reason": str}
        '''
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable

CheckFn = Callable[..., dict[str, Any]]

_CHECKS_DIR = Path(__file__).resolve().parent


def discover_checks() -> dict[str, CheckFn]:
    """Scan this directory for modules that expose a ``check()`` function.

    Returns:
        Dict mapping check name to the callable ``check`` function.
    """
    checks: dict[str, CheckFn] = {}
    for py_file in sorted(_CHECKS_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(
            f"data_quality_checks.{module_name}", str(py_file)
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue
        fn = getattr(module, "check", None)
        if callable(fn):
            checks[module_name] = fn
    return checks


def get_check_description(check_fn: CheckFn) -> str:
    """Return the first line of the check module's docstring, or ''."""
    doc = getattr(check_fn, "__doc__", "") or ""
    # Prefer the module-level docstring if available
    if hasattr(check_fn, "__globals__") and "__doc__" in check_fn.__globals__:
        module_doc = check_fn.__globals__["__doc__"]
        if module_doc:
            doc = module_doc
    return doc.strip().split("\n")[0] if doc else ""
