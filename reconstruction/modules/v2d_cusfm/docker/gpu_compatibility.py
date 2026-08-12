# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side GPU compatibility preflight for the CuSFM container."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


_UNSUPPORTED_COMPUTE_CAPABILITY = (12, 0)
_NVIDIA_SMI_COMMAND = [
    "nvidia-smi",
    "--query-gpu=index,compute_cap,name",
    "--format=csv,noheader,nounits",
]


class GPUCompatibilityError(RuntimeError):
    """Raised when the host GPU cannot run the v0.2 CuSFM container."""


@dataclass(frozen=True)
class GPUInfo:
    """GPU identity and CUDA compute capability reported by ``nvidia-smi``."""

    index: int
    name: str
    compute_capability: tuple[int, int]

    @property
    def compute_capability_text(self) -> str:
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    @property
    def sm(self) -> str:
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"


def _parse_compute_capability(value: str, *, line_number: int) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", value.strip())
    if match is None:
        raise GPUCompatibilityError(
            "GPU compatibility preflight could not parse compute capability "
            f"{value!r} on nvidia-smi output line {line_number}."
        )
    return int(match.group(1)), int(match.group(2))


def parse_nvidia_smi_output(output: str) -> list[GPUInfo]:
    """Parse ``index,compute_cap,name`` CSV output from ``nvidia-smi``."""
    gpus = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", maxsplit=2)]
        if len(fields) != 3:
            raise GPUCompatibilityError(
                "GPU compatibility preflight expected three fields on nvidia-smi "
                f"output line {line_number}, got: {raw_line!r}."
            )
        index_text, compute_capability_text, name = fields
        try:
            index = int(index_text)
        except ValueError as exc:
            raise GPUCompatibilityError(
                "GPU compatibility preflight could not parse GPU index "
                f"{index_text!r} on nvidia-smi output line {line_number}."
            ) from exc
        if not name:
            raise GPUCompatibilityError(
                "GPU compatibility preflight received an empty GPU name on "
                f"nvidia-smi output line {line_number}."
            )
        gpus.append(
            GPUInfo(
                index=index,
                name=name,
                compute_capability=_parse_compute_capability(
                    compute_capability_text,
                    line_number=line_number,
                ),
            )
        )

    if not gpus:
        raise GPUCompatibilityError(
            "GPU compatibility preflight found no NVIDIA GPUs. The v2d_cusfm container "
            "requires a supported NVIDIA GPU and a working NVIDIA driver."
        )
    return gpus


def query_nvidia_gpus() -> list[GPUInfo]:
    """Return GPUs visible to the host through ``nvidia-smi``."""
    try:
        result = subprocess.run(
            _NVIDIA_SMI_COMMAND,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GPUCompatibilityError(
            "GPU compatibility preflight could not find nvidia-smi. Install a supported "
            "NVIDIA driver before building or running the v2d_cusfm container."
        ) from exc
    except OSError as exc:
        raise GPUCompatibilityError(
            f"GPU compatibility preflight could not execute nvidia-smi: {exc}"
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        detail_suffix = f": {detail}" if detail else ""
        raise GPUCompatibilityError(
            "GPU compatibility preflight could not query the NVIDIA driver"
            f" (nvidia-smi exit {result.returncode}){detail_suffix}"
        )
    return parse_nvidia_smi_output(result.stdout)


def validate_cusfm_gpu_compatibility(
    gpus: Iterable[GPUInfo],
    selected_gpu_ids: Sequence[int] | None = None,
) -> list[GPUInfo]:
    """Validate GPUs used by the v0.2 CuSFM container."""
    available = list(gpus)
    if not available:
        raise GPUCompatibilityError(
            "GPU compatibility preflight found no NVIDIA GPUs. The v2d_cusfm container "
            "requires a supported NVIDIA GPU."
        )

    if selected_gpu_ids is None:
        selected = available
    else:
        requested_ids = list(dict.fromkeys(selected_gpu_ids))
        if not requested_ids:
            raise GPUCompatibilityError(
                "GPU compatibility preflight received an empty GPU selection."
            )
        by_index = {gpu.index: gpu for gpu in available}
        missing_ids = [gpu_id for gpu_id in requested_ids if gpu_id not in by_index]
        if missing_ids:
            missing = ", ".join(str(gpu_id) for gpu_id in missing_ids)
            visible = ", ".join(str(gpu.index) for gpu in available)
            raise GPUCompatibilityError(
                f"Requested GPU ID(s) {missing} are not visible to nvidia-smi; "
                f"visible GPU ID(s): {visible}."
            )
        selected = [by_index[gpu_id] for gpu_id in requested_ids]

    unsupported = [
        gpu
        for gpu in selected
        if gpu.compute_capability >= _UNSUPPORTED_COMPUTE_CAPABILITY
    ]
    if unsupported:
        detected = "\n".join(
            f"  - GPU {gpu.index}: {gpu.name} "
            f"(compute capability {gpu.compute_capability_text} / {gpu.sm})"
            for gpu in unsupported
        )
        raise GPUCompatibilityError(
            "Unsupported GPU architecture detected:\n"
            f"{detected}\n"
            "The v0.2 v2d_cusfm image contains TensorRT and cuVSLAM versions "
            "that do not support SM 120 or newer. Use a validated RTX A6000 "
            "(SM 86) or L40S (SM 89), or update and validate the v2d_cusfm "
            "dependency stack for the detected architecture."
        )
    return selected


def require_compatible_cusfm_gpus(
    selected_gpu_ids: Sequence[int] | None = None,
) -> list[GPUInfo]:
    """Query and validate GPUs, printing the accepted devices on success."""
    selected = validate_cusfm_gpu_compatibility(
        query_nvidia_gpus(),
        selected_gpu_ids=selected_gpu_ids,
    )
    summary = ", ".join(
        f"GPU {gpu.index}: {gpu.name} ({gpu.sm})" for gpu in selected
    )
    print(f"[gpu preflight] Compatible GPU(s): {summary}")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate GPUs for the v0.2 v2d_cusfm container."
    )
    parser.add_argument(
        "--gpu-id",
        action="append",
        type=int,
        dest="gpu_ids",
        help="GPU index to validate; repeat for multiple GPUs (default: all visible GPUs)",
    )
    args = parser.parse_args(argv)
    try:
        require_compatible_cusfm_gpus(args.gpu_ids)
    except GPUCompatibilityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
