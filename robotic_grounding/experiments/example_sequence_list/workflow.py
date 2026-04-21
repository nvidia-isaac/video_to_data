"""OSMO workflow generator for example_sequence_list.

Spawns one training task per sequence in osmo_multi_task.sequence_ids.
Each task trains independently on its assigned sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from experiments.utils import (  # noqa: E402
    DEFAULT_OSMO_IMAGE_LATEST,
    make_entry_script,
)


def _sequence_to_short_name(sequence_id: str) -> str:
    """e.g. arctic_s01_capsulemachine_grab_01 -> capsulemachine."""
    parts = sequence_id.split("_")
    if len(parts) >= 3:
        return parts[2]
    return sequence_id.replace("_", "-")[:20]


def get_variant_overrides(variant_name: str, config: dict) -> dict:
    """Return overrides for --local --variant <short_name>.

    variant_name is the short sequence name, e.g. "capsulemachine".
    Passes motion_file so the local run trains on that sequence only.
    """
    mt = config["osmo_multi_task"]
    for seq_id in mt["sequence_ids"]:
        if _sequence_to_short_name(seq_id) == variant_name:
            return {"motion_file": f"arctic_processed/{seq_id}/sharpa_wave"}
    available = [_sequence_to_short_name(s) for s in mt["sequence_ids"]]
    raise ValueError(f"Unknown variant '{variant_name}'. Available: {available}")


def generate_workflow(exp_id: str, config: dict) -> str:
    """Generate OSMO multi-task workflow YAML: one task per sequence_id."""
    mt = config["osmo_multi_task"]
    sequence_ids = mt["sequence_ids"]
    base_overrides = dict(config["train_overrides"])
    template = mt.get("run_name_suffix_template", f"{exp_id}_{{sequence_id}}")

    tasks_yaml = []
    for seq_id in sequence_ids:
        short_name = _sequence_to_short_name(seq_id)
        run_suffix = template.format(sequence_id=seq_id, short_name=short_name)
        entry = make_entry_script(
            run_suffix,
            base_overrides,
            motion_file=f"arctic_processed/{seq_id}/sharpa_wave",
            video=config.get("video", False),
            use_timestamp=True,
        )
        entry_indent = "\n".join("        " + line for line in entry.split("\n"))
        tasks_yaml.append(
            f"""  - name: train-{short_name}
    image: {{{{image}}}}
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    environment:
      ACCEPT_EULA: Y
      OMNI_SERVER: omniverse://isaac-dev.ov.nvidia.com
    files:
    - path: /tmp/entry.sh
      contents: |-
{entry_indent}"""
        )

    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generated from experiments/example_sequence_list/workflow.py

workflow:
  name: {{{{workflow_name}}}}
  resources:
    default:
      cpu: 6
      gpu: 1
      memory: 120Gi
      storage: 200Gi

  tasks:
{chr(10).join(tasks_yaml)}

default-values:
  workflow_name: robotic_grounding_{exp_id}
  image: {DEFAULT_OSMO_IMAGE_LATEST}
"""
