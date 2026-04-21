"""OSMO workflow generator for example_param_sweep.

Sweeps:
  contact_force.weight  : values in osmo_multi_task.contact_force_weights
  action_l1.weight      : values in osmo_multi_task.action_l1_weights

Total tasks = len(contact_force_weights) × len(action_l1_weights).
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


def _fmt(val: str | float) -> str:
    """Format a value into a run-name-safe string (e.g. -5e-3 -> 'n5e-3', 1.0 -> '1p0')."""
    s = str(val)
    s = s.replace("-", "n", 1) if s.startswith("-") else s
    s = s.replace(".", "p")
    return s


def get_variant_overrides(variant_name: str, config: dict) -> dict:
    """Return override dict for --local --variant <name>.

    variant_name format: "cf<cf_val>_al<al_val>" e.g. "cf1p0_aln5e-3"
    """
    mt = config["osmo_multi_task"]
    for cf in mt["contact_force_weights"]:
        for al in mt["action_l1_weights"]:
            if variant_name == f"cf{_fmt(cf)}_al{_fmt(al)}":
                overrides = dict(config["train_overrides"])
                overrides["env.rewards.contact_force_reward.weight"] = str(cf)
                overrides["env.rewards.action_l1.weight"] = str(al)
                return overrides
    available = [
        f"cf{_fmt(cf)}_al{_fmt(al)}"
        for cf in mt["contact_force_weights"]
        for al in mt["action_l1_weights"]
    ]
    raise ValueError(f"Unknown variant '{variant_name}'. Available: {available}")


def generate_workflow(exp_id: str, config: dict) -> str:
    """Generate OSMO multi-task workflow YAML for the parameter sweep."""
    mt = config["osmo_multi_task"]
    base_overrides = dict(config["train_overrides"])
    motion_file = config.get("motion_file")
    video = config.get("video", False)

    cf_weights = mt["contact_force_weights"]
    al_weights = mt["action_l1_weights"]

    tasks_yaml = []
    for cf in cf_weights:
        for al in al_weights:
            overrides = {
                **base_overrides,
                "env.rewards.contact_force_reward.weight": str(cf),
                "env.rewards.action_l1.weight": str(al),
            }
            tag = f"cf{_fmt(cf)}_al{_fmt(al)}"
            run_suffix = f"osmo_{exp_id}_{tag}"
            entry = make_entry_script(
                run_suffix,
                overrides,
                motion_file=motion_file,
                video=video,
                use_timestamp=True,
            )
            entry_indent = "\n".join("        " + line for line in entry.split("\n"))
            tasks_yaml.append(
                f"""  - name: train-{tag}
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
# Generated from experiments/example_param_sweep/workflow.py

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
