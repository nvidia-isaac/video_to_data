# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package-boundary and upstream source-attribution tests for native FlashSAC."""

import subprocess
import sys
from importlib import resources


def test_package_records_pinned_upstream_source():
    from flash_chord.training import flash_sac

    assert flash_sac.UPSTREAM_REPOSITORY == "https://github.com/Holiday-Robot/FlashSAC"
    assert flash_sac.UPSTREAM_COMMIT == "87edc9061150ae9e962dd84e6544e27a1554b3ab"


def test_package_contains_upstream_mit_notice():
    notice = resources.files("flash_chord.training.flash_sac").joinpath("NOTICE").read_text()

    assert "commit 87edc9061150ae9e962dd84e6544e27a1554b3ab" in notice
    assert "MIT License" in notice
    assert "Copyright (c) 2026 Holiday Robotics" in notice


def test_package_import_does_not_eagerly_load_learner_frameworks():
    source = """
import sys
import flash_chord.training.flash_sac
assert "jax" not in sys.modules
assert "flax" not in sys.modules
assert "optax" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], check=True)
