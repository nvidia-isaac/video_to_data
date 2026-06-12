# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2P Whole Body base task configurations."""

from .base_env_cfg import (  # noqa: F401
    BaseCommandsCfg,
    BaseEventsCfg,
    V2PEnvCfg,
    V2PSceneCfg,
)
from .config.sonic import (  # noqa: F401
    G1_SONIC_JOINT_NAMES,
    G1SonicEnvCfg,
    G1SonicReconBodyEnvCfg,
    G1SonicReconHandEnvCfg,
)
