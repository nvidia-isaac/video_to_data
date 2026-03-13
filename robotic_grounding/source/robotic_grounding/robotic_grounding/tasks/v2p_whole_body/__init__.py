"""V2P Whole Body base task configurations."""

from .base_ee_env_cfg import (  # noqa: F401
    EEObservationsCfg,
    EERewardsCfg,
    EETerminationsCfg,
    V2PEEEnvCfg,
)
from .base_env_cfg import (  # noqa: F401
    BaseCommandsCfg,
    BaseCurriculumsCfg,
    BaseEventsCfg,
    BaseObservationsCfg,
    BaseRewardsCfg,
    BaseTerminationsCfg,
    V2PEnvCfg,
    V2PSceneCfg,
)
from .config.sonic import (  # noqa: F401
    G1_SONIC_JOINT_NAMES,
    G1SonicEEEnvCfg,
    G1SonicEnvCfg,
    SonicEEEnvCfg,
    SonicEnvCfg,
)
