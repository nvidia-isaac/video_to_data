from dataclasses import MISSING
from enum import Enum

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass


class SONICActionType(Enum):
    """Types of SONIC action terms."""

    HIERARCHICAL = "hierarchical"
    RESIDUAL = "residual"
    LATENT_RESIDUAL = "latent_residual"
    LATENT = "latent"
    LATENT_HAND_POLICY = "latent_hand_policy"


@configclass
class SONICActionCfg(ActionTermCfg):
    """Common configuration for all SONIC action terms.

    Use action_type to select which SONIC action implementation to use.
    """

    action_type: SONICActionType = SONICActionType.HIERARCHICAL
    """Type of SONIC action term to use."""

    policy_dir: str = MISSING  # type: ignore[assignment]
    """Path to directory containing SONIC ONNX models."""

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    joint_names: list[str] = [".*"]
    """List of joint names or regex patterns for all controllable joints."""

    sonic_joint_names: list[str] = MISSING  # type: ignore[assignment]
    """List of joint names or regex patterns for joints controlled by SONIC."""

    command_name: str = "motion"
    """Name of the tracking command term (used for hierarchical action)."""

    use_default_offset: bool = True
    """Whether SONIC outputs offsets from default position."""

    scale: float | dict[str, float] = 1.0
    """Scale to apply to SONIC outputs."""

    # Hand policy configuration (used by LATENT_HAND_POLICY action type)
    hand_policy_class: type | None = None
    """Class of the pretrained hand policy to use for direct joint (non-SONIC) control."""

    hand_policy_cfg: object = None
    """Configuration for the hand policy. Direct joints are determined by joint_names minus sonic_joint_names."""

    debug: bool = False
    """Whether to debug the action term."""

    def __post_init__(self) -> None:
        """Set the class_type based on action_type."""
        # Import here to avoid circular dependencies
        from .sonic_hierarchical_action import SONICHierachicalAction  # noqa: PLC0415
        from .sonic_latent_action import SONICLatentAction  # noqa: PLC0415
        from .sonic_latent_hand_policy_action import (  # noqa: PLC0415
            SONICLatentHandPolicyAction,
        )
        from .sonic_latent_residual_action import (  # noqa: PLC0415
            SONICLatentResidualAction,
        )
        from .sonic_residual_action import SONICResidualAction  # noqa: PLC0415

        if self.action_type == SONICActionType.HIERARCHICAL:
            self.class_type = SONICHierachicalAction
        elif self.action_type == SONICActionType.RESIDUAL:
            self.class_type = SONICResidualAction
        elif self.action_type == SONICActionType.LATENT_RESIDUAL:
            self.class_type = SONICLatentResidualAction
        elif self.action_type == SONICActionType.LATENT:
            self.class_type = SONICLatentAction
        elif self.action_type == SONICActionType.LATENT_HAND_POLICY:
            self.class_type = SONICLatentHandPolicyAction
        else:
            raise ValueError(f"Unknown SONIC action type: {self.action_type}")

        super().__post_init__()
