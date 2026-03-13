from __future__ import annotations

import argparse
import os
from typing import Any

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run SONIC policy with pretrained encoder/decoder."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--use_tracking",
    action="store_true",
    default=False,
    help="Use tracking command for encoder input (vs dummy zeros).",
)
parser.add_argument(
    "--use_hierarchical_action",
    action="store_true",
    default=False,
    help="Test hierarchical action term (reads from command term).",
)
parser.add_argument(
    "--use_residual_action",
    action="store_true",
    default=False,
    help="Test residual action term (zero residuals).",
)
parser.add_argument(
    "--use_latent_residual_action",
    action="store_true",
    default=False,
    help="Test latent residual action term (zero latent residuals).",
)
parser.add_argument(
    "--use_latent_hand_policy_action",
    action="store_true",
    default=False,
    help="Test latent hand policy action term (dummy actions, overriden by reference motion).",
)
parser.add_argument(
    "--disable_terminations",
    action="store_true",
    default=False,
    help="Disable all termination conditions.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument(
    "--video_length", type=int, default=350, help="Video length in steps."
)
parser.add_argument(
    "--scene_config",
    type=str,
    default=None,
    help="Path to scene configuration YAML file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after app launch (required by Isaac Lab)
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402, PLC0415
from robotic_grounding.assets import (  # noqa: E402, PLC0415
    MOTION_ASSET_DIR,
    POLICY_ASSET_DIR,
)
from robotic_grounding.assets.g1 import (  # noqa: E402, PLC0415
    G1_CYLINDER_MODEL_12_DEX_DELAYED_CFG,
    G1_HAND_JOINT_NAMES,
    G1_MODEL_12_ACTION_SCALE,
)
from robotic_grounding.assets.policies.grasp import (  # noqa: E402, PLC0415
    G1GraspPolicy,
    GraspPolicyCfg,
)
from robotic_grounding.tasks.v2p_whole_body import (  # noqa: E402, PLC0415
    G1_SONIC_JOINT_NAMES,
    G1SonicEEEnvCfg,
    G1SonicEnvCfg,
)
from robotic_grounding.tasks.v2p_whole_body.mdp.actions import (  # noqa: E402, PLC0415
    SONICActionCfg,
    SONICActionType,
    SonicPolicy,
)


def dummy_encoder_input(num_envs: int, device: str = "cpu") -> torch.Tensor:
    """Create dummy encoder inputs for G1 mode (all zeros except mode indicator).

    For testing without tracking command.
    """
    encoder_input = torch.zeros(num_envs, 1772, device=device)
    encoder_input[:, 0] = 0.0  # G1 mode = 0 (scalar index, not one-hot)
    return encoder_input


def run_policy(
    env: Any,
    policy: SonicPolicy,
    num_steps: int = 350,
    use_tracking: bool = False,
) -> None:
    """Run the SONIC policy for a number of steps.

    Args:
        env: Environment
        policy: SonicPolicy instance
        num_steps: Number of steps to run
        use_tracking: If True, use tracking command observations. If False, use dummy zeros.
    """
    print(f"\nRunning policy for {num_steps} steps...")
    print(
        f"Encoder mode: {'Tracking Command' if use_tracking else 'Dummy (all zeros)'}"
    )

    # Get unwrapped env for device access
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = unwrapped_env.device

    # Reset environment
    obs_dict, _ = env.reset()

    # Replace tokenizer observations with dummy if not using tracking
    if not use_tracking:
        obs_dict["sonic_tokenizer"] = dummy_encoder_input(
            unwrapped_env.num_envs, device=device
        )
        print("Using dummy encoder input (all zeros)")
    else:
        print("Using tracking observations from tokenizer group")

    print(f"Encoder obs shape: {obs_dict['sonic_tokenizer'].shape}")
    print(f"Decoder obs shape: {obs_dict['sonic_policy'].shape}")

    # Run for specified steps
    for step in range(num_steps):
        # Run policy (encoder + decoder)
        with torch.inference_mode():
            actions = policy(obs_dict)

        # Step environment
        with torch.inference_mode():
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

        if not use_tracking:
            obs_dict["sonic_tokenizer"] = dummy_encoder_input(
                unwrapped_env.num_envs, device=device
            )

        # Print progress
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{num_steps}")

    print(f"\nCompleted {num_steps} steps!")

    # Print metrics
    with torch.inference_mode():
        info = unwrapped_env.command_manager.reset(
            torch.arange(unwrapped_env.num_envs, device=device)
        )
        print("Metrics:")
        for key, value in info.items():
            print(f"    {key}: {value}")


def run_hierarchical_action_test(env: Any, num_steps: int = 350) -> None:
    """Test hierarchical action term by reading actions directly from command term.

    This is a pass-through test where actions are read from the tracking command
    and should result in perfect tracking (no change from reference motion).

    Args:
        env: Environment with hierarchical action term configured
        num_steps: Number of steps to run
    """
    print(f"\nRunning hierarchical action test for {num_steps} steps...")
    print("Actions will be read directly from command term (pass-through test)")

    # Get unwrapped env for device access
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = unwrapped_env.device

    # Get command term for reading actions
    command = unwrapped_env.command_manager.get_term("motion")

    # Reset environment
    obs_dict, _ = env.reset()

    print(f"Action space: {env.action_space}")
    print(f"Action dim: {env.action_space.shape[-1]}")

    # Run for specified steps
    for step in range(num_steps):
        # Build action from command term
        # Action structure: [joint_commands (N), base_ori_6d (6)]

        # Get current joint positions from command (first future frame)
        joint_pos_multi_future = (
            command.command_joint_pos_multi_future
        )  # (num_envs, num_future_frames, num_joints)
        joint_commands = joint_pos_multi_future[
            :, 0, :
        ]  # (num_envs, num_joints) - absolute positions

        # Get current orientation from command (first future frame)
        # Convert from command's orientation representation to 6D
        ori_multi_future = (
            command.command_root_rot_dif_l_multi_future
        )  # (num_envs, num_future_frames, 6)
        base_ori_6d = ori_multi_future[:, 0, :]  # (num_envs, 6)

        # Concatenate to form action
        actions = torch.cat(
            [joint_commands, base_ori_6d], dim=-1
        )  # (num_envs, num_joints + 6)

        # Step environment
        with torch.inference_mode():
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Print progress
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{num_steps}")

    print(f"\nCompleted {num_steps} steps!")

    # Print metrics
    with torch.inference_mode():
        info = unwrapped_env.command_manager.reset(
            torch.arange(unwrapped_env.num_envs, device=device)
        )
        print("Metrics:")
        for key, value in info.items():
            print(f"    {key}: {value}")


def run_residual_action_test(env: Any, num_steps: int = 350) -> None:
    """Test residual action term with zero residuals (pass-through test).

    This is a pass-through test where zero residuals are applied to all joint positions,
    which should result in base commanded positions being used.

    Args:
        env: Environment with residual action term configured
        num_steps: Number of steps to run
    """
    print(f"\nRunning residual action test for {num_steps} steps...")
    print("All residuals set to zero (pass-through test - base commanded positions)")

    # Get unwrapped env for device access
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = unwrapped_env.device

    # Get action term to determine number of joints
    action_term = unwrapped_env.action_manager.get_term("joint_pos")
    num_joints = action_term._num_joints

    # Reset environment
    obs_dict, _ = env.reset()

    print(f"Action space: {env.action_space}")
    print(f"Action dim (all joints): {env.action_space.shape[-1]}")
    print(f"Number of joints: {num_joints}")

    # Run for specified steps with zero residuals
    for step in range(num_steps):
        # Action structure: [joint_residuals (num_joints)]
        # All zeros = no residual = base positions from command
        actions = torch.zeros(unwrapped_env.num_envs, num_joints, device=device)

        # Step environment
        with torch.inference_mode():
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Print progress
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{num_steps}")

    print(f"\nCompleted {num_steps} steps!")

    # Print metrics
    with torch.inference_mode():
        info = unwrapped_env.command_manager.reset(
            torch.arange(unwrapped_env.num_envs, device=device)
        )
        print("Metrics:")
        for key, value in info.items():
            print(f"    {key}: {value}")


def run_latent_residual_action_test(env: Any, num_steps: int = 350) -> None:
    """Test latent residual action term with zero latent residuals (pass-through test).

    This is a pass-through test where zero latent residuals are applied,
    which should result in normal SONIC policy output (no modification to token state).

    Args:
        env: Environment with latent residual action term configured
        num_steps: Number of steps to run
    """
    print(f"\nRunning latent residual action test for {num_steps} steps...")
    print("All latent residuals set to zero (pass-through test - normal SONIC output)")

    # Get unwrapped env for device access
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = unwrapped_env.device

    # Get action term to determine latent dimension
    action_term = unwrapped_env.action_manager.get_term("joint_pos")
    latent_dim = action_term.policy.encoder_output_dim

    # Reset environment
    obs_dict, _ = env.reset()

    print(f"Action space: {env.action_space}")
    print(f"Action dim (latent space): {env.action_space.shape[-1]}")
    print(f"Encoder output dimension: {latent_dim}")

    # Run for specified steps with zero latent residuals
    for step in range(num_steps):
        # Action structure: [latent_residuals (encoder_output_dim)]
        # All zeros = no latent modification = normal SONIC output
        actions = torch.zeros(unwrapped_env.num_envs, latent_dim, device=device)

        # Step environment
        with torch.inference_mode():
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Print progress
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{num_steps}")

    print(f"\nCompleted {num_steps} steps!")

    # Print metrics
    with torch.inference_mode():
        info = unwrapped_env.command_manager.reset(
            torch.arange(unwrapped_env.num_envs, device=device)
        )
        print("Metrics:")
        for key, value in info.items():
            print(f"    {key}: {value}")


def run_latent_hand_policy_action_test(env: Any, num_steps: int = 350) -> None:
    """Test latent hand policy action term with dummy actions, overriden by reference motion.

    Args:
        env: Environment with latent hand policy action term configured
        num_steps: Number of steps to run
    """
    print(f"\nRunning latent hand policy action test for {num_steps} steps...")
    print("Dummy actions, overriden by reference motion")

    # Get unwrapped env for device access
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = unwrapped_env.device

    # Get action term to determine latent dimension
    action_term = unwrapped_env.action_manager.get_term("joint_pos")
    latent_dim = action_term.policy.encoder_output_dim

    # Reset environment
    obs_dict, _ = env.reset()

    print(f"Action space: {env.action_space}")
    print(f"Action dim (latent space): {env.action_space.shape[-1]}")
    print(f"Encoder output dimension: {latent_dim}")

    # Run for specified steps
    for step in range(num_steps):
        # Action structure: [latent_state (encoder_output_dim)]
        # All zeros = dummy actions
        actions = torch.zeros(unwrapped_env.num_envs, latent_dim, device=device)

        # Step environment
        with torch.inference_mode():
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Print progress
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{num_steps}")

    print(f"\nCompleted {num_steps} steps!")

    # Print metrics
    with torch.inference_mode():
        info = unwrapped_env.command_manager.reset(
            torch.arange(unwrapped_env.num_envs, device=device)
        )
        print("Metrics:")
        for key, value in info.items():
            print(f"    {key}: {value}")


def main() -> None:
    """Main function to run SONIC policy."""
    # Configuration
    POLICY_DIR = f"{POLICY_ASSET_DIR}/sonic"
    MOTION_FILE = f"{MOTION_ASSET_DIR}/apple_pick_and_place_retarget_motion_w_body.h5"

    print("=" * 60)
    print("SONIC Policy Inference")
    print("=" * 60)
    print(f"Hierarchical action: {args_cli.use_hierarchical_action}")
    print(f"Tracking mode: {args_cli.use_tracking}")
    print(
        f"Motion file: {MOTION_FILE if args_cli.use_tracking else 'N/A (using dummy input)'}"
    )
    print(f"Num environments: {args_cli.num_envs}")
    print(f"Terminations disabled: {args_cli.disable_terminations}")
    print(f"Video recording: {args_cli.video}")
    print("=" * 60)

    # Create environment
    print("\nCreating environment...")
    cfg = (
        G1SonicEEEnvCfg() if args_cli.use_latent_hand_policy_action else G1SonicEnvCfg()
    )
    cfg.scene.num_envs = args_cli.num_envs

    # Override scene config if provided
    if args_cli.scene_config is not None:
        from robotic_grounding.tasks.scene_utils import (  # noqa: PLC0415
            SceneConfig,
            apply_scene_config,
        )

        cfg.scene_config_path = args_cli.scene_config
        scene_config = SceneConfig.from_yaml(args_cli.scene_config)
        apply_scene_config(cfg, scene_config)
        MOTION_FILE = scene_config.motion_file

    # Set motion file for tracking command (required)
    cfg.commands.motion.motion_file = MOTION_FILE

    # Configure hierarchical action if requested
    if args_cli.use_hierarchical_action:
        print("\nUsing Hierarchical Action Term")
        original_scale = cfg.actions.joint_pos.scale
        cfg.actions.joint_pos = SONICActionCfg(
            action_type=SONICActionType.HIERARCHICAL,
            policy_dir=POLICY_DIR,
            asset_name="robot",
            joint_names=[".*"],  # All joints (including hands)
            sonic_joint_names=G1_SONIC_JOINT_NAMES,  # SONIC controls only non-hand joints
            command_name="motion",
            use_default_offset=True,
            scale=original_scale,
        )
    elif args_cli.use_residual_action:
        print("\nUsing Residual Action Term")
        original_scale = cfg.actions.joint_pos.scale
        cfg.actions.joint_pos = SONICActionCfg(
            action_type=SONICActionType.RESIDUAL,
            policy_dir=POLICY_DIR,
            asset_name="robot",
            joint_names=[".*"],  # All joints (including hands)
            sonic_joint_names=G1_SONIC_JOINT_NAMES,  # SONIC controls only non-hand joints
            command_name="motion",
            use_default_offset=True,
            scale=original_scale,
        )
    elif args_cli.use_latent_residual_action:
        print("\nUsing Latent Residual Action Term")
        original_scale = cfg.actions.joint_pos.scale
        cfg.actions.joint_pos = SONICActionCfg(
            action_type=SONICActionType.LATENT_RESIDUAL,
            policy_dir=POLICY_DIR,
            asset_name="robot",
            joint_names=[".*"],  # All joints (including hands)
            sonic_joint_names=G1_SONIC_JOINT_NAMES,  # SONIC controls only non-hand joints
            command_name="motion",
            use_default_offset=True,
            scale=original_scale,
        )
    elif args_cli.use_latent_hand_policy_action:
        print("\nUsing Latent Hand Policy Action Term")
        original_scale = cfg.actions.joint_pos.scale
        cfg.actions.joint_pos = SONICActionCfg(
            action_type=SONICActionType.LATENT_HAND_POLICY,
            policy_dir=POLICY_DIR,
            asset_name="robot",
            joint_names=[".*"],
            sonic_joint_names=G1_SONIC_JOINT_NAMES,
            command_name="motion",
            use_default_offset=True,
            scale=original_scale,
            debug=True,
        )
        cfg.actions.joint_pos.hand_policy_class = G1GraspPolicy
        cfg.actions.joint_pos.hand_policy_cfg = GraspPolicyCfg(
            asset_name="robot",
            joint_names=G1_HAND_JOINT_NAMES,
        )
    else:
        # When not using hierarchical action, use non-hands robot and disable filtering
        print("\nUsing Direct SONIC Policy Inference")

        # Switch to non-hands robot configuration
        cfg.scene.robot = G1_CYLINDER_MODEL_12_DEX_DELAYED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        cfg.actions.joint_pos.scale = G1_MODEL_12_ACTION_SCALE

    # Disable terminations if requested
    if args_cli.disable_terminations:
        print("\nDisabling all termination conditions...")
        cfg.terminations = None

    # Create environment with render mode for video
    render_mode = "rgb_array" if args_cli.video else None
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode)

    print(f"Environment created with {env.num_envs} environments")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Record video
    if args_cli.video:
        video_dir = os.path.abspath(os.path.join("videos", "sonic_policy"))
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"\n[INFO] Recording video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Run appropriate test
    if args_cli.use_hierarchical_action:
        run_hierarchical_action_test(env, num_steps=args_cli.video_length)
    elif args_cli.use_residual_action:
        run_residual_action_test(env, num_steps=args_cli.video_length)
    elif args_cli.use_latent_residual_action:
        run_latent_residual_action_test(env, num_steps=args_cli.video_length)
    elif args_cli.use_latent_hand_policy_action:
        run_latent_hand_policy_action_test(env, num_steps=args_cli.video_length)
    else:
        # Load SONIC policy for direct inference
        print("\nLoading SONIC policy...")
        policy = SonicPolicy(POLICY_DIR)
        run_policy(
            env,
            policy,
            num_steps=args_cli.video_length,
            use_tracking=args_cli.use_tracking,
        )

    # Cleanup
    env.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
    simulation_app.close()
