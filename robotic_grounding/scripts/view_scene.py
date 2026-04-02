"""View a scene: spawns object + support surface from a motion file (no RL).

Usage:
    python scripts/view_scene.py \
        --motion_file source/.../arctic_processed/sequence_id=arctic_s01_ketchup_use_01/robot_name=sharpa_wave

    # Headless (validate without GUI)
    python scripts/view_scene.py --motion_file <path> --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a scene from a motion file.")
parser.add_argument(
    "--motion_file", type=str, required=True, help="Path to parquet motion dir."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from robotic_grounding.tasks.scene_utils.scene_viewer_env_cfg import (  # noqa: E402
    SceneViewerEnvCfg,
)


def main() -> None:
    """View a scene from a motion file."""
    cfg = SceneViewerEnvCfg(motion_file=args_cli.motion_file)
    cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedEnv(cfg=cfg)
    env.reset()

    actions = torch.zeros(env.num_envs, 0, device=env.device)

    print(f"[INFO] Scene loaded from: {args_cli.motion_file}")
    print("[INFO] Close the window to exit.")

    step = 0
    while simulation_app.is_running():
        env.step(actions)
        step += 1
        if step % 100 == 0:
            print(f"[INFO] Step: {step}")
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
