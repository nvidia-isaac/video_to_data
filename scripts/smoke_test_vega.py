"""Smoke test for the Vega robot-swap tasks: construct the env, check the 29-DOF mapping, step it.

Builds VegaHammerEnvCfg / VegaScrewdriverEnvCfg directly (no checkpoint), confirms the canonical
joint / palm / fingertip names resolve against the Vega articulation, then steps zero actions a few
times to confirm the scene loads + physics runs without error. Prints obs/state shapes + where the
left hand ends up relative to the table object.

Run: source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
     OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/smoke_test_vega.py \
        --headless --task hammer --num_envs 2 --steps 30
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["hammer", "screwdriver"], default="hammer")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# the Vega cfgs use a per-env camera -> cameras must be enabled
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")  # so `import simtoolreal_lab` resolves

import torch  # noqa: E402
import simtoolreal_lab.tasks  # noqa: E402,F401  (triggers gym registration)


def main():
    if args.task == "hammer":
        from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv
        from simtoolreal_lab.tasks.vega_hammer.vega_hammer_env_cfg import VegaHammerEnvCfg
        cfg, EnvCls = VegaHammerEnvCfg(), HammerEnv
    else:
        from simtoolreal_lab.tasks.screwdriver.screwdriver_env import ScrewdriverEnv
        from simtoolreal_lab.tasks.vega_screwdriver.vega_screwdriver_env_cfg import VegaScrewdriverEnvCfg
        cfg, EnvCls = VegaScrewdriverEnvCfg(), ScrewdriverEnv

    cfg.scene.num_envs = args.num_envs
    env = EnvCls(cfg, render_mode=None)

    print("=" * 75)
    print(f"TASK: vega_{args.task}  num_envs={args.num_envs}")
    print(f"robot joints={env.robot.num_joints} bodies={env.robot.num_bodies}")
    print(f"canonical 29-DOF ids resolved: {len(env.canonical_dof_ids)} "
          f"(arm {env.canonical_dof_ids[:7]})")
    print(f"palm_body_id={env.palm_body_id} ({env.cfg.palm_body})  "
          f"fingertip_ids={env.fingertip_body_ids}")

    obs, _ = env.reset()
    print(f"obs['policy'] shape={tuple(obs['policy'].shape)} "
          f"critic={tuple(obs['critic'].shape) if 'critic' in obs else None}")

    zero = torch.zeros((args.num_envs, cfg.action_space), device=env.device)
    for i in range(args.steps):
        obs, rew, term, trunc, info = env.step(zero)
    # where did the left hand / palm settle vs the object?
    env._compute_intermediate_values()
    palm = env.palm_center[0].tolist()
    objp = env.object_pos[0].tolist()
    ft = env.fingertip_pos[0].mean(0).tolist()
    print(f"after {args.steps} steps:")
    print(f"  palm_center (env0)   = ({palm[0]:+.3f},{palm[1]:+.3f},{palm[2]:+.3f})")
    print(f"  fingertip-mean (env0)= ({ft[0]:+.3f},{ft[1]:+.3f},{ft[2]:+.3f})")
    print(f"  object_pos (env0)    = ({objp[0]:+.3f},{objp[1]:+.3f},{objp[2]:+.3f})")
    print(f"  reward[0]={rew[0].item():.4f}  term={term[0].item()} trunc={trunc[0].item()}")
    print("SMOKE TEST: PASS")
    print("=" * 75)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
