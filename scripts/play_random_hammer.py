"""Smoke-test the 'hammer' DirectRLEnv (claw_hammer + prismatic-jointed nail).

Instantiates the env, resets, prints obs/geometry + the prismatic nail joint, and steps zero
actions to confirm it loads and the nail slides freely (physical_screw / prismatic joint).
Run: ./isaaclab.sh -p <this> --headless --num_envs 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, sys, traceback  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import torch  # noqa: E402

from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402

RESULT_FILE = "/home/cning/simtoolreal_isaaclab/play_hammer_result.txt"


def main():
    rf = open(RESULT_FILE, "w")
    emit = lambda m: (rf.write(m + "\n"), rf.flush(), os.fsync(rf.fileno()))
    try:
        cfg = HammerEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs
        cfg.domain_randomization = False
        cfg.use_tighten_goals = True   # generate the nail-in goals so the generator is exercised
        env = HammerEnv(cfg)
        emit(f"ENV_OK joints={env.num_dofs} obs={env.cfg.observation_space} act={env.cfg.action_space}")
        emit(f"OBJECT_USD={cfg.object_cfg.spawn.usd_path}")
        emit(f"physical_screw={cfg.physical_screw}  goal_gen={cfg.goal_generator_module}  traj_T={env._traj_T}")
        emit(f"goal_gen TIP={env._goal_gen.TIP.tolist()} TOOL={env._goal_gen.TOOL.tolist()}")
        if env.screw_asm is not None:
            emit(f"SCREW_ASM joints={env.screw_asm.num_joints} names={list(env.screw_asm.joint_names)}")

        obs, _ = env.reset()
        emit(f"RESET_OK obs={tuple(obs['policy'].shape)} finite={bool(torch.isfinite(obs['policy']).all())}")
        eo = env.scene.env_origins
        r = lambda t: [round(x, 3) for x in t]
        emit("LOCAL hammer=" + str(r((env.object.data.root_pos_w[0] - eo[0]).tolist())))
        emit("WORLD screw_head=" + str(r((env.screw_head_world[0]).tolist())))

        for i in range(args_cli.steps):
            act = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            obs, rew, term, trunc, info = env.step(act)
            if i % 10 == 0 or i == args_cli.steps - 1:
                nail = env.screw_asm.data.joint_pos[:, 0] if env.screw_asm is not None else torch.zeros(1)
                emit(f"step {i:3d} hammer_z={env.object.data.root_pos_w[:,2].mean().item():.3f} "
                     f"nail_slide(mean)={nail.mean().item():+.4f} "
                     f"obs_finite={bool(torch.isfinite(obs['policy']).all())} "
                     f"term={int(term.sum().item())} trunc={int(trunc.sum().item())}")
        emit("HAMMER_SMOKE_OK")
        env.close()
    except Exception:
        emit("EXCEPTION:\n" + traceback.format_exc())
    finally:
        rf.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
