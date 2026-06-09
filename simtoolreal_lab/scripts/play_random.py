"""Validate the SimToolReal DirectRLEnv: instantiate, reset, step random actions.

Instantiating the env also validates joint/body names (env __init__ indexes them).
Run: ./isaaclab.sh -p <this> --headless --num_envs 16
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--zero", action="store_true", help="apply zero actions (diagnostic)")
parser.add_argument("--fixed", action="store_true", help="use fixed-goal trajectory (eval mode)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402
import traceback  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab/simtoolreal_lab")

import torch  # noqa: E402

from simtoolreal_lab.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv  # noqa: E402
from simtoolreal_lab.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg  # noqa: E402

RESULT_FILE = "/home/cning/simtoolreal_isaaclab/play_result.txt"


def main():
    # Write to a result file (stdout is lost when simulation_app.close() hard-exits).
    rf = open(RESULT_FILE, "w")

    def emit(msg):
        rf.write(msg + "\n")
        rf.flush()
        import os

        os.fsync(rf.fileno())

    try:
        cfg = SimToolRealEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs
        if args_cli.fixed:
            cfg.use_fixed_goal_trajectory = True
        env = SimToolRealEnv(cfg)
        emit(f"ENV_OK joints={env.num_dofs} fingertips={env.num_fingertips} palm_id={env.palm_body_id}")
        emit(f"FINGERTIP_IDS={env.fingertip_body_ids}")
        emit(f"OBS_SPACE={env.cfg.observation_space} ACT_SPACE={env.cfg.action_space}")

        obs, _ = env.reset()
        emit(f"RESET_OK obs={tuple(obs['policy'].shape)} dtype={obs['policy'].dtype}")

        # geometry at reset (env-local frame), no actions applied yet
        env._compute_intermediate_values()
        r = lambda t: [round(x, 3) for x in t]
        emit("FRAME env_origin0=" + str(r(env.scene.env_origins[0].tolist())))
        emit("FRAME obj_world=" + str(r(env.object.data.root_pos_w[0, :3].tolist())))
        emit("FRAME obj_default=" + str(r(env.object.data.default_root_state[0, :3].tolist())))
        emit("FRAME table_world=" + str(r(env.table.data.root_pos_w[0, :3].tolist())))
        emit("FRAME robot_world=" + str(r(env.robot.data.root_pos_w[0, :3].tolist())))
        base0 = (env.robot.data.root_pos_w[0] - env.scene.env_origins[0]).tolist()
        emit("GEO base=" + str([round(x, 3) for x in base0]))
        emit("GEO palm=" + str([round(x, 3) for x in env.palm_center[0].tolist()]))
        emit("GEO object=" + str([round(x, 3) for x in env.object_pos[0].tolist()]))
        emit("GEO goal=" + str([round(x, 3) for x in env.goal_pos[0].tolist()]))
        for k in range(env.num_fingertips):
            emit(
                f"GEO ft{k} pos="
                + str([round(x, 3) for x in env.fingertip_pos[0, k].tolist()])
                + f" dist={env.curr_fingertip_distances[0, k].item():.3f}"
            )
        if obs["policy"].shape != (env.num_envs, env.cfg.observation_space):
            emit(f"OBS_SHAPE_MISMATCH got={tuple(obs['policy'].shape)}")

        zero = args_cli.zero
        for i in range(args_cli.steps):
            if zero:
                act = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            else:
                act = torch.rand((env.num_envs, env.cfg.action_space), device=env.device) * 2 - 1
            obs, rew, term, trunc, info = env.step(act)
            if zero and i < 8:
                op = env.object_pos[0].tolist()
                ov = env.object.data.root_lin_vel_w[0].norm().item()
                emit(f"zstep {i} obj0=" + str([round(x, 3) for x in op]) + f" |v|={ov:.2f} term0={int(term[0].item())}")
            if i % 5 == 0 or i == args_cli.steps - 1:
                ftd = env.curr_fingertip_distances.max(dim=-1).values
                fell = int((env.object_pos[:, 2] < 0.1).sum().item())
                far = int((ftd > 1.5).sum().item())
                emit(
                    f"step {i:3d} rew={rew.mean().item():+.4f} "
                    f"term={int(term.sum().item())} trunc={int(trunc.sum().item())} "
                    f"obj_z={env.object_pos[:,2].mean().item():.3f} "
                    f"ftd_min={ftd.min().item():.2f} ftd_mean={ftd.mean().item():.2f} ftd_max={ftd.max().item():.2f} "
                    f"fell={fell} hand_far={far}"
                )
        emit("RANDOM_ROLLOUT_OK")
        env.close()
    except Exception:
        emit("EXCEPTION:\n" + traceback.format_exc())
    finally:
        rf.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
