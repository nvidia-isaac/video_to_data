"""Smoke-test the SimToolReal 'screwdriver' DirectRLEnv (screwdriver tool + passive screw).

Instantiates the env, resets, prints tool/screw/table geometry + the screw bbox, and steps
zero actions to confirm the screw rests on the table and obs dim is unchanged (140).
Run: ./isaaclab.sh -p <this> --headless --num_envs 8
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import torch  # noqa: E402

from simtoolreal_lab.tasks.screwdriver.screwdriver_env import ScrewdriverEnv  # noqa: E402
from simtoolreal_lab.tasks.screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg  # noqa: E402

RESULT_FILE = "/home/cning/simtoolreal_isaaclab/play_screwdriver_result.txt"


def main():
    rf = open(RESULT_FILE, "w")

    def emit(msg):
        rf.write(msg + "\n")
        rf.flush()
        os.fsync(rf.fileno())

    try:
        cfg = ScrewdriverEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs
        cfg.domain_randomization = False  # clean geometry check
        env = ScrewdriverEnv(cfg)
        emit(f"ENV_OK joints={env.num_dofs} obs_space={env.cfg.observation_space} act={env.cfg.action_space}")
        emit(f"OBJECT_USD={cfg.object_cfg.spawn.usd_path}")
        emit(f"SCREW_USD={cfg.screw_cfg.spawn.usd_path}")

        obs, _ = env.reset()
        emit(f"RESET_OK obs={tuple(obs['policy'].shape)} finite={bool(torch.isfinite(obs['policy']).all())}")
        if "critic" in obs:
            emit(f"CRITIC obs={tuple(obs['critic'].shape)}")

        env._compute_intermediate_values()
        r = lambda t: [round(x, 3) for x in t]
        eo = env.scene.env_origins[0]
        emit("WORLD obj=" + str(r(env.object.data.root_pos_w[0, :3].tolist())))
        emit("WORLD screw=" + str(r(env.screw.data.root_pos_w[0, :3].tolist())))
        emit("WORLD table=" + str(r(env.table.data.root_pos_w[0, :3].tolist())))
        emit("LOCAL obj=" + str(r((env.object.data.root_pos_w[0] - eo).tolist())))
        emit("LOCAL screw=" + str(r((env.screw.data.root_pos_w[0] - eo).tolist())))

        # screw bounding box (env 0) via USD geometry cache -> validates physical scale
        try:
            from pxr import Usd, UsdGeom

            stage = env.sim.stage
            screw_path = env.screw.cfg.prim_path.replace(".*", "0")
            prim = stage.GetPrimAtPath(screw_path)
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            ext = [round(mx[i] - mn[i], 4) for i in range(3)]
            emit(f"SCREW_BBOX path={screw_path} extents_xyz={ext} (meters)")
        except Exception as e:  # noqa: BLE001
            emit(f"SCREW_BBOX_FAILED {e!r}")

        # step zero actions: screw should rest, tool should not explode
        for i in range(args_cli.steps):
            act = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            obs, rew, term, trunc, info = env.step(act)
            if i % 10 == 0 or i == args_cli.steps - 1:
                sv = env.screw.data.root_lin_vel_w.norm(dim=-1).mean().item()
                emit(
                    f"step {i:3d} screw_z={env.screw.data.root_pos_w[:,2].mean().item():.3f} "
                    f"screw|v|={sv:.3f} obj_z={env.object.data.root_pos_w[:,2].mean().item():.3f} "
                    f"obs_finite={bool(torch.isfinite(obs['policy']).all())} "
                    f"term={int(term.sum().item())} trunc={int(trunc.sum().item())}"
                )
        emit("SCREWDRIVER_SMOKE_OK")
        env.close()
    except Exception:
        emit("EXCEPTION:\n" + traceback.format_exc())
    finally:
        rf.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
