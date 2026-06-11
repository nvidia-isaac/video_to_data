"""Verify ScrewdriverEnv layout randomization:
  (a) screw pose RELATIVE to thread_test is constant across envs/resets (screw stays in hole),
  (b) the screwdriver never overlaps the thread_test bar footprint.
Run: ./isaaclab.sh -p <this> --headless --num_envs 128
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--resets", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, sys, traceback  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import torch  # noqa: E402
from isaaclab.utils.math import quat_conjugate, quat_apply  # noqa: E402

from simtoolreal_lab.tasks.screwdriver.screwdriver_env import ScrewdriverEnv  # noqa: E402
from simtoolreal_lab.tasks.screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg  # noqa: E402

RESULT_FILE = "/home/cning/simtoolreal_isaaclab/check_layout_result.txt"


def main():
    rf = open(RESULT_FILE, "w")
    emit = lambda m: (rf.write(m + "\n"), rf.flush(), os.fsync(rf.fileno()))
    try:
        cfg = ScrewdriverEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs
        cfg.domain_randomization = False
        env = ScrewdriverEnv(cfg)
        hx, hy = cfg.layout_threadtest_half_extents
        r = cfg.layout_min_clearance
        eo = env.scene.env_origins

        env.reset()
        for k in range(args_cli.resets):
            env.step(torch.zeros((env.num_envs, env.cfg.action_space), device=env.device))
            tt_p = env.thread_test.data.root_pos_w - eo
            tt_q = env.thread_test.data.root_quat_w
            sc_p = env.screw.data.root_pos_w - eo
            ob_p = env.object.data.root_pos_w - eo

            # (a) screw pose in thread_test local frame -> should be identical for every env
            rel = quat_apply(quat_conjugate(tt_q), sc_p - tt_p)
            rel_std = rel.std(dim=0)
            emit(f"[reset {k}] screw-in-TT-frame mean={[round(x,4) for x in rel.mean(0).tolist()]} "
                 f"std={[round(x,5) for x in rel_std.tolist()]}  (std~0 => screw stays in hole)")

            # (b) screwdriver vs thread_test bar OBB (xy): distance from tool center to rect
            # thread_test rotation is pure-z (qz), so yaw = 2*atan2(z, w)
            yaw = 2.0 * torch.atan2(tt_q[:, 3], tt_q[:, 0])
            cos, sin = torch.cos(yaw), torch.sin(yaw)
            # the bar's mesh origin is at its END; geometry center = origin + Rz @ (0.1325, 0)
            lox, loy = cfg.layout_pivot_xy[0] - cfg.thread_test_cfg.init_state.pos[0], 0.0
            gcx = tt_p[:, 0] + cos * lox - sin * loy
            gcy = tt_p[:, 1] + sin * lox + cos * loy
            dx, dy = ob_p[:, 0] - gcx, ob_p[:, 1] - gcy
            lx = cos * dx + sin * dy
            ly = -sin * dx + cos * dy
            ddx = lx - lx.clamp(-hx, hx)
            ddy = ly - ly.clamp(-hy, hy)
            dist = torch.sqrt(ddx * ddx + ddy * ddy + 1e-12)
            tool_radius = 0.127                               # 044 footprint radius from root
            n_overlap = int((dist < tool_radius).sum().item())  # tool footprint actually intersects bar
            n_below_clear = int((dist < r).sum().item())        # closer than required clearance
            emit(f"[reset {k}] tool->bar dist: min={dist.min().item():.4f} mean={dist.mean().item():.4f}  "
                 f"FOOTPRINT_OVERLAPS(<{tool_radius})={n_overlap}  below_clearance({r})={n_below_clear}")
            env.reset()

        emit("LAYOUT_CHECK_OK")
        env.close()
    except Exception:
        emit("EXCEPTION:\n" + traceback.format_exc())
    finally:
        rf.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
