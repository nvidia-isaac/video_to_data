"""Measure the ORIGINAL hammer task's hand->table offset at reset (env-local).

Builds the original IIWA+left-Sharpa HammerEnvCfg, resets to the deterministic startArmHigher pose,
and prints the palm-center + fingertip-centroid relative to the table (cuboid center AND top-surface
center). Used to replicate the same hand-to-table offset for the Vega robot.

Run: source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
     OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/measure_hand_table_offset.py --headless
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import sys  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import torch  # noqa: E402
import simtoolreal_lab.tasks  # noqa: E402,F401
from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402


def main():
    cfg = HammerEnvCfg()
    cfg.scene.num_envs = 1
    cfg.per_env_camera = False     # no rendering needed for a reset measurement
    env = HammerEnv(cfg, render_mode=None)
    env.reset()
    env._compute_intermediate_values()

    eo = env.scene.env_origins[0]
    palm = env.palm_center[0]                      # env-local
    ft = env.fingertip_pos[0]                       # (5,3) env-local
    ftc = ft.mean(0)
    table_w = env.table.data.root_pos_w[0] - eo     # env-local table cuboid center
    table_top = table_w.clone(); table_top[2] = table_w[2] + cfg.table_cfg.spawn.size[2] / 2.0
    obj = env.object_pos[0]

    def v(t):
        return f"({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})"

    print("=" * 70)
    print("ORIGINAL HAMMER (IIWA + left Sharpa), reset pose:")
    print(f"  palm_center        {v(palm)}")
    print(f"  fingertip_centroid {v(ftc)}")
    print(f"  per-fingertip:")
    names = ["index", "middle", "ring", "thumb", "pinky"]
    for i, nm in enumerate(names):
        print(f"     {nm:7s} {v(ft[i])}")
    print(f"  table cuboid center {v(table_w)}   table TOP center {v(table_top)}")
    print(f"  object_pos          {v(obj)}")
    print("-" * 70)
    print("OFFSETS (hand - table):")
    print(f"  palm - table_center     {v(palm - table_w)}")
    print(f"  palm - table_top        {v(palm - table_top)}")
    print(f"  fingertipC - table_cntr {v(ftc - table_w)}")
    print(f"  fingertipC - table_top  {v(ftc - table_top)}")
    print("=" * 70)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
