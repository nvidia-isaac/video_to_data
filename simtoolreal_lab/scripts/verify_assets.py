"""Prove the robot + object are really loaded as physical assets (masses, collision, dynamics).

Run: ./isaaclab.sh -p scripts/verify_assets.py --headless
Writes findings to verify_assets.txt (stdout is lost when the app hard-exits).
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab/simtoolreal_lab")
from simtoolreal_lab.tasks.simtoolreal.simtoolreal_env_cfg import (  # noqa: E402
    SimToolRealEnvCfg,
)

OUT = "/home/cning/simtoolreal_isaaclab/verify_assets.txt"


def main():
    rf = open(OUT, "w")

    def emit(m):
        rf.write(m + "\n")
        rf.flush()
        os.fsync(rf.fileno())

    cfg = SimToolRealEnvCfg()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    # single-env prim paths (no /envs/env_.* cloning)
    cfg.robot_cfg.prim_path = "/World/Robot"
    cfg.object_cfg.prim_path = "/World/Object"
    cfg.table_cfg.prim_path = "/World/Table"
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(cfg.robot_cfg)
    obj = RigidObject(cfg.object_cfg)
    table = RigidObject(cfg.table_cfg)
    sim.reset()

    # --- ROBOT ---
    masses = robot.root_physx_view.get_masses()  # (1, num_bodies)
    lim = robot.root_physx_view.get_dof_limits()[0]  # (num_dofs, 2)
    emit("=== ROBOT (articulation) ===")
    emit(f"num_joints (DOF) = {robot.num_joints}")
    emit(f"num_bodies       = {robot.num_bodies}")
    emit(f"total mass (kg)  = {masses.sum().item():.3f}")
    emit(f"per-body mass min/max = {masses.min().item():.4f} / {masses.max().item():.4f}")
    emit(f"DOF limit ranges (rad): min={lim[:,0].min().item():.3f} max={lim[:,1].max().item():.3f}")
    emit(f"arm joint 2 limits = [{lim[1,0].item():.3f}, {lim[1,1].item():.3f}]  (real IIWA ~±2.09)")
    emit(f"sample bodies: {[b for b in robot.body_names if b in ('iiwa14_link_7','left_thumb_DP','left_index_DP')]}")

    # --- OBJECT ---
    om = obj.root_physx_view.get_masses()
    emit("")
    emit("=== OBJECT (rigid body: claw_hammer) ===")
    emit(f"num_bodies = {obj.num_bodies}")
    emit(f"mass (kg)  = {om.sum().item():.4f}")
    emit(f"spawn world pos = {[round(x,3) for x in obj.data.root_pos_w[0,:3].tolist()]}")

    # --- DYNAMICS: drop the object, confirm it collides with the table (doesn't fall through) ---
    emit("")
    emit("=== PHYSICS (zero torque, object should rest ON the table ~0.53, not fall to ground) ===")
    for i in range(60):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(1 / 120)
        obj.update(1 / 120)
        if i % 15 == 0 or i == 59:
            emit(f"  step {i:2d}: object_z = {obj.data.root_pos_w[0,2].item():.4f}  |v|={obj.data.root_lin_vel_w[0].norm().item():.3f}")
    emit("VERIFY_DONE")
    rf.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
