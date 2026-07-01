"""Replay the real chirp through the IsaacSim Vega arm at the SYSID-TUNED gains and dump the
sim trajectories, so we can plot tuned-sim vs real-world data points offline.

Validation-only (no inertia calibration): loads logs/sysid/fit/tuned_gains.json, sets each arm
joint's tuned stiffness/damping, replays the recorded real q_cmd (delayed by the tuned delay_steps),
records the sim joint angle, and saves logs/sysid/fit/tuned_traj.npz with, per responsive joint:
  q_cmd (N,), q_real (N,), q_sim (N,), dt. Gravity off + neighbours locked stiff = the same single-
  joint condition the real per-joint probe used.

Run: source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
     OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/replay_tuned_arm.py --headless
"""

import argparse
import glob
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/cning/simtoolreal_isaaclab/logs/sysid/arm_sysid_amp_0.35_dur_60")
parser.add_argument("--gains", default="/home/cning/simtoolreal_isaaclab/logs/sysid/fit/tuned_gains.json")
parser.add_argument("--out", default="/home/cning/simtoolreal_isaaclab/logs/sysid/fit/tuned_traj.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
from simtoolreal_lab.tasks.vega_sharpa_robot import (  # noqa: E402
    VEGA_ROBOT_USD, VEGA_ARM_JOINTS, VEGA_INIT_JOINT_POS,
)

DEVICE = "cuda:0"


def main():
    with open(args.gains) as fh:
        gains = json.load(fh)["gains"]
    data = {}
    for p in sorted(glob.glob(os.path.join(args.data_dir, "L_arm_j*.npz"))):
        d = np.load(p, allow_pickle=True)
        nm = str(d["joint_name"])
        data[nm] = dict(q_cmd=np.asarray(d["q_cmd"], float), q_state=np.asarray(d["q_state"], float),
                        base=float(d["probe_center"]), rate=float(d["rate"]))
    dt = 1.0 / data[VEGA_ARM_JOINTS[0]]["rate"]

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=DEVICE))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=VEGA_ROBOT_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos=dict(VEGA_INIT_JOINT_POS)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=5000.0, damping=100.0,
                                              effort_limit=2000.0)}))
    sim.reset()

    base_pose = robot.data.default_joint_pos.clone()
    arm_ids = {}
    for j in VEGA_ARM_JOINTS:
        jid = robot.find_joints(j)[0][0]
        arm_ids[j] = jid
        if j in data:
            base_pose[0, jid] = data[j]["base"]

    def reset_to_base():
        robot.write_joint_state_to_sim(base_pose, torch.zeros_like(base_pose))
        robot.set_joint_position_target(base_pose); robot.write_data_to_sim()
        for _ in range(50):
            sim.step(); robot.update(dt)

    out = {"dt": dt}
    for j in VEGA_ARM_JOINTS:
        if j not in gains or j not in data:
            continue
        jid = arm_ids[j]
        robot.write_joint_stiffness_to_sim(float(gains[j]["stiffness"]), joint_ids=[jid])
        robot.write_joint_damping_to_sim(float(gains[j]["damping"]), joint_ids=[jid])
        delay = int(gains[j]["delay_steps"])
        cmd = data[j]["q_cmd"]
        cmd_t = torch.tensor(cmd, dtype=torch.float32, device=DEVICE)
        n = len(cmd)
        reset_to_base()
        q_buf = torch.zeros(n, device=DEVICE)
        for k in range(n):
            tgt = base_pose.clone()
            tgt[0, jid] = cmd_t[max(0, k - delay)]
            robot.set_joint_position_target(tgt); robot.write_data_to_sim()
            sim.step(); robot.update(dt)
            q_buf[k] = robot.data.joint_pos[0, jid]
        out[f"{j}_cmd"] = cmd
        out[f"{j}_real"] = data[j]["q_state"]
        out[f"{j}_sim"] = q_buf.cpu().numpy()
        print(f"{j}: replayed Kp={gains[j]['stiffness']:.2f} Kd={gains[j]['damping']:.2f} delay={delay}")

    np.savez(args.out, **out)
    print(f"saved -> {args.out}")
    simulation_app.close()


if __name__ == "__main__":
    main()
