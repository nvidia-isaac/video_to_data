"""Tune Vega arm implicit-PD stiffness/damping (+delay) to match the real sysid chirp.

Two-pass, per-joint, in IsaacSim (gravity off, like the real probe + the deploy cfg):

  Pass 1 (calibrate inertia): lock all non-probed joints stiff, drive the probed joint with the
    RECORDED real q_cmd at a known Kp0,Kd0, fit the sim response -> wn_sim. Since the implicit PD
    on a (neighbours-locked) single joint is exactly 2nd-order with wn^2 = Kp/I, the effective
    inertia is  I = Kp0 / wn_sim^2.   (permutation-agnostic: only uses IsaacLab-order joint cmds/reads)

  Pass 2 (validate): set Kp = I*wn_target^2, Kd = 2*zeta_target*I*wn_target, replay the same real
    q_cmd DELAYED by round(T/dt) steps, and compare sim q vs the REAL q_state (VAF/RMS) + refit wn.

wn_target/zeta_target/T come from fit_arm_sysid.py (logs/sysid/fit/fit_params.json). Writes the tuned
per-joint stiffness/damping/delay to logs/sysid/fit/tuned_gains.json + an overlay plot.

Run: source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
     OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/tune_arm_sysid.py --headless
"""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/home/cning/simtoolreal_isaaclab/logs/sysid/arm_sysid_amp_0.35_dur_60")
parser.add_argument("--fit", default="/home/cning/simtoolreal_isaaclab/logs/sysid/fit/fit_params.json")
parser.add_argument("--out", default="/home/cning/simtoolreal_isaaclab/logs/sysid/fit")
parser.add_argument("--calib_steps", type=int, default=3000, help="samples used for the inertia calibration pass")
parser.add_argument("--kp0", type=float, default=200.0)
parser.add_argument("--kd0", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
sys.path.insert(0, "/home/cning/simtoolreal_isaaclab/scripts")
from fit_arm_sysid import fit_joint, clean  # noqa: E402
from simtoolreal_lab.tasks.vega_sharpa_robot import (  # noqa: E402
    VEGA_ROBOT_USD, VEGA_ARM_JOINTS, VEGA_INIT_JOINT_POS,
)

DEVICE = "cuda:0"


def main():
    # ---- load real data + fit targets ------------------------------------------------------------
    with open(args.fit) as fh:
        fit = json.load(fh)
    data = {}
    for p in sorted(glob.glob(os.path.join(args.data_dir, "L_arm_j*.npz"))):
        d = np.load(p, allow_pickle=True)
        name = str(d["joint_name"])
        data[name] = dict(q_cmd=np.asarray(d["q_cmd"], float),
                          q_state=np.asarray(d["q_state"], float),
                          base=float(d["probe_center"]), rate=float(d["rate"]))
    rate = data[VEGA_ARM_JOINTS[0]]["rate"]
    dt = 1.0 / rate

    # ---- spawn one Vega robot (gravity off, fixed base; same as make_vega_robot_cfg) -------------
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=DEVICE))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=VEGA_ROBOT_USD, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos=dict(VEGA_INIT_JOINT_POS)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=5000.0, damping=100.0,
                                              effort_limit=2000.0)},
    ))
    sim.reset()

    n_dof = robot.num_joints
    # baseline pose vector (IsaacLab joint order): per-joint real baseline for the 7 arm joints, else default
    base_pose = robot.data.default_joint_pos.clone()  # (1, n_dof)
    arm_ids = {}
    for j in VEGA_ARM_JOINTS:
        jid = robot.find_joints(j)[0][0]
        arm_ids[j] = jid
        if j in data:
            base_pose[0, jid] = data[j]["base"]

    def reset_to_base():
        robot.write_joint_state_to_sim(base_pose, torch.zeros_like(base_pose))
        robot.set_joint_position_target(base_pose)
        robot.write_data_to_sim()
        for _ in range(50):
            sim.step(); robot.update(dt)

    def run_chirp(jid, cmd, n_steps, delay_steps=0):
        """Drive joint jid with target sequence cmd (others held at base_pose); return sim q (n_steps,)."""
        reset_to_base()
        q_out = np.zeros(n_steps)
        cmd_t = torch.tensor(cmd, dtype=torch.float32, device=DEVICE)
        for k in range(n_steps):
            tgt = base_pose.clone()
            src = max(0, k - delay_steps)
            tgt[0, jid] = cmd_t[src]
            robot.set_joint_position_target(tgt)
            robot.write_data_to_sim()
            sim.step(); robot.update(dt)
            q_out[k] = float(robot.data.joint_pos[0, jid])
        return q_out

    results = {}
    tuned = {}
    responsive = [j for j in VEGA_ARM_JOINTS if j in fit and not fit[j].get("skipped")]

    print(f"\n{'joint':10s} {'I_eff':>8s} {'Kp':>9s} {'Kd':>8s} {'delaystp':>8s} "
          f"{'wn_tgt':>7s} {'wn_sim':>7s} {'VAF%':>7s} {'RMS':>8s}")
    fig, ax = plt.subplots(len(responsive), 1, figsize=(11, 2.0 * len(responsive)), squeeze=False)
    for i, j in enumerate(responsive):
        jid = arm_ids[j]
        f = fit[j]
        wn_t, zeta_t, T = f["wn_rad"], f["zeta"], f["delay_s"]
        cmd = data[j]["q_cmd"]
        base = data[j]["base"]

        # ---- Pass 1: calibrate inertia with Kp0,Kd0 --------------------------------------------
        robot.write_joint_stiffness_to_sim(args.kp0, joint_ids=[jid])
        robot.write_joint_damping_to_sim(args.kd0, joint_ids=[jid])
        nC = min(args.calib_steps, len(cmd))
        q_cal = run_chirp(jid, cmd[:nC], nC, delay_steps=0)
        u_cal = cmd[:nC] - base
        fc = fit_joint(u_cal, q_cal - base, np.ones(nC, bool), dt)
        I_eff = args.kp0 / (fc["wn"] ** 2)

        # ---- map to target gains --------------------------------------------------------------
        Kp = I_eff * wn_t ** 2
        Kd = 2.0 * zeta_t * I_eff * wn_t
        delay_steps = int(round(T / dt))

        # ---- Pass 2: validate vs REAL q_state -------------------------------------------------
        robot.write_joint_stiffness_to_sim(float(Kp), joint_ids=[jid])
        robot.write_joint_damping_to_sim(float(Kd), joint_ids=[jid])
        q_val = run_chirp(jid, cmd, len(cmd), delay_steps=delay_steps)

        y_real, valid = clean(data[j]["q_state"])
        e = (q_val - y_real)[valid]
        rms = float(np.sqrt(np.mean(e ** 2)))
        vaf = float(100.0 * (1.0 - np.var(e) / np.var((y_real - base)[valid])))
        fv = fit_joint(cmd - base, q_val - base, np.ones(len(cmd), bool), dt)

        tuned[j] = dict(stiffness=float(Kp), damping=float(Kd), delay_steps=delay_steps,
                        delay_s=float(T), inertia=float(I_eff))
        results[j] = dict(wn_target_hz=wn_t / (2 * np.pi), wn_sim_hz=fv["wn"] / (2 * np.pi),
                          zeta_target=zeta_t, vaf=vaf, rms=rms)
        print(f"{j:10s} {I_eff:8.3f} {Kp:9.2f} {Kd:8.2f} {delay_steps:8d} "
              f"{wn_t/(2*np.pi):7.3f} {fv['wn']/(2*np.pi):7.3f} {vaf:7.1f} {rms:8.4f}")

        ax[i, 0].plot(cmd, color="tab:blue", lw=0.6, label="q_cmd")
        ax[i, 0].plot(y_real, color="tab:orange", lw=0.9, label="real q_state")
        ax[i, 0].plot(q_val, color="tab:red", lw=0.9, ls="--", label="sim (tuned)")
        ax[i, 0].set_title(f"{j}  Kp={Kp:.1f} Kd={Kd:.1f} delay={delay_steps}stp  VAF={vaf:.0f}%", fontsize=9)
        ax[i, 0].set_ylabel("rad"); ax[i, 0].grid(alpha=0.3)
        if i == 0:
            ax[i, 0].legend(fontsize=7, loc="upper right", ncol=3)
    ax[-1, 0].set_xlabel("time step (100 Hz)")
    fig.suptitle("real q_state vs IsaacSim implicit-PD with TUNED gains"); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "tuned_validation.png"), dpi=130)

    with open(os.path.join(args.out, "tuned_gains.json"), "w") as fh:
        json.dump({"gains": tuned, "validation": results, "rate_hz": rate}, fh, indent=2)
    print(f"\nsaved -> {args.out}/tuned_gains.json, tuned_validation.png")
    simulation_app.close()


if __name__ == "__main__":
    main()
