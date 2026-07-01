"""Validate the converted Vega+Sharpa robot USD and dump joints/bodies + key body world poses.

Used to plan the SimToolReal robot-swap: confirms the canonical 29 joints (L_arm + un-prefixed
left Sharpa) + the palm/fingertip bodies resolve, and prints where the left hand sits at the zero
pose so we can place the base + pick an arm pose that hovers the hand over the table (top z=0.53).

Run: source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
     OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/inspect_vega.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--base_x", type=float, default=0.0)
parser.add_argument("--base_y", type=float, default=0.0)
parser.add_argument("--base_z", type=float, default=0.0)
# optional candidate L_arm pose "j1,j2,...,j7" (rad) to preview the hand location
parser.add_argument("--arm", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

ROBOT_USD = "/home/cning/simtoolreal_isaaclab/assets/usd/vega_sharpa/robot.usd"

VEGA_JOINTS = [
    "L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4", "L_arm_j5", "L_arm_j6", "L_arm_j7",
    "left_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
    "left_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
]
PALM = "L_arm_l7"
HAND_BASE = "left_hand_C_MC"
FINGERTIPS = ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]
ARM_LINKS = [f"L_arm_l{i}" for i in range(1, 8)]


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    jp = {}
    if args.arm:
        vals = [float(x) for x in args.arm.split(",")]
        jp = {f"L_arm_j{i+1}": v for i, v in enumerate(vals)}

    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(args.base_x, args.base_y, args.base_z), joint_pos=jp),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=10.0)},
    ))
    sim.reset()
    # a few light steps so FK populates body_pos_w at the init pose (gravity off -> no contact churn)
    for _ in range(4):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim(); sim.step(render=False); robot.update(1.0 / 120.0)

    bn = list(robot.body_names)
    print("=" * 75)
    print(f"num_joints={robot.num_joints} num_bodies={robot.num_bodies}")
    print("JOINT_NAMES:", list(robot.joint_names))
    missing_j = [j for j in VEGA_JOINTS if j not in robot.joint_names]
    missing_b = [b for b in [PALM, HAND_BASE, *FINGERTIPS] if b not in bn]
    print("MISSING_JOINTS:", missing_j)
    print("MISSING_BODIES:", missing_b)
    lim = robot.root_physx_view.get_dof_limits()[0]
    print("L_arm limits:")
    for j in [f"L_arm_j{i}" for i in range(1, 8)]:
        k = robot.joint_names.index(j)
        print(f"  {j}: [{lim[k,0]:.3f}, {lim[k,1]:.3f}]")

    pos = robot.data.body_pos_w[0]
    def show(name):
        if name in bn:
            p = pos[bn.index(name)]
            print(f"  {name:16s} ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
    print(f"BODY WORLD POS (base=({args.base_x},{args.base_y},{args.base_z}) arm={args.arm or 'zero'}):")
    show("root")
    for n in ARM_LINKS: show(n)
    show(HAND_BASE)
    for n in FINGERTIPS: show(n)
    # palm center estimate = mean of fingertip bases (where the obs palm ~ should sit)
    ft = torch.stack([pos[bn.index(n)] for n in FINGERTIPS if n in bn]).mean(0)
    print(f"  fingertip-centroid ({ft[0]:+.3f},{ft[1]:+.3f},{ft[2]:+.3f})")
    print("VALIDATION:", "PASS" if not missing_j and not missing_b else "FAIL")
    print("=" * 75)
    simulation_app.close()


if __name__ == "__main__":
    main()
