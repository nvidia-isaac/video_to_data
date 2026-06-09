"""Validate the converted robot/object USDs load in Isaac Lab and dump joint/body names.

Run: ./isaaclab.sh -p <this> --headless
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402

ASSETS = "/home/cning/simtoolreal_isaaclab/simtoolreal_lab/assets/usd"
ROBOT_USD = f"{ASSETS}/iiwa14_left_sharpa/robot.usd"
OBJECT_USD = f"{ASSETS}/claw_hammer/claw_hammer.usd"

# Canonical SimToolReal joint order + bodies we need for obs.
JOINT_NAMES_ISAACGYM = [
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
]
NEEDED_BODIES = ["iiwa14_link_7", "left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.8, 0.0)),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=10.0)},
        )
    )
    obj = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Object",
            spawn=sim_utils.UsdFileCfg(usd_path=OBJECT_USD),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.8, 0.65)),
        )
    )
    sim.reset()

    print("=" * 70)
    print(f"ROBOT num_joints={robot.num_joints} num_bodies={robot.num_bodies}")
    print("ROBOT_JOINT_NAMES:", list(robot.joint_names))
    print("ROBOT_BODY_NAMES:", list(robot.body_names))
    missing_j = [j for j in JOINT_NAMES_ISAACGYM if j not in robot.joint_names]
    missing_b = [b for b in NEEDED_BODIES if b not in robot.body_names]
    print("MISSING_JOINTS:", missing_j)
    print("MISSING_BODIES:", missing_b)
    print(f"OBJECT num_bodies={obj.num_bodies} body_names={list(obj.body_names)}")
    print("VALIDATION:", "PASS" if not missing_j and not missing_b else "FAIL")
    print("=" * 70)
    simulation_app.close()


if __name__ == "__main__":
    main()
