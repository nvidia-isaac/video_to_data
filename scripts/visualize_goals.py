"""Kinematically play the tighten GOAL poses (no policy / no physics control).

Teleports the screwdriver through its per-env goal sequence (`per_env_goals` — lift -> reorient
tip-down + blade-to-slot -> over screw -> lower to contact -> rotate) so you can SEE whether the
goal poses make sense relative to the screw + thread_test. The robot is parked out of the way and
the screwdriver is made kinematic (set-pose, no fall). Records an RTX mp4.

Run:
  cd IsaacLab && (venv) OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
    ~/simtoolreal_isaaclab/scripts/visualize_goals.py --headless --demo_task tighten_screw
"""

import argparse
from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="screwdriver", choices=["screwdriver", "screwdriver043"],
                    help="which screwdriver env's goals to visualize (044 flat slot, or 043 cross slot)")
parser.add_argument("--demo_task", type=str, default="tighten_screw")
parser.add_argument("--hold", type=int, default=5, help="render frames held at each goal pose")
parser.add_argument("--cam_eye", type=str, default="-0.45,-0.50,0.78")
parser.add_argument("--cam_lookat", type=str, default="0.04,0.0,0.64")
parser.add_argument("--randomize_layout", action="store_true", help="randomize the screw/screwdriver layout (goals adapt)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # rendering

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg  # noqa: E402
from simtoolreal_lab.tasks.screwdriver043.screwdriver043_env_cfg import Screwdriver043EnvCfg  # noqa: E402

VIDEO_DIR = "/home/cning/simtoolreal_isaaclab/videos"

ENV_MAP = {
    "screwdriver":    (ScrewdriverEnvCfg,    "Isaac-SimToolReal-Screwdriver-Direct-v0"),
    "screwdriver043": (Screwdriver043EnvCfg, "Isaac-SimToolReal-Screwdriver043-Direct-v0"),
}


def main():
    torch.manual_seed(args_cli.seed)
    cfg_cls, task_id = ENV_MAP[args_cli.env]
    cfg = cfg_cls()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = 1
    cfg.demo_mode = True                     # makes the env build per_env_goals at reset
    cfg.use_fixed_goal_trajectory = True     # REQUIRED: loads start_pose -> demo_start_pose so the
                                             # env actually generates per_env_goals (else they're 0)
    cfg.randomize_layout = args_cli.randomize_layout
    cfg.pretrained_object_scale = (2.5, 0.75, 0.75)
    cfg.trajectory_path = f"{ORIG_REPO}/dextoolbench/trajectories/screwdriver/044_screwdriver/{args_cli.demo_task}.json"
    cfg.max_consecutive_successes = 0        # no success-based reset
    cfg.episode_length_s = 1.0e6             # never time out
    # kinematic screwdriver: hold whatever pose we write (no gravity/physics drift)
    cfg.object_cfg.spawn.rigid_props.kinematic_enabled = True
    # park the arm straight up so it doesn't sit on top of the teleported screwdriver
    for j in [f"iiwa14_joint_{i}" for i in range(1, 8)]:
        cfg.robot_cfg.init_state.joint_pos[j] = 0.0
    # camera framed on the screw region
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.resolution = (1280, 720)
    cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
    cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))

    cfg.sim.physx.gpu_collision_stack_size = 2 ** 28  # SDF colliders -> bump (cheap for 1 env)
    env = gym.make(task_id, cfg=cfg, render_mode="rgb_array")
    obs, _ = env.reset()
    base = env.unwrapped
    goals = base.per_env_goals[0].clone()    # (T,7) xyz + xyzw, env-local
    T = goals.shape[0]

    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env, video_folder=VIDEO_DIR, step_trigger=lambda s: s == 0,
        video_length=(T + 4) * args_cli.hold,
        name_prefix=f"goalposes_{args_cli.env}_{args_cli.demo_task}", disable_logger=True)
    video_env = env
    env.reset()

    eo = base.scene.env_origins[0]
    eidx = torch.tensor([0], device=base.device)
    zero_act = torch.zeros((1, base.cfg.action_space), device=base.device)
    z6 = torch.zeros((1, 6), device=base.device)
    gp = goals[:, 0:3] + eo               # (T,3) world positions
    gq = goals[:, [6, 3, 4, 5]]           # (T,4) wxyz
    SUB = args_cli.hold                   # interpolated frames between consecutive goals (smooth)

    def step_pose(pos, quat):
        pose = torch.cat([pos, quat / quat.norm()]).unsqueeze(0)
        base.object.write_root_pose_to_sim(pose, eidx)
        base.object.write_root_velocity_to_sim(z6, eidx)
        env.step(zero_act)

    print(f"[viz] smoothly playing {T} goals, {SUB} interp frames/segment (~{(T + 3) * SUB} frames)")
    for _ in range(SUB):                  # settle on the first goal
        step_pose(gp[0], gq[0])
    for i in range(T - 1):                # continuous LERP(pos) + nlerp(quat) between goals
        q0, q1 = gq[i], gq[i + 1]
        if torch.dot(q0, q1) < 0:
            q1 = -q1                       # shortest-path quaternion blend
        for k in range(SUB):
            t = (k + 1) / SUB
            step_pose((1 - t) * gp[i] + t * gp[i + 1], (1 - t) * q0 + t * q1)
    for _ in range(SUB * 3):              # hold on the final (rotated) pose
        step_pose(gp[-1], gq[-1])

    try:
        video_env.render()
    except Exception:
        pass
    video_env.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
