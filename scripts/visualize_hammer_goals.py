"""Kinematically play the HAMMER nail-in GOAL poses (no policy / no physics control).

Teleports the claw_hammer through its per-env goal sequence (`per_env_goals` from nail_traj:
lift -> reorient face-down -> over the nail -> lower to contact -> repeated strikes) so you can SEE
whether the goal poses make sense relative to the nail + thread_test. The robot is parked out of
the way; the hammer is made kinematic (set-pose, no fall). The nail/bar are shown (kinematic).
Records an RTX mp4.

Run: cd IsaacLab && OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
       ~/simtoolreal_isaaclab/scripts/visualize_hammer_goals.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out_prefix", type=str, default="goalposes_hammer")
parser.add_argument("--cam_eye", type=str, default="-0.45,-0.55,1.00")
parser.add_argument("--cam_lookat", type=str, default="0.15,0.0,0.60")
parser.add_argument("--hold", type=int, default=6, help="interpolated frames between consecutive goals")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pin", action="store_true", help="pin the layout (fixed nominal screw/bar at center 0.18, no randomization) so a tight camera can stay framed on the nail")
parser.add_argument("--physical", action="store_true", help="use the PHYSICAL prismatic screw (dynamic, friction-held) so the teleported hammer DRIVES it down via contact, instead of the kinematic fixed screw")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402

VIDEO_DIR = "/home/cning/simtoolreal_isaaclab/videos"


def main():
    torch.manual_seed(args_cli.seed)
    cfg = HammerEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = 1
    cfg.domain_randomization = False
    cfg.use_tighten_goals = True       # build per_env_goals (nail_traj) from the hammer's spawn pose
    cfg.randomize_layout = True         # REQUIRED so per_env_goals are generated (+ hammer clear of bar)
    if args_cli.pin:                    # deterministic nominal layout (screw at world ~(0.076,0,0.587))
        cfg.layout_threadtest_center_x_range = (0.18, 0.18)
        cfg.layout_threadtest_center_y_range = (0.0, 0.0)
        # hammer START clear of the bar (bar left edge ~x=0.023, tool footprint r~0.127) -> -0.15
        cfg.layout_screwdriver_x_range = (-0.15, -0.15)
        cfg.layout_screwdriver_y_range = (0.0, 0.0)
        cfg.layout_yaw_range = (0.0, 0.0)
    # physical: dynamic prismatic screw (the teleported hammer drives it down via contact); the hits
    # target the SEATED depth (clearance = -nail_start_height) so the raised nail is driven in.
    cfg.physical_screw = args_cli.physical
    if args_cli.physical:
        cfg.screw_contact_clearance = -cfg.nail_start_height
        cfg.sim.physx.gpu_collision_stack_size = 2 ** 29
    cfg.max_consecutive_successes = 0
    cfg.episode_length_s = 1.0e6
    # kinematic hammer: hold whatever pose we teleport it to (no gravity/physics drift)
    cfg.object_cfg.spawn.rigid_props.kinematic_enabled = True
    # park the arm straight up so it isn't sitting on top of the teleported hammer
    for j in [f"iiwa14_joint_{i}" for i in range(1, 8)]:
        cfg.robot_cfg.init_state.joint_pos[j] = 0.0
    # viewer camera + RecordVideo
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.resolution = (1280, 720)
    cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
    cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))

    env = gym.make("Isaac-SimToolReal-Hammer-Direct-v0", cfg=cfg, render_mode="rgb_array")
    env.reset()
    base = env.unwrapped
    goals = base.per_env_goals[0].clone()       # (T,7) xyz + xyzw, env-local
    T = goals.shape[0]

    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env, video_folder=VIDEO_DIR, step_trigger=lambda s: s == 0,
        video_length=(T + 5) * args_cli.hold, name_prefix=args_cli.out_prefix, disable_logger=True)
    env.reset()

    eo = base.scene.env_origins[0]
    eidx = torch.tensor([0], device=base.device)
    zero_act = torch.zeros((1, base.cfg.action_space), device=base.device)
    z6 = torch.zeros((1, 6), device=base.device)
    gp = goals[:, 0:3] + eo                      # (T,3) world
    gq = goals[:, [6, 3, 4, 5]]                  # (T,4) wxyz
    SUB = args_cli.hold

    def step_pose(pos, quat):
        pose = torch.cat([pos, quat / quat.norm()]).unsqueeze(0)
        base.object.write_root_pose_to_sim(pose, eidx)
        base.object.write_root_velocity_to_sim(z6, eidx)
        env.step(zero_act)

    phys = args_cli.physical and getattr(base, "screw_asm", None) is not None

    def nail_mm():
        return base.screw_asm.data.joint_pos[0, 0].item() * 1000.0 if phys else 0.0

    print(f"[viz] playing {T} hammer goals, {SUB} interp frames/segment (~{(T + 5) * SUB} frames)"
          f"{' | PHYSICAL nail (driven by contact)' if phys else ''}", flush=True)
    if phys:
        print(f"  [viz] nail height at start = {nail_mm():.1f} mm (raised)", flush=True)
    for _ in range(SUB):                         # settle on the first goal
        step_pose(gp[0], gq[0])
    for i in range(T - 1):                        # LERP(pos) + nlerp(quat) between goals
        q0, q1 = gq[i], gq[i + 1]
        if torch.dot(q0, q1) < 0:
            q1 = -q1
        for k in range(SUB):
            t = (k + 1) / SUB
            step_pose((1 - t) * gp[i] + t * gp[i + 1], (1 - t) * q0 + t * q1)
        if i % 12 == 0:
            print(f"  [viz] goal {i + 1}/{T}" + (f"  nail={nail_mm():.1f} mm" if phys else ""), flush=True)
    for _ in range(SUB * 3):                      # hold on the final pose
        step_pose(gp[-1], gq[-1])
    if phys:
        print(f"  [viz] nail height at end = {nail_mm():.1f} mm (driven in)", flush=True)

    try:
        env.render()
    except Exception:
        pass
    env.close()
    simulation_app.close()
    print(f"[viz] wrote {VIDEO_DIR}/{args_cli.out_prefix}-step-0.mp4", flush=True)


if __name__ == "__main__":
    main()
