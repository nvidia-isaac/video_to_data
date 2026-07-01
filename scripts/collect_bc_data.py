"""Collect a behavior-cloning dataset by rolling out a (pretrained) policy on a SimToolReal task.

For every control step it records, per sub-environment:
  - the rendered camera image (facing the robot), default 640x360;
  - the robot's CURRENT 29-DOF joint state (arm 7 + fingers 22), canonical order;
  - the 29-DOF DELTA joint-position ACTION = commanded joint target - current joint pos.

The per-env camera is isolated to its OWN sub-environment: a large --env_spacing pushes the
neighbor sub-envs far away, and a small --cam_z_far clips them out of the view, so each image
shows only the current sub-environment.

Defaults: the `hammer` task + the ORIGINAL pretrained SimToolReal SAPG policy. --task selects
another env (claw_hammer / screwdriver / screwdriver043); --orig_config + --checkpoint select
another policy (the net is rebuilt from the config so any matching checkpoint loads).

Output: a robomimic-style HDF5 ->
  /data                         attrs: task, env, policy, fps, resolution, joint_names, action_space, num_demos
  /data/demo_<i>/obs/image      (T, H, W, 3) uint8     gzip
  /data/demo_<i>/obs/joint_pos  (T, 29) float32        canonical (arm7 + fingers22)
  /data/demo_<i>/actions        (T, 29) float32        delta joint-position command
  /data/demo_<i>                attrs: num_samples, goals_reached, env_index
  /mask/qualified               demo names with goals_reached >= --min_goals (robomimic filter key)

Run:
  cd IsaacLab && ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/collect_bc_data.py \
      --headless --num_envs 30 --steps 600           # hammer + pretrained policy
"""

import argparse
import math

from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"
REPO = "/home/cning/simtoolreal_isaaclab"
DEFAULT_CONFIG = f"{ORIG_REPO}/pretrained_policy/config.yaml"
DEFAULT_CKPT = f"{ORIG_REPO}/pretrained_policy/model.pth"
TRAJ_ROOT = f"{ORIG_REPO}/dextoolbench/trajectories"
LOG_DIR = f"{REPO}/logs/simtoolreal"

# task name -> (gym id, cfg module, cfg class)
TASKS = {
    "hammer":         ("Isaac-SimToolReal-Hammer-Direct-v0",         "simtoolreal_lab.tasks.hammer.hammer_env_cfg",                 "HammerEnvCfg"),
    "claw_hammer":    ("Isaac-SimToolReal-ClawHammer-Direct-v0",      "simtoolreal_lab.tasks.simtoolreal.simtoolreal_env_cfg",       "SimToolRealEnvCfg"),
    "screwdriver":    ("Isaac-SimToolReal-Screwdriver-Direct-v0",     "simtoolreal_lab.tasks.screwdriver.screwdriver_env_cfg",       "ScrewdriverEnvCfg"),
    "screwdriver043": ("Isaac-SimToolReal-Screwdriver043-Direct-v0",  "simtoolreal_lab.tasks.screwdriver043.screwdriver043_env_cfg", "Screwdriver043EnvCfg"),
    # pretrained policy on the Vega RIGHT hand via shadow-IIWA retarget (left arm parked). Same hammer
    # task/reward/goal as "hammer"; only the embodiment + obs/action retarget differ.
    "vega_hammer_retarget": ("Isaac-SimToolReal-VegaHammerRetarget-Direct-v0", "simtoolreal_lab.tasks.vega_hammer_retarget.vega_hammer_retarget_env_cfg", "VegaHammerRetargetEnvCfg"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="hammer", choices=list(TASKS), help="which task env to roll out (default hammer)")
parser.add_argument("--orig_config", type=str, default=DEFAULT_CONFIG, help="policy net-arch config.yaml (default = original pretrained)")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="policy checkpoint .pth (default = original pretrained)")
parser.add_argument("--num_envs", type=int, default=100, help="parallel rollouts (keep a multiple of 6 for the SAPG net)")
parser.add_argument("--num_demos", type=int, default=50, help="stop once this many SUCCESSFUL demos (screw seated in the hole) are collected")
parser.add_argument("--max_ep_steps", type=int, default=1500, help="per-episode step budget; an env that fails to seat the screw within this is discarded + reset (>=1500 lets the expert recover from tool-displacement teleports)")
parser.add_argument("--success_joint", type=float, default=-0.006, help="(hammer) success when the nail prismatic joint <= this (near the -0.008 lower limit = screw in the hole)")
parser.add_argument("--strike_contact", type=float, default=-1.0, help="(hammer) TIGHTER success: also require the striking face within this distance (m) of the nail head at the seated moment (the head CONTACTING the nail = a genuine strike). <0 = use the cfg default; large value (e.g. 5) = effectively off (still logs the distance for calibration)")
parser.add_argument("--hand_reject", type=float, default=-1.0, help="(hammer) RULE OUT hand-nailing: reject a seated nail if a fingertip is within this distance (m) of the nail head (the HAND pressed it in, not the hammer). <0 = use cfg default (0.04); 0 = disable the hand check")
parser.add_argument("--nail_move_eps", type=float, default=-1.0, help="(hammer) TIGHTEST: fail the episode if the nail joint moves > this (m) in a step while the hammer head is far (the nail must move only when struck). <0 = use cfg default (0.001); 0 = disable")
parser.add_argument("--success_deg", type=float, default=150.0, help="(screwdriver) success when the physical screw has rotated CLOCKWISE >= this many degrees from its start (tightened), with the tip in the slot")
parser.add_argument("--success_tolerance", type=float, default=0.01, help="goal-reach tolerance: 0.01 makes the policy track each goal to the hit pose (clean hammering). A loose value lets goals race ahead -> degenerate fast behavior.")
parser.add_argument("--warmup", type=int, default=40, help="render-only frames after reset (loads RTX textures) before recording starts")
parser.add_argument("--action_noise", type=float, default=0.0, help="std of zero-mean Gaussian noise on the executed expert action ([-1,1] space) -> DART-lite recovery data (the expert corrects, recorded)")
parser.add_argument("--force_perturbation", action="store_true", help="apply the original mass-scaled random force/torque perturbations to the object once lifted (forceScale=20, torqueScale=2, per-env log-uniform[0.001,0.1] trigger). Off by default -> clean expert demos; on -> the expert recovers from random kicks (more robust BC data)")
parser.add_argument("--force_scale", type=float, default=-1.0, help="(with --force_perturbation) override the random-force scale (cfg default 20.0 = randn*mass*scale per kick). Higher = stronger kicks -> more recovery data but lower yield. <0 = use cfg default")
parser.add_argument("--torque_scale", type=float, default=-1.0, help="(with --force_perturbation) override the random-torque scale (cfg default 2.0). <0 = use cfg default")
parser.add_argument("--tool_displacement", action="store_true", help="TELEPORT the tool by a random delta pose (once lifted, with a cooldown) to simulate slips/drops/failed grasps -> the expert recovers (re-grasp). Much more disruptive than force kicks; success-filtering keeps the recovered demos. Off by default")
parser.add_argument("--tool_displace_pos", type=float, default=-1.0, help="(with --tool_displacement) MAX position offset (m) per teleport; each teleport's magnitude is sampled HALF-NORMAL over [min,max] (more small than big) (cfg default max 0.10, min 0.02). <0 = use cfg default")
parser.add_argument("--tool_displace_rot", type=float, default=-1.0, help="(with --tool_displacement) MAX rotation (rad) per teleport; sampled half-normal over [min,max] (more small than big; cfg default max 0.50 ~29deg, min 0.10). <0 = use cfg default")
parser.add_argument("--tool_displace_pregrasp", action="store_true", help="(with --tool_displacement) also teleport the tool BEFORE the grasp (default only-when-lifted) -> simulates failed grasps / tool not where expected; the expert re-approaches. Hammer only (the screwdriver freeze overrides pre-grasp teleports)")
parser.add_argument("--joint_displacement", action="store_true", help="TELEPORT the robot's 29 arm+hand joint positions by a random per-joint delta (+zero velocity), INDEPENDENTLY sampled from the tool teleport (own prob/cooldown/scale, fires any time). Simulates a control glitch / the robot getting bumped; PD+expert recover. Feeds the same per-step teleport flag (chunk-loss masking covers it). Off by default")
parser.add_argument("--random_action", action="store_true", help="DART/DAgger: with random_action_prob/step the robot EXECUTES random 29-dim delta actions for a BURST of N=round(|N(0,steps_std^2)|) steps (off-distribution flailing); the RECORDED action stays the expert correction + the burst is flagged for chunk-loss masking, and the expert recovers after. Off by default")
parser.add_argument("--random_action_std", type=float, default=-1.0, help="std of the random delta action (action space ~[-1,1]); <0 keeps the cfg default (0.5)")
parser.add_argument("--random_action_steps_std", type=float, default=-1.0, help="std of the burst length N=round(|N(0,this^2)|) control steps; <0 keeps the cfg default (27 -> mean ~21 steps ~0.36s)")
parser.add_argument("--random_action_prob", type=float, default=-1.0, help="per-env per-step probability of STARTING a burst (no cooldown); <0 keeps the cfg default (0.007)")
parser.add_argument("--joint_displace_arm_scale", type=float, default=-1.0, help="(with --joint_displacement) MAX ARM-joint delta std (rad) per teleport, sampled half-normal over [min,max] (cfg default max 0.10 ~6deg). Arm causes big end-effector movement -> keep small. <0 = use cfg default")
parser.add_argument("--joint_displace_hand_scale", type=float, default=-1.0, help="(with --joint_displacement) MAX HAND-joint delta std (rad) per teleport, sampled half-normal over [min,max] (cfg default max 0.30 ~17deg; fingers move less per rad). <0 = use cfg default")
parser.add_argument("--no_tool_displacement", action="store_true", help="(hammer) disable the default-on tool-displacement teleport perturbation (collect a CLEAN hammer dataset)")
parser.add_argument("--no_simtoolreal", action="store_true", help="(hammer) disable the default-on SimToolReal-specialist recording (obs/proprio[joint_vel...] + obs/keypoints_rel_palm)")
parser.add_argument("--num_episodes", type=int, default=0, help="stop after this many TOTAL episodes (success+failure), not just successes. 0 = no cap (stop at --num_demos or --step_cap). Use with --save_failures to grab a fixed set of mixed episodes for review")
parser.add_argument("--save_failures", action="store_true", help="also save an mp4 for FAILED episodes (to videos/<task>_bc_demos_fail/) so you can review failure/recovery behavior. Failures are NOT added to the HDF5")
parser.add_argument("--fail_videos", type=int, default=-1, help="cap the number of FAILURE mp4s saved (with --save_failures). -1 = unlimited. Use to get a balanced success+failure set at a low success rate")
parser.add_argument("--goal_noise", action="store_true", help="add per-env, per-waypoint Gaussian noise to the goal poses the expert tracks (task-agnostic decay schedule: large at the trajectory start, ~0 at the end). Diversifies the expert's path -> wider state coverage (the BC student doesn't see the goal). Off by default")
parser.add_argument("--goal_noise_scale", type=float, default=1.0, help="multiplier on the goal-noise schedule's per-waypoint sigma (only with --goal_noise). 1.0 = base magnitudes (3cm/8.6deg -> 2mm/0.6deg); lower if it hurts the success yield")
parser.add_argument("--goal_diversify", action="store_true", help="diversify the goal TRAJECTORY per episode (distinct from --goal_noise's per-waypoint jitter): per-env random GENERATION params (hammer: lift_height/swing_angle/n_strikes -> different shapes) + a SMOOTH correlated approach-path offset (decays to 0 by the strike). Coherent, success-filtered. Off by default")
parser.add_argument("--goal_diversify_scale", type=float, default=1.0, help="(with --goal_diversify) multiplier on the param-noise ranges + smooth-offset magnitude. 0 = base/no diversity; >1 = wider (may lower yield)")
parser.add_argument("--goal_diversify_offset_std", type=float, default=-1.0, help="(with --goal_diversify) std (m) of the smooth approach-path offset control points; <0 = cfg default (0.03)")
parser.add_argument("--simtoolreal", action="store_true", help="also record obs/keypoints_rel_palm (T,8,3) + obs/proprio (T,109) for the SimToolReal specialist: PALM-RELATIVE keypoints + joint_vel/prev_targets/palm pose/fingertips (expert-faithful state)")
parser.add_argument("--wrist", action="store_true", help="also record a palm-facing wrist camera as obs/image_wrist")
parser.add_argument("--wrist_cam_width", type=int, default=640)
parser.add_argument("--wrist_cam_height", type=int, default=480)
parser.add_argument("--wrist_eye", type=str, default="0.08,-0.02,0.08", help="wrist cam eye (iiwa14_link_7-local): ~8cm to the side of the wrist base")
parser.add_argument("--wrist_lookat", type=str, default="-0.02,-0.015,0.18", help="wrist cam look-at (link-local): the grasp region")
parser.add_argument("--wrist_up", type=str, default="Y")
parser.add_argument("--wrist_focal", type=float, default=14.0)
parser.add_argument("--out", type=str, default="", help="output .hdf5 path (default datasets/<task>_bc.hdf5)")
parser.add_argument("--min_goals", type=int, default=1, help="demos whose env reached >= this many goals are listed in /mask/qualified (all demos are still saved)")
# recording camera (facing the robot) + sub-env isolation
parser.add_argument("--cam_width", type=int, default=640)
parser.add_argument("--cam_height", type=int, default=480)
parser.add_argument("--cam_eye", type=str, default="0.0,-0.65,0.85", help="camera eye (env-local) -- default faces the robot down +y")
parser.add_argument("--cam_lookat", type=str, default="0.0,0.30,0.55", help="camera look-at (env-local)")
parser.add_argument("--env_spacing", type=float, default=4.0, help="LARGE -> neighbor sub-envs are far away (out of the z_far-clipped view)")
parser.add_argument("--table_dist", type=float, default=0.15, help="move the work-table this far (m) FURTHER from the robot (objects stay put -> still reachable; the table slides under them). Capped in-env so the objects stay supported. DEFAULT 0.15 (matches the datasets/eval); pass 0 for the original position")
parser.add_argument("--cam_z_far", type=float, default=2.5, help="camera far clip (m): clips out neighbor sub-envs so only the current one is captured")
parser.add_argument("--ground_texture", type=str, default="tiles", choices=["tiles", "white", "none"], help="background floor+backdrop: 'tiles' = the in-house Isaac Lab marble-tile MDL material; 'white' = a plain solid-white floor + backdrop; 'none' = the default grid")
parser.add_argument("--ground_rgb", type=str, default="", help="plain solid-color floor+backdrop as 'r,g,b' in [0,1] (e.g. '0.2,0.2,0.2' = dark grey). Overrides --ground_texture when set")
parser.add_argument("--screw_rgb", type=str, default="", help="(hammer) recolor the driven screw/nail as 'r,g,b' in [0,1] (e.g. '0.55,0.72,0.82' = light blue like the hammer head). '' = keep the asset color")
parser.add_argument("--ground_texture_scale", type=float, default=2.0, help="UV tiling scale for the ground texture (bigger -> smaller/more tiles)")
parser.add_argument("--dome_texture", type=str, default="none", choices=["none", "studio", "carpentry", "autoshop", "hospital", "sky"], help="environment-light HDRI (existing Isaac Lab/Sim sky asset): lights the scene AND forms the camera background. 'none' = plain gray dome")
parser.add_argument("--dome_intensity", type=float, default=1500.0, help="brightness scale for the HDRI dome light")
# task knobs
parser.add_argument("--kinematic_screw", action="store_true", help="hammer/screwdriver: use the kinematic screw instead of the dynamic (physical) one")
parser.add_argument("--hit_depth", type=float, default=0.04, help="(hammer) goal overshoot below the screw head (m)")
parser.add_argument("--no_video", action="store_true", help="skip saving per-demo mp4 videos (videos are saved by default)")
parser.add_argument("--video_demos", type=int, default=100, help="save mp4 videos only for the first N successful demos (the HDF5 still stores ALL demos). -1 = all")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--step_cap", type=int, default=0, help="override the loop's safety step cap (0 = auto). Raise it to run MORE rounds at a low success rate.")
parser.add_argument("--viz_slot_box", action="store_true", help="(screwdriver debug) at the first engaged-turning step, spawn markers for the slot box + its computed center + the ACTUAL physical screw + the tip, save the per-env-cam frame, and exit")
parser.add_argument("--viz_cw_trigger", type=float, default=25.0, help="(viz_slot_box) capture once any env's clockwise rotation exceeds this many degrees")
parser.add_argument("--no_image", action="store_true", help="do NOT record/render RGB images (no obs/image, no camera) -> fast, tiny state-only dataset for the state specialist. State fields (keypoints/proprio/goal) come from physics, not the camera")
parser.add_argument("--sysid_gains", action="store_true", help="(vega_hammer_retarget only) swap the arm actuators to the SYSID-TUNED soft gains from vega_sharpa_sysid.py (matches the real harmonic-drive joints' ~0.2 Hz bandwidth). Stiffness/damping only; the ~40 ms command delay is NOT applied (the implicit actuator has no delay buffer).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.no_image  # cameras only needed when recording RGB

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, REPO)
import simtoolreal_lab.tasks  # noqa: E402, F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from simtoolreal_lab.tasks.simtoolreal.simtoolreal_env_cfg import JOINT_NAMES_ISAACGYM  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from simtoolreal_lab.tasks.simtoolreal.keypoint_utils import compute_simtoolreal_obs, compute_goal_rel  # noqa: E402

# Object-centric KEYPOINTS recorded alongside the rgb (for the state-based keypoint policy):
# 4 TOOL keypoints (the env's object_keypoints = screwdriver / hammer bbox corners) + 4 SCREW
# keypoints (corners at the screw pose). Unit cube corners + small screw half-extents (m).
_KP_CORNERS = [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
_SCREW_HALF = (0.008, 0.008, 0.015)

GROUND_MDLS = {
    "tiles": f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
}

# HDRI environment-light assets (existing Isaac Lab/Sim skies): light the scene AND form the background.
DOME_HDRIS = {
    "studio":     f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
    "carpentry":  f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    "autoshop":   f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    "hospital":   f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    "sky":        f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
}


def build_agent_cfg(num_envs: int) -> dict:
    """Rebuild the SAPG net from the policy config.yaml so the checkpoint loads bit-for-bit."""
    with open(args_cli.orig_config) as f:
        full = yaml.safe_load(f)
    params = full["train"]["params"]
    params["seed"] = args_cli.seed
    params["load_checkpoint"] = True
    params["load_path"] = args_cli.checkpoint
    c = params["config"]
    c["name"] = "00_bc_collect"
    c["device_name"] = c["device"] = "cuda:0"
    c["multi_gpu"] = False
    c["num_actors"] = num_envs
    c["clip_actions"] = False
    c["max_epochs"] = 1
    c["max_frames"] = 100
    c["expl_coef_block_size"] = max(1, num_envs // 6)
    c["minibatch_size"] = num_envs
    if "central_value_config" in c:
        c["central_value_config"]["minibatch_size"] = num_envs
    c["train_dir"] = LOG_DIR
    c["defer_summaries_sec"] = 5
    c["summaries_interval_sec_min"] = 5
    c["summaries_interval_sec_max"] = 300
    c.setdefault("player", {})
    c["player"]["games_num"] = 128
    c["player"]["deterministic"] = True
    return {"params": params}


def make_cfg():
    """Configure the chosen task env for zero-shot pretrained deploy + per-env recording camera."""
    gym_id, mod, cls = TASKS[args_cli.task]
    EnvCfg = getattr(importlib.import_module(mod), cls)
    cfg = EnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = args_cli.env_spacing  # large -> neighbors out of the (z_far-clipped) view
    VEGA = args_cli.task == "vega_hammer_retarget"  # Vega retarget env: its cfg already positioned the
    #   scene (move_scene) + set its own init pose; skip the IIWA-frame table_dist / startArmHigher surgery.
    if args_cli.sysid_gains:
        if not VEGA:
            raise SystemExit("--sysid_gains only applies to --task vega_hammer_retarget")
        from simtoolreal_lab.tasks.vega_sharpa_sysid import make_vega_robot_cfg_bimanual_sysid, ARM_DELAY_STEPS
        cfg.robot_cfg = make_vega_robot_cfg_bimanual_sysid()  # SYSID-tuned soft gains on BOTH arms (j7=j6)
        print(f"[sysid] swapped in sysid-tuned arm gains on both arms (real ~0.2 Hz BW); "
              f"command delay {ARM_DELAY_STEPS} steps @100 Hz NOT applied (implicit actuator has no delay buffer)")
    cfg.table_dist = 0.0 if VEGA else args_cli.table_dist  # move the table further from the robot (objects unchanged)
    # zero-shot deploy of the pretrained checkpoint (original obs/action convention, clean eval)
    cfg.pretrained_compat = True
    cfg.eval_append_expl_coef = True
    cfg.domain_randomization = False
    # optional faithful force/torque perturbations on the object (independent of full DR): the expert
    # then has to recover from random kicks while lifted -> more robust recovery data in the dataset.
    cfg.force_perturbation = args_cli.force_perturbation
    if args_cli.force_scale >= 0.0:                      # stronger/weaker kicks (cfg default 20.0)
        cfg.perturb_force_scale = args_cli.force_scale
    if args_cli.torque_scale >= 0.0:
        cfg.perturb_torque_scale = args_cli.torque_scale
    # tool-displacement perturbation (teleport the tool to simulate slips/drops -> recovery data)
    cfg.tool_displacement = args_cli.tool_displacement
    cfg.tool_displace_pregrasp = args_cli.tool_displace_pregrasp
    if args_cli.tool_displace_pos >= 0.0:
        cfg.tool_displace_pos = args_cli.tool_displace_pos
    if args_cli.tool_displace_rot >= 0.0:
        cfg.tool_displace_rot = args_cli.tool_displace_rot
    # joint-displacement perturbation (teleport the robot's joints -> recovery data; independent sampling)
    cfg.joint_displacement = args_cli.joint_displacement
    if args_cli.joint_displace_arm_scale >= 0.0:
        cfg.joint_displace_arm_scale = args_cli.joint_displace_arm_scale
    if args_cli.joint_displace_hand_scale >= 0.0:
        cfg.joint_displace_hand_scale = args_cli.joint_displace_hand_scale
    # random-action burst (DART/DAgger): robot executes random delta actions for a burst -> recovery data
    cfg.random_action = args_cli.random_action
    if args_cli.random_action_std >= 0.0:
        cfg.random_action_std = args_cli.random_action_std
    if args_cli.random_action_steps_std >= 0.0:
        cfg.random_action_steps_std = args_cli.random_action_steps_std
    if args_cli.random_action_prob >= 0.0:
        cfg.random_action_prob = args_cli.random_action_prob
    # optional goal-pose noise (task-agnostic decay: big early, ~0 at the trajectory end) -> the expert
    # tracks perturbed waypoints -> diversifies its path -> wider state coverage in the dataset.
    if args_cli.goal_noise:
        cfg.goal_noise_module = "simtoolreal_lab.tasks.screwdriver.goal_noise_decay"
        cfg.goal_noise_scale = args_cli.goal_noise_scale
    if args_cli.goal_diversify:   # diversify the trajectory shape (gen-param noise) + path (smooth offset)
        cfg.goal_diversify = True
        cfg.goal_diversify_scale = args_cli.goal_diversify_scale
        if args_cli.goal_diversify_offset_std >= 0.0:
            cfg.goal_diversify_offset_std = args_cli.goal_diversify_offset_std
    cfg.use_tolerance_curriculum = False
    cfg.success_steps = 1
    cfg.success_tolerance = args_cli.success_tolerance      # 0.01 -> clean hammering (policy tracks each goal)
    cfg.max_consecutive_successes = 0                       # goal-successes don't reset; the env resets on
    #                                                         nail-driven (success) or time_out (budget)
    cfg.episode_length_s = args_cli.max_ep_steps / 60.0     # time_out = the per-episode step budget
    # ORIGINAL eval/demo robot init (eval_interactive.py): startArmHigher + no reset noise -> the
    # hand starts above the table (no clipping) and the rollout is deterministic per seed.
    cfg.reset_dof_pos_noise_arm = 0.0
    cfg.reset_dof_pos_noise_fingers = 0.0
    cfg.reset_position_noise_x = cfg.reset_position_noise_y = cfg.reset_position_noise_z = 0.0
    if not VEGA:  # IIWA startArmHigher (the Vega cfg sets its own init joint_pos; no iiwa14_* joints)
        cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_2"] = 1.571 - math.radians(10)
        cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_4"] = 1.376 + math.radians(10)
    # recording camera: facing the robot, isolated to its own sub-env via cam_z_far + env_spacing
    cfg.per_env_camera = not args_cli.no_image
    cfg.cam_width, cfg.cam_height = args_cli.cam_width, args_cli.cam_height
    # For VEGA, KEEP the env cfg's own (approved front) camera unless the user explicitly passes --cam_eye.
    if not VEGA or args_cli.cam_eye != parser.get_default("cam_eye"):
        cfg.cam_eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
        cfg.cam_lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))
        cfg.cam_z_far = args_cli.cam_z_far
    if args_cli.wrist:                       # palm-facing wrist camera (2nd view)
        cfg.wrist_camera = True
        cfg.wrist_cam_width, cfg.wrist_cam_height = args_cli.wrist_cam_width, args_cli.wrist_cam_height
        cfg.wrist_cam_eye = tuple(float(v) for v in args_cli.wrist_eye.split(","))
        cfg.wrist_cam_lookat = tuple(float(v) for v in args_cli.wrist_lookat.split(","))
        cfg.wrist_cam_up = args_cli.wrist_up
        cfg.wrist_cam_focal = args_cli.wrist_focal
    if args_cli.dome_texture != "none":     # HDRI environment light = lighting + background
        cfg.dome_texture = DOME_HDRIS[args_cli.dome_texture]
        cfg.dome_intensity = args_cli.dome_intensity
    elif args_cli.ground_rgb:               # plain solid custom-color floor + backdrop (table kept default/white)
        cfg.ground_color = tuple(float(x) for x in args_cli.ground_rgb.split(","))
    elif args_cli.ground_texture == "white":  # plain solid-white floor + backdrop (no texture, no HDRI)
        cfg.ground_color = (1.0, 1.0, 1.0)
    elif args_cli.ground_texture != "none":  # else textured floor/backdrop (in-house Isaac Lab material)
        cfg.ground_mdl = GROUND_MDLS[args_cli.ground_texture]
        cfg.ground_texture_scale = (args_cli.ground_texture_scale, args_cli.ground_texture_scale)
    if args_cli.screw_rgb:   # recolor the driven screw/nail (hammer nail OR screwdriver screw)
        cfg.screw_color = tuple(float(x) for x in args_cli.screw_rgb.split(","))
    # task-specific goal + screw setup
    phys = not args_cli.kinematic_screw
    if args_cli.task in ("hammer", "vega_hammer_retarget"):
        cfg.use_fixed_goal_trajectory = False
        cfg.use_tighten_goals = True
        cfg.randomize_layout = True
        cfg.physical_screw = phys
        # DEFAULT hammer BC dataset = teleport-recovery perturbation ON (disable with --no_tool_displacement).
        # goal-noise / force-perturbation / action-noise stay OFF unless their flags are passed.
        # The Vega retarget implements tool teleport on the hammer object -> respect --tool_displacement
        # (default OFF for VEGA so the bimanual eval/deploy stays clean; the IIWA hammer stays default-ON).
        if VEGA:
            cfg.tool_displacement = args_cli.tool_displacement
            cfg.tool_displace_pregrasp = args_cli.tool_displace_pregrasp or args_cli.tool_displacement
        else:
            cfg.tool_displacement = not args_cli.no_tool_displacement
            cfg.tool_displace_pregrasp = not args_cli.no_tool_displacement
        cfg.screw_contact_clearance = -args_cli.hit_depth
        cfg.terminate_on_nail_driven = args_cli.success_joint   # episode ends (success) when screw seated
        if args_cli.strike_contact >= 0.0:                      # tighter: require head contacting the nail
            cfg.nail_strike_contact_dist = args_cli.strike_contact
        if args_cli.hand_reject >= 0.0:                         # rule out hand-nailing (0 = disable check)
            cfg.nail_hand_reject_dist = args_cli.hand_reject if args_cli.hand_reject > 0.0 else None
        if args_cli.nail_move_eps >= 0.0:                       # fail on nail-move-while-hammer-far (0 = disable)
            cfg.nail_move_eps = args_cli.nail_move_eps if args_cli.nail_move_eps > 0.0 else None
        cfg.sim.physx.gpu_collision_stack_size = 2 ** 30 if phys else 2 ** 28   # 2**30 for ~100 physical envs
    elif args_cli.task in ("screwdriver", "screwdriver043"):
        cfg.use_fixed_goal_trajectory = True
        cfg.use_tighten_goals = True
        cfg.randomize_layout = True
        cfg.trajectory_path = f"{TRAJ_ROOT}/screwdriver/044_screwdriver/tighten_screw.json"
        cfg.pretrained_object_scale = (2.5, 0.75, 0.75)
        cfg.physical_screw = True   # need the revolute screw_spin joint to measure rotation (success)
        cfg.terminate_on_screw_rotated = math.radians(args_cli.success_deg)  # success = screw tightened >= this
        cfg.sim.physx.gpu_collision_stack_size = 2 ** 30 if args_cli.task == "screwdriver043" else 2 ** 28
        if args_cli.task == "screwdriver":
            # use the SDF-collider screwdriver (thin blade physically ENTERS the slot) instead of the
            # convex-decomp default (blunt blade can't enter -> just shoves the head). This is the
            # collision config that worked in best10_screwdriver_6_14.
            cfg.object_cfg.spawn.usd_path = f"{REPO}/assets/usd/044_screwdriver_sdf/044_screwdriver_sdf.usd"
    else:  # claw_hammer (base simtoolreal): fixed swing_down trajectory goals
        cfg.use_fixed_goal_trajectory = True
        cfg.trajectory_path = f"{TRAJ_ROOT}/hammer/claw_hammer/swing_down.json"
    return gym_id, cfg


def _screw_offsets(base):
    """Per-corner offsets (4,3) for the SCREW keypoints (unit corners * small half-extents)."""
    c = torch.tensor(_KP_CORNERS, device=base.device, dtype=torch.float)
    return c * torch.tensor(_SCREW_HALF, device=base.device, dtype=torch.float)


def _find_screw_body(base):
    """Body index of the driven screw/nail in screw_asm (the body that rotates/sinks), or None."""
    asm = getattr(base, "screw_asm", None)
    if asm is None:
        return None
    names = list(asm.body_names)
    cand = [i for i, n in enumerate(names) if "screw" in n.lower() or "nail" in n.lower()]
    return cand[-1] if cand else (len(names) - 1)  # else the last (non-base) body


def _compute_keypoints(base, screw_off, screw_body_idx):
    """Env-local object-centric keypoints -> (N, 8, 3): 4 TOOL (object_keypoints) + 4 SCREW.

    Tool = the manipulated object's bbox corners (screwdriver / hammer). Screw = corners at the
    ACTUAL (dynamic) screw pose so the policy sees tightening progress: the physical screw_asm body
    (rotates for the screwdriver, sinks for the hammer), else the env-driven kinematic screw, else
    the nominal pose / goal keypoints."""
    tool_kp = base.object_keypoints                                  # (N,4,3) env-local
    eo = base.scene.env_origins
    asm = getattr(base, "screw_asm", None)
    if asm is not None and screw_body_idx is not None:               # physical: live rotating/sinking body
        spos = asm.data.body_pos_w[:, screw_body_idx] - eo
        squat = asm.data.body_quat_w[:, screw_body_idx]
    elif getattr(base, "screw", None) is not None:                   # kinematic: env-written driven pose
        spos = base.screw.data.root_pos_w - eo
        squat = base.screw.data.root_quat_w
    elif getattr(base, "screw_nom_pos", None) is not None:           # fallback: nominal (static)
        spos = base.screw_nom_pos - eo
        squat = base.screw_nom_quat
    else:
        return torch.cat([tool_kp, base.goal_keypoints], dim=1)      # no screw (base claw_hammer)
    screw_kp = spos.unsqueeze(1) + quat_apply(
        squat.unsqueeze(1).expand(-1, 4, -1), screw_off.unsqueeze(0).expand(base.num_envs, -1, -1))
    return torch.cat([tool_kp, screw_kp], dim=1)                     # (N,8,3)


_OVERLAY_FONT = None
def _overlay_perturb(frame, text="UNDER PERTURBATION"):
    """Return a copy of the RGB uint8 `frame` with a red border + `text` banner (top-center) burned in,
    to mark control steps where a perturbation was active. Uses PIL (verified available)."""
    global _OVERLAY_FONT
    from PIL import Image, ImageDraw, ImageFont
    H, W = frame.shape[:2]
    if _OVERLAY_FONT is None:
        try:
            _OVERLAY_FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", max(13, H // 18))
        except Exception:
            _OVERLAY_FONT = ImageFont.load_default()
    im = Image.fromarray(np.ascontiguousarray(frame)); d = ImageDraw.Draw(im)
    bw = max(3, H // 80)                                  # red attention border
    d.rectangle([0, 0, W - 1, H - 1], outline=(255, 40, 40), width=bw)
    try:
        bb = d.textbbox((0, 0), text, font=_OVERLAY_FONT); tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = d.textsize(text, font=_OVERLAY_FONT)
    x = (W - tw) // 2; y = max(2, H // 30)
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        d.text((x + dx, y + dy), text, font=_OVERLAY_FONT, fill=(0, 0, 0))   # black outline for legibility
    d.text((x, y), text, font=_OVERLAY_FONT, fill=(255, 80, 80))
    return np.asarray(im)


_VIDEO_FPS = 30
def _teleport_window(tele_full, win=60):
    """Expand a per-step teleport flag into a mask covering `win` frames AFTER each teleport event, so the
    'TELEPORT' overlay stays up for ~1 s of SIM time (win=60 control steps @ 60 Hz; matches the env's 1s=60-step
    convention -> ~2 s of 30-fps video, since frames are 1-per-control-step)."""
    t = np.asarray(tele_full).astype(bool)
    out = np.zeros(len(t), dtype=bool)
    for f0 in np.nonzero(t)[0]:
        out[f0:min(len(t), f0 + win)] = True
    return out


def _save_demo(g, idx, imgs, jss, acts, video_dir, save_video, wimgs=None, kps=None, kprs=None, pros=None, goals=None, teleports=None, burst=None):
    """Write one SUCCESSFUL demo to the HDF5 (+ an mp4 of its camera frames). RANDOM-ACTION-BURST steps
    (`burst`==1: the robot executed RANDOM, not expert, actions) are DROPPED from the HDF5 -- only the
    expert-controlled steps (incl. the post-burst recovery) are recorded, so every recorded action is the
    expert's. The VIDEO keeps EVERY frame (with an "UNDER PERTURBATION" overlay on the burst frames) for
    inspection. The kept `teleport` flag also marks the last step before each dropped-burst gap, so the
    trainers won't build an action chunk that spans the gap. `imgs` may be empty (--no_image)."""
    T = len(jss)
    ra = np.asarray(burst).astype(bool) if burst is not None else np.zeros(T, dtype=bool)
    keep = np.nonzero(~ra)[0]                                    # non-burst steps == the only ones in the HDF5
    tele_full = np.asarray(teleports, dtype="uint8") if teleports is not None else np.zeros(T, dtype="uint8")
    tele = tele_full[keep].copy() if len(keep) else tele_full[:0].copy()
    for k in range(len(keep) - 1):                               # flag last kept step before each dropped gap
        if keep[k + 1] - keep[k] > 1:                            # >=1 burst step was dropped between them
            tele[k] = 1                                          # -> chunk-loss mask stops here (no spanning)
    sel = lambda seq: [seq[t] for t in keep]                     # filter a per-step list to the kept steps
    jsa = np.stack(sel(jss)); aca = np.stack(sel(acts))          # (Tk,29)
    no_img = not imgs
    dg = g.create_group(f"demo_{idx}")
    if not no_img:
        imgk = np.stack(sel(imgs))                               # (Tk,H,W,3)
        dg.create_dataset("obs/image", data=imgk, dtype="uint8", chunks=(1,) + imgk.shape[1:],
                          compression="gzip", compression_opts=4)
    if wimgs is not None:
        wimgk = np.stack(sel(wimgs))   # palm-facing wrist view (kept steps)
        dg.create_dataset("obs/image_wrist", data=wimgk, dtype="uint8", chunks=(1,) + wimgk.shape[1:],
                          compression="gzip", compression_opts=4)
    if kps is not None:
        dg.create_dataset("obs/keypoints", data=np.stack(sel(kps)).astype("float32"), dtype="float32")  # (Tk,8,3)
    if kprs is not None:   # SimToolReal specialist: palm-relative keypoints
        dg.create_dataset("obs/keypoints_rel_palm", data=np.stack(sel(kprs)).astype("float32"), dtype="float32")
    if pros is not None:   # SimToolReal specialist: expert-faithful proprio
        dg.create_dataset("obs/proprio", data=np.stack(sel(pros)).astype("float32"), dtype="float32")  # (Tk,109)
    if goals is not None:  # goal-conditioned variant: object keypoints relative to the GOAL pose
        dg.create_dataset("obs/keypoints_rel_goal", data=np.stack(sel(goals)).astype("float32"), dtype="float32")
    dg.create_dataset("obs/joint_pos", data=jsa, dtype="float32")
    dg.create_dataset("actions", data=aca, dtype="float32")
    dg.create_dataset("teleport", data=tele)                     # (Tk,) gap markers + any tool/joint teleport flags
    dg.attrs["num_samples"] = int(jsa.shape[0])
    if save_video and not no_img:
        import imageio.v2 as imageio
        tele_win = _teleport_window(tele_full)                   # 'TELEPORT' banner held ~1 s after each event
        def render(fr, t):
            if t < len(tele_win) and tele_win[t]:
                return _overlay_perturb(fr, text="TELEPORT")     # 1 s teleport marker (tool OR joint)
            if ra[t]:
                return _overlay_perturb(fr, text="RANDOM ACTION")
            return fr
        w = imageio.get_writer(f"{video_dir}/demo_{idx:03d}.mp4", fps=_VIDEO_FPS, macro_block_size=None, codec="libx264")
        for t, fr in enumerate(imgs):                            # FULL rollout (burst frames included)
            w.append_data(render(fr, t))
        w.close()
        if wimgs is not None:   # also render the palm-facing wrist view
            ww = imageio.get_writer(f"{video_dir}/demo_{idx:03d}_wrist.mp4", fps=_VIDEO_FPS, macro_block_size=None, codec="libx264")
            for t, fr in enumerate(wimgs):
                ww.append_data(render(fr, t))
            ww.close()


def _save_fail_video(fail_dir, idx, imgs, wimgs=None, burst=None, teleports=None):
    """Write an mp4 of a FAILED episode's frames (for review; NOT added to the HDF5). Overlays a 1 s
    'TELEPORT' banner after each tool/joint teleport + 'UNDER PERTURBATION' on random-action-burst steps."""
    import imageio.v2 as imageio
    ra = np.asarray(burst).astype(bool) if burst is not None else np.zeros(len(imgs), bool)
    tele_win = _teleport_window(teleports) if teleports is not None else np.zeros(len(imgs), bool)
    def render(fr, t):
        if t < len(tele_win) and tele_win[t]:
            return _overlay_perturb(fr, text="TELEPORT")
        if t < len(ra) and ra[t]:
            return _overlay_perturb(fr, text="RANDOM ACTION")
        return fr
    w = imageio.get_writer(f"{fail_dir}/fail_{idx:03d}.mp4", fps=_VIDEO_FPS, macro_block_size=None, codec="libx264")
    for t, fr in enumerate(imgs):
        w.append_data(render(fr, t))
    w.close()
    if wimgs is not None:
        ww = imageio.get_writer(f"{fail_dir}/fail_{idx:03d}_wrist.mp4", fps=_VIDEO_FPS, macro_block_size=None, codec="libx264")
        for t, fr in enumerate(wimgs):
            ww.append_data(render(fr, t))
        ww.close()


def _viz_slot_box(base):
    """Spawn debug markers for the slot box (8 corner balls) + its computed center, the ACTUAL physical
    screw link, and the screwdriver tip; save the per-env-cam frame. Reveals whether the box is placed/
    sized correctly relative to the real screw + tip."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils.math import quat_apply as _qa
    import imageio.v2 as imageio
    c = base.cfg
    dev = base.device
    e = int(base.tip_dist.argmin())                          # the most-engaged env (closest tip)
    head = base.screw_head_world[e]                           # box CENTER (computed, from kinematic screw)
    th = float(base.screw_asm.data.joint_pos[e, 0])
    nom = base._nominal_slot[e]
    ct, st = math.cos(th), math.sin(th)
    along = torch.tensor([float(nom[0])*ct - float(nom[1])*st, float(nom[0])*st + float(nom[1])*ct, 0.0], device=dev)
    perp = torch.tensor([-float(along[1]), float(along[0]), 0.0], device=dev)
    zax = torch.tensor([0.0, 0.0, 1.0], device=dev)
    hl, hw, dn, tp = c.slot_half_length, c.slot_half_width, c.slot_depth, c.slot_top_tol
    zc = head + zax * ((tp - dn) / 2.0); hz = (dn + tp) / 2.0
    tip = base.object_pos[e] + base.scene.env_origins[e] + _qa(base.object_quat[e:e+1], base._tip_local.unsqueeze(0))[0]
    names = list(base.screw_asm.body_names)
    sidx = names.index("screw") if "screw" in names else len(names) - 1
    phys = base.screw_asm.data.body_pos_w[e, sidx]            # ACTUAL physical screw link
    print(f"[viz] env {e}: box_center(screw_head_world)={[round(float(x),3) for x in head]}  "
          f"physical_screw_link={[round(float(x),3) for x in phys]}  tip={[round(float(x),3) for x in tip]}", flush=True)
    print(f"[viz] |box_center - physical_screw| = {float((head-phys).norm())*1000:.0f}mm  "
          f"|tip - box_center| = {float((tip-head).norm())*1000:.0f}mm", flush=True)

    def ball(path, pos, col, r=0.003):
        s = sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col))
        s.func(path, s, translation=tuple(float(x) for x in pos))
    k = 0
    for sa in (-hl, hl):
        for sp in (-hw, hw):
            for sz in (-hz, hz):
                ball(f"/World/Viz/box_{k}", zc + along*sa + perp*sp + zax*sz, (0.1, 0.9, 0.1), 0.0016); k += 1
    ball("/World/Viz/center", head, (0.1, 0.4, 1.0))          # box center -> blue
    ball("/World/Viz/phys", phys, (1.0, 0.85, 0.0))           # physical screw -> yellow
    ball("/World/Viz/tip", tip, (1.0, 0.1, 0.1))              # screwdriver tip -> red
    pe = base.scene.sensors["per_env_cam"]
    for _ in range(8):
        base.sim.render(); pe.update(0.0, force_recompute=True)
    img = pe.data.output["rgb"][e, ..., :3].to(torch.uint8).cpu().numpy()
    out = f"{REPO}/videos/check_frames/slot_box_viz.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.imwrite(out, img)
    # close-up: crop to the green box-corner markers (so it works wherever the randomized screw landed)
    gm = (img[..., 1].astype(int) > 120) & (img[..., 0].astype(int) < 110) & (img[..., 2].astype(int) < 110)
    ys, xs = np.where(gm)
    if len(ys) > 4:
        pad = 45
        y0, y1 = max(0, ys.min()-pad), min(img.shape[0], ys.max()+pad)
        x0, x1 = max(0, xs.min()-pad), min(img.shape[1], xs.max()+pad)
        imageio.imwrite(f"{REPO}/videos/check_frames/slot_box_viz_closeup.png", img[y0:y1, x0:x1])
        print(f"[viz] close-up crop saved (green markers at px x[{x0}:{x1}] y[{y0}:{y1}])", flush=True)
    print(f"[viz] saved {out}  (green=slot box, blue=box center, yellow=physical screw, red=tip)", flush=True)


def main():
    torch.manual_seed(args_cli.seed)
    gym_id, cfg = make_cfg()
    N, TARGET, MAXEP = args_cli.num_envs, args_cli.num_demos, args_cli.max_ep_steps
    H, W = args_cli.cam_height, args_cli.cam_width
    # the per-env success flag the env sets in _get_dones (and the terminal cfg that must be configured):
    #   hammer -> nail_driven (terminate_on_nail_driven);  screwdriver -> screw_rotated (terminate_on_screw_rotated)
    if args_cli.task in ("screwdriver", "screwdriver043"):
        success_attr, term_cfg = "screw_rotated", "terminate_on_screw_rotated"
    else:
        success_attr, term_cfg = "nail_driven", "terminate_on_nail_driven"
    hammer_diag = (success_attr == "nail_driven")
    assert getattr(cfg, term_cfg, None) is not None, \
        f"success-collection needs the env's terminal condition '{term_cfg}' set (task={args_cli.task})"

    agent_cfg = build_agent_cfg(N)
    env = gym.make(gym_id, cfg=cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, "cuda:0", math.inf, math.inf)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.has_batch_dimension = True

    base = env.unwrapped
    # bimanual: a 2nd policy instance (own rnn state, same checkpoint) drives the LEFT arm/hand to hold
    # the thread_tester. The env's _step_left_instance runs it internally each step.
    player_left = None
    if getattr(base.cfg, "left_hold", False) and hasattr(base, "attach_left_policy"):
        player_left = runner.create_player()
        player_left.restore(args_cli.checkpoint)
        player_left.has_batch_dimension = True
        if player_left.is_rnn:
            player_left.init_rnn()
        base.attach_left_policy(player_left)
        print("[collect] bimanual: attached LEFT-hand thread_tester-hold policy instance", flush=True)
    screw_off = _screw_offsets(base)   # for the screw keypoints
    screw_body_idx = _find_screw_body(base)
    if screw_body_idx is not None:
        print(f"[collect] screw_asm bodies={list(base.screw_asm.body_names)} -> driven screw body idx={screw_body_idx}", flush=True)
    NO_IMG = args_cli.no_image
    pe_cam = None if NO_IMG else base.scene.sensors["per_env_cam"]
    WRIST = args_cli.wrist and not NO_IMG
    wrist_cam = base.scene.sensors["wrist_cam"] if WRIST else None
    STR = args_cli.simtoolreal or (args_cli.task == "hammer" and not args_cli.no_simtoolreal)   # default ON for hammer

    out = args_cli.out or f"{REPO}/datasets/{args_cli.task}_bc_success.hdf5"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    video_dir = f"{REPO}/videos/{args_cli.task}_bc_demos"
    fail_dir = f"{REPO}/videos/{args_cli.task}_bc_demos_fail"
    save_video = not args_cli.no_video
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        for fcl in os.listdir(video_dir):
            if fcl.endswith(".mp4"):
                os.remove(os.path.join(video_dir, fcl))
        if args_cli.save_failures:
            os.makedirs(fail_dir, exist_ok=True)
            for fcl in os.listdir(fail_dir):
                if fcl.endswith(".mp4"):
                    os.remove(os.path.join(fail_dir, fcl))
    crit = (f"screw_rotated>={args_cli.success_deg}deg" if success_attr == "screw_rotated"
            else f"nail_joint<={args_cli.success_joint}")
    print(f"[collect] target={TARGET} SUCCESSFUL demos | num_envs={N} max_ep={MAXEP} "
          f"success: {crit} | image {W}x{H} -> {out}", flush=True)

    f = h5py.File(out, "w")
    g = f.create_group("data")
    g.attrs["task"] = args_cli.task
    g.attrs["env"] = gym_id
    g.attrs["policy"] = args_cli.checkpoint
    g.attrs["fps"] = 60
    g.attrs["control_dt"] = 1.0 / 60.0
    g.attrs["resolution_hw"] = np.array([H, W], dtype=np.int32)
    g.attrs["joint_names"] = json.dumps(list(getattr(base.cfg, "joint_names", None) or JOINT_NAMES_ISAACGYM))
    g.attrs["action_space"] = "delta_joint_position: commanded joint target - current joint pos, 29-dof (arm0:7 + fingers7:29), canonical order"
    g.attrs["success_criterion"] = f"nail prismatic joint <= {args_cli.success_joint} (screw seated in the hole)"
    g.attrs["cam_eye"] = np.array(cfg.cam_eye, dtype=np.float32)
    g.attrs["cam_lookat"] = np.array(cfg.cam_lookat, dtype=np.float32)
    g.attrs["has_wrist_cam"] = bool(args_cli.wrist)
    g.attrs["keypoints"] = "obs/keypoints (T,8,3) env-local: 4 TOOL (object) keypoints + 4 SCREW keypoints"
    g.attrs["action_noise"] = float(args_cli.action_noise)
    if STR:
        g.attrs["simtoolreal"] = ("obs/keypoints_rel_palm (T,8,3) palm-relative keypoints + "
                                  "obs/proprio (T,109)=joint_pos29+joint_vel29+prev_targets29+palm_pos3+palm_rot4+fingertip_rel_palm15")

    # per-env episode buffers (variable length; flushed on success, dropped on failure)
    bimg = [[] for _ in range(N)]
    bjs = [[] for _ in range(N)]
    bact = [[] for _ in range(N)]
    bwrist = [[] for _ in range(N)]
    bkp = [[] for _ in range(N)]
    bkpp = [[] for _ in range(N)]   # SimToolReal: palm-relative keypoints
    bpro = [[] for _ in range(N)]   # SimToolReal: proprio (109)
    bgoal = [[] for _ in range(N)]  # goal-conditioned variant: keypoints_rel_goal (12)
    btp = [[] for _ in range(N)]    # per-step teleport flag (1 if the tool teleported THIS step) for chunk-loss masking
    bra = [[] for _ in range(N)]    # per-step random-action-burst flag (1 = robot executing random actions) -> DROPPED from the HDF5

    obs = env.reset()
    o = obs["obs"]
    if not NO_IMG:
        for _ in range(max(1, args_cli.warmup)):   # RTX texture warmup before recording
            base.sim.render()
    base.scene.update(base.physics_dt)
    if player.is_rnn:
        player.init_rnn()

    saved = 0
    fail_saved = 0
    attempted = 0
    step = 0
    EP_CAP = args_cli.num_episodes   # stop after this many TOTAL episodes (0 = no cap)
    # safety cap: enough steps to reach TARGET demos even at a LOW success rate. attempts ~= TARGET/rate,
    # waves ~= attempts/N, steps ~= waves*MAXEP -> TARGET*MAXEP/(N*rate). Use a 0.10 min-rate floor so a
    # low-yield task (e.g. screwdriver ~19%) isn't cut short; floored at MAXEP*8. Loop still exits at TARGET.
    STEP_CAP = args_cli.step_cap if args_cli.step_cap > 0 else max(MAXEP * 8, int(TARGET * MAXEP / max(N, 1) / 0.10))
    # diagnostic (screwdriver): track per-env max screw rotation this episode, to see how close the
    # tighten gets to the success threshold (and pick the budget / threshold).
    screw_diag = (success_attr == "screw_rotated")
    ep_max_cw = torch.full((N,), -1e9, device=base.device)   # max CLOCKWISE (tightening) rotation, deg
    ep_min_tip = torch.full((N,), 9.9, device=base.device)   # min tip-to-head dist this episode, m
    ep_engaged = torch.zeros(N, dtype=torch.bool, device=base.device)  # tip-in-slot ever this episode
    # slot-box coords (m) captured at the DEEPEST engagement (min tip_dist) -- for calibrating the box
    ep_along = torch.zeros(N, device=base.device); ep_across = torch.zeros(N, device=base.device)
    ep_depth = torch.zeros(N, device=base.device)
    # box state AT the MOST-ROTATED step (is the blade still in the box when the screw is most turned?)
    ep_mc_inbox = torch.zeros(N, dtype=torch.bool, device=base.device)
    ep_mc_across = torch.zeros(N, device=base.device); ep_mc_depth = torch.zeros(N, device=base.device)
    while saved < TARGET and step < STEP_CAP and (EP_CAP <= 0 or attempted < EP_CAP):
        # state S_t: image + current joint pos (BEFORE this step's action)
        rgb = None if NO_IMG else pe_cam.data.output["rgb"][..., :3]
        wrgb = wrist_cam.data.output["rgb"][..., :3] if WRIST else None
        js = base.joint_pos.clone()
        kp = _compute_keypoints(base, screw_off, screw_body_idx)     # (N,8,3) tool+screw keypoints, same instant as rgb/js
        if STR:   # SimToolReal-specialist obs at the SAME pre-step instant (palm-rel keypoints + proprio)
            kp_rel_t, proprio_t = compute_simtoolreal_obs(base, screw_off, screw_body_idx)
            goal_t = compute_goal_rel(base)   # (N,12) tool keypoints rel-goal (the teacher's goal signal)
        succ_pre = base.successes.clone() if screw_diag else None  # goal index BEFORE the (reset-on-done) step
        inbox_pre = base._inbox_max_cw.clone() if screw_diag else None  # max in-slot cw rotation so far (rad)
        nail_joint_pre = base.screw_asm.data.joint_pos[:, 0].clone() if (hammer_diag and base.screw_asm is not None) else None  # nail depth BEFORE reset
        displace_pre = base.tool_displace_events.clone() if base.cfg.tool_displacement else None  # teleports this episode (BEFORE reset)
        jdisplace_pre = base.joint_displace_events.clone() if base.cfg.joint_displacement else None  # joint teleports this episode
        rburst_pre = base.random_action_events.clone() if base.cfg.random_action else None  # random-action bursts this episode
        a = player.get_action(o, is_deterministic=True)
        if args_cli.action_noise > 0.0:              # DART-lite: noise -> off-distribution states -> expert recovers (recorded)
            a = (a + args_cli.action_noise * torch.randn_like(a)).clamp(-1.0, 1.0)
        obs, _, done, _ = env.step(a)                # auto-resets done envs (terminated=nail-driven, or time_out)
        o = obs["obs"]
        # record the EXPERT's intended delta (expert_targets == cur_targets except on a joint-teleport step,
        # where cur_targets is overridden to HOLD the teleported config -> expert_targets keeps the action clean)
        delta = base.expert_targets - js
        success = getattr(base, success_attr).clone()  # env set this in _get_dones this step (task-specific)
        if screw_diag:
            cw_deg = base.screw_cw_rot * (180.0 / math.pi)
            newmax = cw_deg > ep_max_cw                               # capture box state AT the most-rotated step
            ep_mc_inbox = torch.where(newmax, base.tip_in_slot, ep_mc_inbox)
            ep_mc_across = torch.where(newmax, base.slot_across, ep_mc_across)
            ep_mc_depth = torch.where(newmax, base.slot_depth, ep_mc_depth)
            ep_max_cw = torch.maximum(ep_max_cw, cw_deg)
            newmin = base.tip_dist < ep_min_tip                       # capture slot coords at the deepest engagement
            ep_along = torch.where(newmin, base.slot_along, ep_along)
            ep_across = torch.where(newmin, base.slot_across, ep_across)
            ep_depth = torch.where(newmin, base.slot_depth, ep_depth)
            ep_min_tip = torch.minimum(ep_min_tip, base.tip_dist)
            ep_engaged = ep_engaged | base.tip_in_slot
            if args_cli.viz_slot_box and base.screw_asm is not None:
                if bool((base.tip_dist < 0.010).any()) or step >= MAXEP - 20:  # capture at engagement (tip near screw)
                    _viz_slot_box(base)
                    break
        rgb_np = None if NO_IMG else rgb.to(torch.uint8).cpu().numpy()
        wrgb_np = wrgb.to(torch.uint8).cpu().numpy() if WRIST else None
        js_np = js.cpu().numpy()
        d_np = delta.cpu().numpy()
        kp_np = kp.cpu().numpy()
        kpr_np = kp_rel_t.cpu().numpy() if STR else None
        pro_np = proprio_t.cpu().numpy() if STR else None
        goal_np = goal_t.cpu().numpy() if STR else None
        # per-step teleport flag (set during env.step's _pre_physics_step): True if the tool teleported
        # THIS step -> the NEXT action onward is a recovery the chunk's input obs couldn't predict.
        tp_np = base._teleported_this_step.cpu().numpy()
        # per-step random-action-burst flag: this step the robot executed RANDOM (not expert) actions ->
        # the BC dataset DROPS these (only expert-controlled steps recorded); the video keeps them (overlay).
        ra_np = base._random_action_this_step.cpu().numpy()
        for i in range(N):
            if not NO_IMG:
                bimg[i].append(rgb_np[i])
            bjs[i].append(js_np[i]); bact[i].append(d_np[i]); bkp[i].append(kp_np[i])
            btp[i].append(tp_np[i]); bra[i].append(ra_np[i])
            if WRIST:
                bwrist[i].append(wrgb_np[i])
            if STR:
                bkpp[i].append(kpr_np[i]); bpro[i].append(pro_np[i]); bgoal[i].append(goal_np[i])
        done_b = done.bool()
        for i in torch.nonzero(done_b).flatten().tolist():
            attempted += 1
            if screw_diag:
                print(f"[collect]   episode-end env {i}: cw_rot={ep_max_cw[i]:.0f}deg "
                      f"inslot_cw={float(inbox_pre[i])*180/math.pi:.0f}deg "
                      f"min_tip_dist={ep_min_tip[i]*1000:.0f}mm slot[along={ep_along[i]*1000:.0f} "
                      f"across={ep_across[i]*1000:.0f} depth={ep_depth[i]*1000:.0f}]mm "
                      f"tip_in_slot={bool(ep_engaged[i])} goals={int(succ_pre[i])}/{base._traj_T} "
                      f"atmaxcw[inbox={bool(ep_mc_inbox[i])} across={ep_mc_across[i]*1000:.0f} "
                      f"depth={ep_mc_depth[i]*1000:.0f}]mm "
                      f"success={bool(success[i])} ({len(bjs[i])} steps)", flush=True)
                ep_max_cw[i] = -1e9; ep_min_tip[i] = 9.9; ep_engaged[i] = False
                ep_along[i] = 0.0; ep_across[i] = 0.0; ep_depth[i] = 0.0
                ep_mc_inbox[i] = False; ep_mc_across[i] = 0.0; ep_mc_depth[i] = 0.0
            if hammer_diag:
                nj = float(nail_joint_pre[i]) * 1000 if nail_joint_pre is not None else float("nan")
                dsp = ""
                if displace_pre is not None:
                    sd = float(base.nail_since_displace[i]); sd_s = "inf" if sd > 1e8 else f"{int(sd)}"
                    dsp = f" displace={int(displace_pre[i])} since_disp={sd_s}"
                if jdisplace_pre is not None:
                    dsp += f" jdisplace={int(jdisplace_pre[i])}"
                if rburst_pre is not None:
                    dsp += f" rburst={int(rburst_pre[i])}"
                mf = f" move_far={float(base.nail_move_far_log[i]) * 1000:.2f}mm" if hasattr(base, "nail_move_far_log") else ""
                print(f"[collect]   episode-end env {i}: nail_joint={nj:.1f}mm "
                      f"strike_dist={float(base.nail_strike_dist[i]) * 1000:.0f}mm "
                      f"hand_dist={float(base.nail_hand_dist[i]) * 1000:.0f}mm{dsp}{mf} "
                      f"success={bool(success[i])} ({len(bjs[i])} steps)", flush=True)
            if bool(success[i]) and saved < TARGET:
                vid = save_video and (args_cli.video_demos < 0 or saved < args_cli.video_demos)
                _save_demo(g, saved, bimg[i], bjs[i], bact[i], video_dir, vid, bwrist[i] if WRIST else None,
                           bkp[i], bkpp[i] if STR else None, bpro[i] if STR else None, bgoal[i] if STR else None,
                           teleports=btp[i], burst=bra[i])
                saved += 1
                f.flush()   # crash-resilience: persist each demo so a kill can't corrupt the whole file
                print(f"[collect]  + demo {saved}/{TARGET}  (env {i}, {len(bjs[i])} steps, {int(np.sum(bra[i]))} burst steps dropped)", flush=True)
            elif (args_cli.save_failures and not bool(success[i]) and save_video
                  and (args_cli.fail_videos < 0 or fail_saved < args_cli.fail_videos)):
                _save_fail_video(fail_dir, fail_saved, bimg[i], bwrist[i] if WRIST else None, burst=bra[i], teleports=btp[i])
                fail_saved += 1
                print(f"[collect]  - fail {fail_saved}  (env {i}, {len(bjs[i])} steps)", flush=True)
            bimg[i] = []; bjs[i] = []; bact[i] = []; bwrist[i] = []; bkp[i] = []; bkpp[i] = []; bpro[i] = []; bgoal[i] = []; btp[i] = []; bra[i] = []   # clear
        if player.is_rnn and bool(done_b.any()):       # reset the LSTM state for reset envs
            for s in player.states:
                s[:, done_b, :] = 0.0
            if player_left is not None and player_left.is_rnn:   # same for the LEFT-hold policy
                for s in player_left.states:
                    s[:, done_b, :] = 0.0
        if bool(done_b.any()):
            # FRAME ALIGNMENT: env.step renders DURING the physics substeps, but auto-resets
            # (_reset_idx) run AFTER those renders. So a just-reset env's joint_pos/obs are the new
            # episode's but its camera image is STALE (the old episode's last frame). Re-render + refresh
            # the sensors here so the NEXT captured frame is the new episode's actual observation,
            # keeping (image, joint_pos) aligned with the expert action recorded for that step.
            if not NO_IMG:   # the re-render is only to refresh the camera; state fields don't need it
                base.sim.render()
                base.scene.update(base.physics_dt)
        step += 1
        if step % 50 == 0:
            print(f"[collect] step={step} saved={saved}/{TARGET} attempted={attempted} "
                  f"success_rate={saved / max(1, attempted):.0%}", flush=True)

    g.attrs["num_demos"] = saved
    f.close()
    print(f"[collect] DONE: {saved} successful demos of {attempted} attempts "
          f"(success_rate {saved / max(1, attempted):.0%}) -> {out}", flush=True)
    if save_video:
        print(f"[collect] videos -> {video_dir}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
