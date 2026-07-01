"""Zero-shot deploy of the ORIGINAL pretrained checkpoint on the HAMMER task, following the
nail-in goal trajectory (nail_traj) end-to-end (the policy controls the robot; no teleporting).

The pretrained policy was trained on the claw_hammer (this task's tool); the thread_test + screw
are not in its observation, so it grasps the hammer and tracks the per-env nail goals, which advance
on success (lift -> reorient -> over the nail -> hit). pretrained_compat=True makes obs/action match
the original convention so the checkpoint runs bit-for-bit.

Modes:
  --video         : single-env clip via the viewer camera.
  --per_env_cam   : MANY envs at once (each a randomized layout = a 'seed'); one camera per env.
                    Tracks peak goals reached per env and keeps the BEST --best clips.

Run (50 seeds, best 10):
  cd IsaacLab && ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/deploy_hammer.py \
      --headless --num_envs 50 --per_env_cam --best 10 --steps 1000
"""

import argparse
import math

from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"
DEFAULT_CONFIG = f"{ORIG_REPO}/pretrained_policy/config.yaml"
DEFAULT_CKPT = f"{ORIG_REPO}/pretrained_policy/model.pth"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--orig_config", type=str, default=DEFAULT_CONFIG)
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
parser.add_argument("--steps", type=int, default=1000, help="rollout length (control steps)")
parser.add_argument("--success_tolerance", type=float, default=0.01, help="goal-reach tolerance (-> kp_tol = tol*1.5); advances to the next nail goal. 0.01 (kp_tol 1.5cm) = the training-tightest; tighter than this makes the policy track each goal to the hit pose so the tip reaches the screw (a loose value lets goals race ahead -> the hammer hovers, never contacts).")
parser.add_argument("--physical_screw", action="store_true", help="dynamic prismatic nail (hammer drives it); default = kinematic raised nail")
parser.add_argument("--hit_depth", type=float, default=0.04, help="how far BELOW the screw head the hit goal drives the tip (m). The zero-shot policy undershoots ~4cm, so overshooting the goal by this much makes the tip actually reach/contact the screw. Larger -> harder hit / deeper drive.")
parser.add_argument("--closed_loop", action="store_true", help="closed-loop goals: each step, re-aim the strike at the nail's CURRENT head (which sinks as it's driven), so repeated strikes keep driving it in (vs the open-loop trajectory that targets the original head).")
parser.add_argument("--pin", action="store_true", help="pin the bar/nail layout (deterministic); hammer still sampled clear of the bar")
parser.add_argument("--video", action="store_true", help="single-env viewer-camera clip")
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--per_env_cam", action="store_true", help="one camera per env; record a clip per env (keeps the best --best)")
parser.add_argument("--best", type=int, default=10, help="(per_env_cam) how many top clips (by peak goals reached) to keep")
parser.add_argument("--cam_width", type=int, default=1280)
parser.add_argument("--cam_height", type=int, default=800)
parser.add_argument("--per_env_cam_fps", type=int, default=60)
# camera faces the robot along the +y axis: robot base is at y=0.8 facing -y, so the camera sits
# in FRONT of it (low y, centered x) looking toward +y (down the y axis) at the manipulation+robot.
parser.add_argument("--cam_eye", type=str, default="0.0,-0.65,0.85")
parser.add_argument("--cam_lookat", type=str, default="0.0,0.30,0.55")
parser.add_argument("--overview", action="store_true", help="record a SINGLE zoomed-out video of the WHOLE env grid (all envs at once) from a world-frame viewport, instead of per-env clips. Use with --num_envs 25 (5x5 grid).")
parser.add_argument("--overview_eye", type=str, default="0.0,-6.5,7.0", help="(--overview) world-frame camera eye, pulled back+up to frame the whole grid")
parser.add_argument("--overview_lookat", type=str, default="0.0,0.8,0.1", help="(--overview) world-frame look-at (grid center)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--diag", action="store_true", help="log, during the rollout (env 0), the GOAL tip vs screw head (is the goal correct?) and the HAMMER tip vs screw head (does it actually reach the screw?)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video or args_cli.per_env_cam or args_cli.overview:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import simtoolreal_lab.tasks  # noqa: E402, F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402

RESULT_FILE = "/home/cning/simtoolreal_isaaclab/deploy_hammer_result.txt"
LOG_DIR = "/home/cning/simtoolreal_isaaclab/logs/simtoolreal"
VIDEO_DIR = "/home/cning/simtoolreal_isaaclab/videos"
TASK = "Isaac-SimToolReal-Hammer-Direct-v0"


def build_agent_cfg(num_envs: int) -> dict:
    with open(args_cli.orig_config) as f:
        full = yaml.safe_load(f)
    params = full["train"]["params"]
    params["seed"] = args_cli.seed
    params["load_checkpoint"] = True
    params["load_path"] = args_cli.checkpoint
    c = params["config"]
    c["name"] = "00_pretrained_deploy"
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


def main():
    import torch

    torch.manual_seed(args_cli.seed)
    cfg = HammerEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.pretrained_compat = True
    cfg.eval_append_expl_coef = True
    cfg.domain_randomization = False
    cfg.use_tolerance_curriculum = False
    cfg.use_fixed_goal_trajectory = False
    cfg.use_tighten_goals = True
    # CLOSED-LOOP: re-aim every step at the nail's CURRENT (sinking) head so repeated strikes keep
    # driving it in (HammerEnv._set_responsive_goal). Overrides the open-loop per_env_goals.
    cfg.responsive_goals = args_cli.closed_loop
    cfg.randomize_layout = True
    cfg.success_tolerance = args_cli.success_tolerance
    cfg.success_steps = 1
    cfg.physical_screw = args_cli.physical_screw
    # OVERSHOOT: drive the hit goal hit_depth BELOW the screw head so the policy's ~4cm undershoot
    # still lands the tip on the screw (instead of hovering above it).
    cfg.screw_contact_clearance = -args_cli.hit_depth
    # the screw uses an SDF collider (many contacts); bump the GPU collision stack so it doesn't
    # overflow ("Contacts have been dropped") across many envs.
    cfg.sim.physx.gpu_collision_stack_size = 2 ** 28
    if args_cli.physical_screw:
        cfg.sim.physx.gpu_collision_stack_size = 2 ** 29
    if args_cli.pin:
        cfg.layout_threadtest_center_x_range = (0.18, 0.18)
        cfg.layout_threadtest_center_y_range = (0.0, 0.0)
        cfg.layout_yaw_range = (0.0, 0.0)
        cfg.layout_screwdriver_x_range = (-0.12, -0.06)
        cfg.layout_screwdriver_y_range = (-0.04, 0.04)

    import simtoolreal_lab.tasks.hammer.nail_traj as nail_traj
    T = nail_traj.T
    cfg.max_consecutive_successes = 0       # no success-reset -> full-length episode (longer)
    cfg.episode_length_s = (args_cli.steps + 120) / 60.0

    # Match the ORIGINAL eval/demo robot init (dextoolbench/eval_interactive.py): deterministic arm
    # pose with NO reset DOF noise, plus startArmHigher so the hand starts well ABOVE the table --
    # otherwise the lower default pose + 0.1 rad arm noise makes the arm clip the table at reset.
    cfg.reset_dof_pos_noise_arm = 0.0
    cfg.reset_dof_pos_noise_fingers = 0.0
    cfg.reset_position_noise_x = 0.0
    cfg.reset_position_noise_y = 0.0
    cfg.reset_position_noise_z = 0.0
    cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_2"] = 1.571 - math.radians(10)  # startArmHigher
    cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_4"] = 1.376 + math.radians(10)  # startArmHigher

    if args_cli.per_env_cam:
        cfg.per_env_camera = True
        cfg.cam_width, cfg.cam_height = args_cli.cam_width, args_cli.cam_height
        cfg.cam_eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
        cfg.cam_lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))
    if args_cli.video:
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.resolution = (1280, 720)
        cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
        cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))
    if args_cli.overview:
        # single world-frame viewport pulled back+up to frame the WHOLE env grid (all envs at once)
        cfg.viewer.origin_type = "world"
        cfg.viewer.resolution = (1920, 1080)
        cfg.viewer.eye = tuple(float(v) for v in args_cli.overview_eye.split(","))
        cfg.viewer.lookat = tuple(float(v) for v in args_cli.overview_lookat.split(","))

    rec = args_cli.video or args_cli.overview
    agent_cfg = build_agent_cfg(args_cli.num_envs)
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array" if rec else None)
    if rec:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder=VIDEO_DIR, step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            name_prefix="deploy_hammer" + ("_overview" if args_cli.overview
                                           else ("_physical" if args_cli.physical_screw else "")),
            disable_logger=True)
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
    rf = open(RESULT_FILE, "w")

    def emit(m):
        print(m, flush=True)
        rf.write(m + "\n"); rf.flush(); os.fsync(rf.fileno())

    emit(f"DEPLOY_HAMMER pretrained={args_cli.checkpoint}")
    emit(f"num_envs={base.num_envs} steps={args_cli.steps} T(goals)={T} "
         f"physical_screw={args_cli.physical_screw} success_tol={args_cli.success_tolerance}")

    # per-env cameras: one mp4 writer per env
    pe_cam = pe_writers = None
    pe_dir = f"{VIDEO_DIR}/hammer_per_env"
    if args_cli.per_env_cam:
        import imageio.v2 as imageio
        pe_cam = base.scene.sensors["per_env_cam"]
        os.makedirs(pe_dir, exist_ok=True)
        for f in os.listdir(pe_dir):
            if f.endswith(".mp4"):
                os.remove(os.path.join(pe_dir, f))
        pe_writers = {i: imageio.get_writer(f"{pe_dir}/env_{i:03d}.mp4", fps=args_cli.per_env_cam_fps,
                                            macro_block_size=None, codec="libx264") for i in range(base.num_envs)}
        emit(f"PER_ENV_CAM: {base.num_envs} cameras @ {args_cli.cam_width}x{args_cli.cam_height}")

    from isaaclab.utils.math import quat_apply  # noqa: E402
    TIP = base._tip_local
    min_htip = torch.full((base.num_envs,), 1e9, device=base.device)  # closest the hammer tip gets to the screw
    # how far the nail is driven (physical only): init joint - lowest joint reached (m). >0 => driven in.
    has_nail = args_cli.physical_screw and base.screw_asm is not None
    nail_lo = torch.zeros(base.num_envs, device=base.device)          # lowest (most-driven) joint pos seen
    if has_nail:
        nail_lo = base.screw_asm.data.joint_pos[:, 0].clone()
    nail_init = float(getattr(base.cfg, "nail_start_height", 0.0))

    STEPS = args_cli.steps
    peak_succ = torch.zeros(base.num_envs, device=base.device)   # max goals reached per env (the score)
    lifted_frames = torch.zeros(base.num_envs, device=base.device)
    obs = env.reset()
    o = obs["obs"]
    if args_cli.diag:   # verify the per-env GOAL poses actually put the hammer tip on the screw
        g0 = base.per_env_goals[0]
        head0 = base.screw_head_world[0] - base.scene.env_origins[0]
        gtips = g0[:, 0:3] + quat_apply(g0[:, [6, 3, 4, 5]], TIP.unsqueeze(0).expand(T, 3))
        d0 = torch.norm(gtips - head0, dim=-1)
        emit(f"  [goalcheck] per_env_goals[0] TIP->screw: overall min={d0.min().item()*100:.1f}cm @goal{int(d0.argmin())} "
             f"max={d0.max().item()*100:.1f}cm | HIT-phase(40..67) min={d0[40:].min().item()*100:.1f}cm mean={d0[40:].mean().item()*100:.1f}cm")
    if player.is_rnn:
        player.init_rnn()
    for t in range(STEPS):
        a = player.get_action(o, is_deterministic=True)
        obs, rew, done, info = env.step(a)
        o = obs["obs"]
        # hammer tip vs screw head (env-local), tracked for ALL envs
        eo = base.scene.env_origins
        head = base.screw_head_world - eo
        htip = base.object_pos + quat_apply(base.object_quat, TIP.unsqueeze(0).expand(base.num_envs, 3))
        min_htip = torch.minimum(min_htip, torch.norm(htip - head, dim=-1))
        if has_nail:
            nail_lo = torch.minimum(nail_lo, base.screw_asm.data.joint_pos[:, 0])
        if args_cli.diag and (t % 40 == 0 or t == STEPS - 1):
            gtip0 = base.goal_pos[0] + quat_apply(base.goal_quat[0:1], TIP.unsqueeze(0))[0]
            d_goal = (gtip0 - head[0]).norm().item()
            d_ham = (htip[0] - head[0]).norm().item()
            emit(f"  [diag t={t:4d}] goal#={int(base.successes[0].item()) % T:2d} "
                 f"GOALtip->screw={d_goal*100:5.1f}cm  HAMMERtip->screw={d_ham*100:5.1f}cm")
        if pe_cam is not None:
            rgb = pe_cam.data.output["rgb"]
            for i in pe_writers:
                pe_writers[i].append_data(rgb[i].cpu().numpy())
        if player.is_rnn and bool(done.any()):
            d = done.bool()
            for s in player.states:
                s[:, d, :] = 0.0
        peak_succ = torch.maximum(peak_succ, base.successes)
        lifted_frames += base.lifted_object.float()
        if t % 100 == 0 or t == STEPS - 1:
            emit(f"t={t:4d} peak_goals mean={peak_succ.mean().item():.1f} max={int(peak_succ.max().item())}/{T} "
                 f"lift_rate={base.lifted_object.float().mean().item():.2f}")

    if pe_writers is not None:
        for w in pe_writers.values():
            w.close()
        # rank envs by CLOSEST hammer-tip-to-screw contact (the whole point: did it hit the screw)
        order = torch.argsort(min_htip, descending=False).tolist()
        best_dir = f"{VIDEO_DIR}/hammer_best"
        os.makedirs(best_dir, exist_ok=True)
        for f in os.listdir(best_dir):
            if f.endswith(".mp4"):
                os.remove(os.path.join(best_dir, f))
        emit(f"=== BEST {args_cli.best} of {base.num_envs} (by closest hammer-tip->screw contact) ===")
        for rank, i in enumerate(order[:args_cli.best]):
            mm = int(min_htip[i].item() * 1000)
            g = int(peak_succ[i].item())
            src = f"{pe_dir}/env_{i:03d}.mp4"
            dst = f"{best_dir}/rank{rank+1:02d}_env{i:03d}_contact{mm:03d}mm_goals{g:03d}.mp4"
            shutil.copyfile(src, dst)
            emit(f"  rank {rank+1:2d}: env {i:3d}  closest_contact={mm:3d}mm  peak_goals={g:3d}/{T}  -> {os.path.basename(dst)}")
        emit(f"[best] {args_cli.best} clips -> {best_dir}")

    emit(f"FINAL peak_goals mean={peak_succ.mean().item():.2f} max={int(peak_succ.max().item())}/{T} "
         f"n_envs_reaching>=10={int((peak_succ >= 10).sum().item())}/{base.num_envs}")
    emit(f"HAMMER tip -> screw head CLOSEST over rollout: mean={min_htip.mean().item()*100:.1f}cm "
         f"min={min_htip.min().item()*100:.1f}cm  (==0 => the hammer face reaches the screw)")
    if has_nail:
        driven = (nail_init - nail_lo) * 100.0   # cm the nail was pushed below its raised start
        emit(f"NAIL DRIVEN (init {nail_init*100:.1f}cm raised -> lowest joint): mean={driven.mean().item():.2f}cm "
             f"max={driven.max().item():.2f}cm  n_envs_driven>=1cm={int((driven >= 1.0).sum().item())}/{base.num_envs}")
    rf.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
