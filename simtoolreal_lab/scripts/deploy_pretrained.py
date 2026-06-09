"""Zero-shot deploy of the ORIGINAL Isaac Gym SimToolReal pretrained checkpoint in Isaac Lab.

Unlike play.py (which evals a from-scratch Isaac Lab policy), this loads the original
`pretrained_policy/{config.yaml,model.pth}` and runs it directly in the Isaac Lab env with
`pretrained_compat=True`, so the obs/action are produced in the EXACT original convention:
  - object_scales = dextoolbench "scale given to policy" (claw_hammer = (2.5,0.5625,0.375));
  - palm_rot/object_rot in XYZW; joint_pos unscale + hand-action scale via Q_{LOWER,UPPER};
  - elevated goal target volume (targetVolumeMins/Maxs from the original config).
The network is built from the ORIGINAL config.yaml's train.params so the checkpoint loads
bit-for-bit (LSTM 1024 + MLP [1024,1024,512,512] + coef_cond sigma + 32-d expl embedding).

Run:
  cd IsaacLab && ./isaaclab.sh -p ~/simtoolreal_isaaclab/simtoolreal_lab/scripts/deploy_pretrained.py \
      --headless --num_envs 64 --delta
  # add --video --cam_env_index <i> for an RTX mp4 of one env
"""

import argparse
import math

from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"
DEFAULT_CONFIG = f"{ORIG_REPO}/pretrained_policy/config.yaml"
DEFAULT_CKPT = f"{ORIG_REPO}/pretrained_policy/model.pth"

ASSETS_USD = "/home/cning/simtoolreal_isaaclab/simtoolreal_lab/assets/usd"
TRAJ_ROOT = f"{ORIG_REPO}/dextoolbench/trajectories"
# DexToolBench objects: name -> (category, object_scale-given-to-policy). Scale = dextoolbench
# objects.py base-bbox * 25 (= bbox / object_base_size 0.04); NOT applied to the mesh.
OBJECTS = {
    "claw_hammer":       ("hammer",      (2.5, 0.5625, 0.375)),
    "long_screwdriver":  ("screwdriver", (2.5, 0.75, 0.75)),
    "short_screwdriver": ("screwdriver", (1.75, 0.875, 0.875)),
    "sharpie_marker":    ("marker",      (2.125, 0.55, 0.55)),
    "staples_marker":    ("marker",      (3.0, 0.45, 0.45)),
}

parser = argparse.ArgumentParser()
parser.add_argument("--object", type=str, default="claw_hammer", choices=list(OBJECTS), help="DexToolBench object to manipulate")
parser.add_argument("--demo_task", type=str, default="swing_down", help="trajectory task name (e.g. swing_down, spin_vertical, spin_horizontal, draw_smile, write_c)")
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-ClawHammer-Direct-v0")
parser.add_argument("--num_envs", type=int, default=96)  # multiple of 6 (SAPG blocks)
parser.add_argument("--orig_config", type=str, default=DEFAULT_CONFIG, help="original pretrained config.yaml (net arch)")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="original pretrained model.pth")
parser.add_argument("--delta", action="store_true", help="eval on elevated delta-goal distribution instead of the fixed swing_down trajectory")
parser.add_argument("--demo", action="store_true", help="reproduce the eval_interactive.py demo scenario exactly (fixed object init from trajectory start_pose, fixed-size-keypoint success @ tol 0.015, startArmHigher, no noise, 1 env)")
parser.add_argument("--steps", type=int, default=600, help="rollout length (control steps)")
parser.add_argument("--video", action="store_true", help="record an Omniverse RTX mp4 of the rollout")
parser.add_argument("--video_length", type=int, default=400, help="number of steps to record")
parser.add_argument("--cam_eye", type=str, default="-0.55,-0.45,0.80", help="render camera position (x,y,z), env-local")
parser.add_argument("--cam_lookat", type=str, default="0.0,0.20,0.62", help="render camera target (x,y,z), env-local")
parser.add_argument("--cam_env_index", type=int, default=0, help="render camera follows this env index")
parser.add_argument("--seed", type=int, default=0, help="seed for reproducible env resets across runs")
parser.add_argument("--zero_action", action="store_true", help="DEBUG: ignore policy, apply zero actions (isolate physics stability)")
parser.add_argument("--debug_steps", type=int, default=0, help="DEBUG: print per-component finite-status for the first N steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True  # rendering requires cameras

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab/simtoolreal_lab")

import simtoolreal_lab.tasks  # noqa: E402, F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

from simtoolreal_lab.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg  # noqa: E402

RESULT_FILE = "/home/cning/simtoolreal_isaaclab/deploy_result.txt"
LOG_DIR = "/home/cning/simtoolreal_isaaclab/logs/simtoolreal"

# Original SimToolReal targetVolumeMins/Maxs (env-local = world frame; robot base at y=0.8).
TARGET_VOLUME_MIN = (-0.35, -0.1, 0.68)
TARGET_VOLUME_MAX = (0.35, 0.2, 1.05)


def build_agent_cfg(num_envs: int) -> dict:
    """Load the ORIGINAL config.yaml and turn train.params into a self-contained rl_games cfg.

    The original yaml is full of unresolved OmegaConf ${...} interpolations; resolve the few
    the Runner/player actually touches to concrete values. The network section (mlp/rnn/
    coef_cond/expl embedding) is left UNCHANGED so the checkpoint state_dict matches.
    """
    with open(args_cli.orig_config) as f:
        full = yaml.safe_load(f)
    params = full["train"]["params"]
    params["seed"] = args_cli.seed
    params["load_checkpoint"] = True
    params["load_path"] = args_cli.checkpoint

    c = params["config"]
    c["name"] = "00_pretrained_deploy"  # SAPG fork parses int(name.split('_')[0]) -> must start with a number
    c["device_name"] = "cuda:0"
    c["device"] = "cuda:0"
    c["multi_gpu"] = False
    c["num_actors"] = num_envs
    # The env's action_space is Box(-inf, inf); rl_games' player rescale_actions() would map
    # [-1,1] onto those infinite bounds -> NaN. Disable it; the env already clamps actions to
    # [-1,1] in _pre_physics_step, so get_action returns mu directly. (Defaults to True.)
    c["clip_actions"] = False
    c["max_epochs"] = 1
    c["max_frames"] = 100
    # SAPG block size is only used by the trainer to partition envs; the player's net is built
    # with 6 blocks hardcoded (linspace(50,0,6)) + a 32-d embedding, independent of this value.
    # So any >=1 works for eval/demo (incl. num_envs=1, like the original demo's numEnvs=1).
    c["expl_coef_block_size"] = max(1, num_envs // 6)
    # minibatch must divide batch = num_actors * horizon_length; player doesn't train so just keep it valid.
    c["minibatch_size"] = num_envs
    if "central_value_config" in c:
        c["central_value_config"]["minibatch_size"] = num_envs
    c["train_dir"] = LOG_DIR
    # resolve the ${if:...pbt...} summary interpolations
    c["defer_summaries_sec"] = 5
    c["summaries_interval_sec_min"] = 5
    c["summaries_interval_sec_max"] = 300
    c.setdefault("player", {})
    c["player"]["games_num"] = 128
    c["player"]["deterministic"] = True
    return {"params": params}  # runner.load expects the top-level {"params": ...} wrapper


def main():
    import torch

    torch.manual_seed(args_cli.seed)
    if args_cli.demo:
        args_cli.num_envs = 1  # the original eval_interactive.py demo runs a single env
    env_cfg = SimToolRealEnvCfg()
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    # select the DexToolBench object + its trajectory task (default claw_hammer / swing_down)
    _cat, _scale = OBJECTS[args_cli.object]
    env_cfg.object_cfg.spawn.usd_path = f"{ASSETS_USD}/{args_cli.object}/{args_cli.object}.usd"
    env_cfg.pretrained_object_scale = _scale
    env_cfg.trajectory_path = f"{TRAJ_ROOT}/{_cat}/{args_cli.object}/{args_cli.demo_task}.json"
    # zero-shot deploy of the original checkpoint: original obs/action convention, clean eval
    env_cfg.pretrained_compat = True
    env_cfg.eval_append_expl_coef = True   # coef_cond: append exploit coef 0.0 at obs idx 140
    env_cfg.domain_randomization = False   # clean/deterministic eval
    env_cfg.use_tolerance_curriculum = False
    env_cfg.use_fixed_goal_trajectory = not args_cli.delta
    # elevated goal volume from the original config (forces lifting on the first goal)
    env_cfg.target_volume_min = TARGET_VOLUME_MIN
    env_cfg.target_volume_max = TARGET_VOLUME_MAX

    if args_cli.demo:
        # Reproduce eval_interactive.py exactly: fixed object init from the trajectory start_pose
        # (env loads it when demo_mode), fixed-size-keypoint success @ evalSuccessTolerance, no noise.
        import json as _json

        env_cfg.demo_mode = True
        env_cfg.use_fixed_goal_trajectory = True
        env_cfg.success_tolerance = 0.01   # evalSuccessTolerance (-> kp_tol = 0.01*1.5 = 0.015)
        env_cfg.success_steps = 1          # successSteps
        # deterministic init: no reset noise
        env_cfg.reset_position_noise_x = 0.0
        env_cfg.reset_position_noise_y = 0.0
        env_cfg.reset_position_noise_z = 0.0
        env_cfg.reset_dof_pos_noise_arm = 0.0
        env_cfg.reset_dof_pos_noise_fingers = 0.0
        env_cfg.reset_dof_vel_noise = 0.0
        # startArmHigher: arm joint_2 -= 10deg, joint_4 += 10deg (absolute, to avoid double-apply)
        env_cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_2"] = 1.571 - math.radians(10)
        env_cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_4"] = 1.376 + math.radians(10)
        # episode ends after all trajectory goals reached (max_consecutive_successes = #goals)
        with open(env_cfg.trajectory_path) as _f:
            env_cfg.max_consecutive_successes = len(_json.load(_f)["goals"])
        # per-episode time cap exceeds the requested rollout so it stays one continuous episode
        # (only resets early if all goals reached or the object is dropped).
        env_cfg.episode_length_s = (args_cli.steps + 120) / 60.0

    agent_cfg = build_agent_cfg(args_cli.num_envs)
    rl_device = "cuda:0"
    clip_obs = math.inf       # env already clamps obs to +/-10 (clampAbsObservations) internally
    clip_actions = math.inf   # env already clamps actions to [-1,1] in _pre_physics_step

    if args_cli.video:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = args_cli.cam_env_index
        env_cfg.viewer.resolution = (1280, 720)
        env_cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
        env_cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    video_env = None
    if args_cli.video:
        video_dir = "/home/cning/simtoolreal_isaaclab/videos"
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix="pretrained_" + ("delta" if args_cli.delta else f"{args_cli.object}_{args_cli.demo_task}"),
            disable_logger=True,
        )
        video_env = env  # keep a handle so we can flush the (possibly partial) video on early stop
        print(f"[video] recording up to {args_cli.video_length} steps -> {video_dir}")
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

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
        rf.write(m + "\n")
        rf.flush()
        os.fsync(rf.fileno())

    emit(f"DEPLOY pretrained={args_cli.checkpoint}")
    emit(f"mode={'delta' if args_cli.delta else 'swing_down_trajectory'} num_envs={base.num_envs} steps={args_cli.steps}")

    STEPS = args_cli.steps
    lift_frames = torch.zeros(base.num_envs, device=base.device)
    peak_succ = torch.zeros(base.num_envs, device=base.device)  # max consecutive trajectory goals reached, per env
    obs = env.reset()
    o = obs["obs"]
    if player.is_rnn:
        player.init_rnn()

    if args_cli.debug_steps > 0:
        # physical-sanity dump of the aligned obs for env 0 at reset
        with torch.no_grad():
            base._compute_intermediate_values()
            i = 0
            emit(f"[obscheck] object_pos(env-local)={base.object_pos[i].tolist()}")
            emit(f"[obscheck] object_quat(wxyz)={base.object_quat[i].tolist()} -> xyzw={base.object_quat[i][[1,2,3,0]].tolist()}")
            emit(f"[obscheck] object_scales={base.object_scales[i].tolist()}")
            emit(f"[obscheck] palm_center(env-local)={base.palm_center[i].tolist()}")
            emit(f"[obscheck] goal_pos={base.goal_pos[i].tolist()} goal_quat(wxyz)={base.goal_quat[i].tolist()}")
            emit(f"[obscheck] kp_rel_palm absmax={ (base.object_keypoints[i]-base.palm_center[i]).abs().max().item():.3f} "
                 f"kp_rel_goal absmax={(base.object_keypoints[i]-base.goal_keypoints[i]).abs().max().item():.3f}")
            emit(f"[obscheck] arm_q(deg)={[round(float(q)*57.2958,1) for q in base.joint_pos[i,:7]]}")
            emit(f"[obscheck] fingertip_rel_palm absmax={(base.fingertip_pos[i]-base.palm_center[i]).abs().max().item():.3f}")
        player.model.eval()
        with torch.no_grad():
            rms = getattr(player.model, "running_mean_std", None)
            emit(f"[introspect] reset obs finite={bool(torch.isfinite(o).all())} absmax={o.abs().max().item():.3f} shape={tuple(o.shape)}")
            if rms is not None:
                var = rms.running_var
                emit(f"[introspect] rms var: min={var.min().item():.3e}@{int(var.argmin())} max={var.max().item():.3e}; mean absmax={rms.running_mean.abs().max().item():.3f}; n_zerovar={int((var<1e-8).sum())}")
            try:
                normed = player.model.norm_obs(o.clone())
                bad = (~torch.isfinite(normed[0])).nonzero().flatten().tolist()
                emit(f"[introspect] normed finite={bool(torch.isfinite(normed).all())} absmax={normed.abs().max().item():.3e} argmax_dim={int(normed[0].abs().argmax())} nonfinite_dims={bad[:10]}")
            except Exception as e:
                emit(f"[introspect] norm_obs raised: {type(e).__name__}: {e}")
            try:
                inp = {"is_train": False, "prev_actions": None, "obs": o.clone(), "rnn_states": player.states}
                res = player.model(inp)
                mu, sg = res.get("mus"), res.get("sigmas")
                emit(f"[introspect] mu finite={bool(torch.isfinite(mu).all())} absmax={mu.abs().max().item():.3e}; sigma finite={bool(torch.isfinite(sg).all())} min={sg.min().item():.3e} max={sg.max().item():.3e}")
            except Exception as e:
                emit(f"[introspect] model fwd raised: {type(e).__name__}: {e}")
        if player.is_rnn:
            player.init_rnn()  # reset states corrupted by the introspection forward

    def dbg(tag, t):
        # report which physical quantities are non-finite + ranges (env 0 + global)
        jp = base.joint_pos
        emit(
            f"[dbg t={t} {tag}] "
            f"obj_pos_finite={bool(torch.isfinite(base.object_pos).all())} "
            f"obj_z0={base.object_pos[0,2].item():.3f} "
            f"jpos_finite={bool(torch.isfinite(jp).all())} jpos_absmax={jp.abs().max().item():.2f} "
            f"jvel_absmax={base.joint_vel.abs().max().item():.2f} "
            f"palm_finite={bool(torch.isfinite(base.palm_center).all())} "
            f"tgt_absmax={base.cur_targets.abs().max().item():.2f}"
        )

    for t in range(STEPS):
        if args_cli.zero_action:
            a = torch.zeros((base.num_envs, base.cfg.action_space), device=base.device)
        else:
            a = player.get_action(o, is_deterministic=True)
            if t < args_cli.debug_steps:
                emit(f"[dbg t={t} action] absmax={a.abs().max().item():.3f} mean={a.mean().item():.3f} finite={bool(torch.isfinite(a).all())}")
        obs, rew, done, info = env.step(a)
        o = obs["obs"]
        if t < args_cli.debug_steps:
            dbg("post_step", t)
            emit(f"[dbg t={t} obs] finite={bool(torch.isfinite(o).all())} absmax={o.abs().max().item():.2f}")
        if player.is_rnn and bool(done.any()):
            d = done.bool()
            for s in player.states:
                s[:, d, :] = 0.0
        lift_frames += base.lifted_object.float()
        peak_succ = torch.maximum(peak_succ, base.successes)
        if t % 60 == 0 or t == STEPS - 1:
            emit(
                f"t={t:3d} lift_rate={base.lifted_object.float().mean().item():.2f} "
                f"obj_z={base.object_pos[:,2].mean().item():.3f} "
                f"kp_dist_to_goal={base.keypoints_max_dist.mean().item():.3f} "
                f"succ_mean={base.successes.mean().item():.2f} succ_max={int(base.successes.max().item())}"
            )
        # demo: stop once the trajectory episode ends (all goals reached -> max_consecutive_successes
        # -> done). prev_episode_successes holds the pre-reset count (captured in env._reset_idx).
        # This stops the rollout/recording at completion instead of resetting and replaying.
        if args_cli.demo and bool(done.any()):
            ng = base.trajectory_goals.shape[0] if base.trajectory_goals is not None else 0
            done_goals = int(base.prev_episode_successes.max().item())
            peak_succ = torch.maximum(peak_succ, base.prev_episode_successes)
            emit(f"EPISODE_END t={t} goals_reached={done_goals}/{ng} (all_goals={'yes' if done_goals >= ng else 'no'})")
            break
    emit(
        f"FINAL succ_mean={base.successes.mean().item():.3f} "
        f"succ_max={int(base.successes.max().item())} "
        f"lift_rate={base.lifted_object.float().mean().item():.3f} "
        f"frac_envs_succeeded={(base.successes>0).float().mean().item():.3f}"
    )
    best = int(lift_frames.argmax().item())
    emit(f"BEST_LIFT_ENV={best} lifted_frac={(lift_frames[best] / STEPS).item():.2f}")
    bs = int(peak_succ.argmax().item())
    emit(f"BEST_SUCCESS_ENV={bs} peak_goals_reached={int(peak_succ[bs].item())}/{base.trajectory_goals.shape[0] if base.trajectory_goals is not None else '-'}")
    rf.close()

    # flush the recorded video explicitly (RlGamesVecEnvWrapper.close may not reach the inner
    # RecordVideo; and on an early stop we broke before video_length was hit).
    if video_env is not None:
        try:
            video_env.render()  # capture the final frame
        except Exception:
            pass
        video_env.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
