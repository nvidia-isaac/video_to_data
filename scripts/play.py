"""Evaluate a trained SimToolReal policy on the fixed-goal swing_down trajectory.

Loads an rl_games checkpoint, puts the env in fixed-goal-trajectory mode (steps through
the 37 swing_down goals as the object reaches each), and runs the policy, reporting the
env-logged success metric.

Run: ./isaaclab.sh -p scripts/play.py --headless --checkpoint <path.pth> --num_envs 64
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-ClawHammer-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_cfg", type=str, default="rl_games_sapg_cfg.yaml")
parser.add_argument("--delta", action="store_true", help="eval on training delta-goal distribution instead of fixed trajectory")
parser.add_argument("--video", action="store_true", help="record an Omniverse RTX mp4 of the rollout")
parser.add_argument("--video_length", type=int, default=300, help="number of steps to record")
parser.add_argument("--cam_eye", type=str, default="-1.0,-0.6,0.95", help="render camera position (x,y,z), env-0-local")
parser.add_argument("--cam_lookat", type=str, default="0.0,0.35,0.5", help="render camera target (x,y,z), env-local")
parser.add_argument("--cam_env_index", type=int, default=0, help="render camera follows this env index")
parser.add_argument("--seed", type=int, default=0, help="seed for reproducible env resets across runs")
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

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import simtoolreal_lab.tasks  # noqa: E402, F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

from simtoolreal_lab.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg  # noqa: E402

AGENTS_DIR = "/home/cning/simtoolreal_isaaclab/simtoolreal_lab/tasks/simtoolreal/agents"
RESULT_FILE = "/home/cning/simtoolreal_isaaclab/eval_result.txt"


def main():
    import torch

    torch.manual_seed(args_cli.seed)  # reproducible env resets so the same env lifts across the 2 passes
    env_cfg = SimToolRealEnvCfg()
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.use_fixed_goal_trajectory = not args_cli.delta  # fixed swing_down goals (default) or training delta goals
    env_cfg.eval_append_expl_coef = True  # SAPG coef_cond: feed exploit coefficient at play
    env_cfg.domain_randomization = False  # eval is clean/deterministic (no DR perturbations/noise)
    env_cfg.use_tolerance_curriculum = False  # eval uses the fixed base success tolerance

    with open(os.path.join(AGENTS_DIR, args_cli.agent_cfg)) as f:
        agent_cfg = yaml.safe_load(f)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args_cli.video:
        # frame the render camera close on env 0's robot (eye/lookat are env-0-local)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = args_cli.cam_env_index
        env_cfg.viewer.resolution = (1280, 720)
        env_cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
        env_cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = "/home/cning/simtoolreal_isaaclab/videos"
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,  # record from the first step
            video_length=args_cli.video_length,
            name_prefix="simtoolreal_" + ("delta" if args_cli.delta else "swing_down"),
            disable_logger=True,
        )
        print(f"[video] recording {args_cli.video_length} steps -> {video_dir}")
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    # cap eval episodes (default games_num is huge -> would run forever)
    agent_cfg["params"]["config"].setdefault("player", {})
    agent_cfg["params"]["config"]["player"]["games_num"] = 128
    agent_cfg["params"]["config"]["player"]["deterministic"] = True

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    # build player, restore checkpoint, drive deterministically (manual loop -> clean metrics)
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.has_batch_dimension = True

    base = env.unwrapped
    rf = open(RESULT_FILE, "w")

    def emit(m):
        rf.write(m + "\n")
        rf.flush()
        os.fsync(rf.fileno())

    STEPS = 600
    lift_frames = torch.zeros(base.num_envs, device=base.device)  # per-env count of lifted frames
    obs = env.reset()
    o = obs["obs"]
    if player.is_rnn:
        player.init_rnn()
    for t in range(STEPS):
        a = player.get_action(o, is_deterministic=True)
        obs, rew, done, info = env.step(a)
        o = obs["obs"]
        # reset LSTM hidden state for envs that ended this step
        if player.is_rnn and bool(done.any()):
            d = done.bool()
            for s in player.states:
                s[:, d, :] = 0.0
        lift_frames += base.lifted_object.float()
        if t % 60 == 0 or t == STEPS - 1:
            emit(
                f"t={t:3d} lift_rate={base.lifted_object.float().mean().item():.2f} "
                f"obj_z={base.object_pos[:,2].mean().item():.3f} "
                f"kp_dist_to_goal={base.keypoints_max_dist.mean().item():.3f} "
                f"succ_mean={base.successes.mean().item():.2f} succ_max={int(base.successes.max().item())}"
            )
    emit(
        f"FINAL succ_mean={base.successes.mean().item():.3f} "
        f"succ_max={int(base.successes.max().item())} "
        f"lift_rate={base.lifted_object.float().mean().item():.3f} "
        f"frac_lifted_envs={(base.successes>0).float().mean().item():.3f}"
    )
    best = int(lift_frames.argmax().item())
    emit(f"BEST_LIFT_ENV={best} lifted_frac={(lift_frames[best] / STEPS).item():.2f}")
    rf.close()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
