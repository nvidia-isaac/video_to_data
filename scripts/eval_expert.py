"""Evaluate the SAPG EXPERT on the current hammer eval harness (for an apples-to-apples bar vs the BC
specialists). Rolls out the pretrained rl_games player in the standard HammerEnvCfg eval env (no
teleport, no noise) and reports the nail-driven success rate over --episodes episodes.

Run:
  source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/eval_expert.py --headless --num_envs 25 --episodes 100
"""

import argparse
import math

from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"
REPO = "/home/cning/simtoolreal_isaaclab"
DEFAULT_CONFIG = f"{ORIG_REPO}/pretrained_policy/config.yaml"
DEFAULT_CKPT = f"{ORIG_REPO}/pretrained_policy/model.pth"
LOG_DIR = f"{REPO}/logs/simtoolreal"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--max_ep_steps", type=int, default=800)
parser.add_argument("--success_joint", type=float, default=-0.006)
parser.add_argument("--orig_config", type=str, default=DEFAULT_CONFIG)
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app = AppLauncher(args_cli).app

import sys  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, REPO)
import gymnasium as gym  # noqa: E402
import simtoolreal_lab.tasks  # noqa: E402, F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402

GYM_ID = "Isaac-SimToolReal-Hammer-Direct-v0"


def build_agent_cfg(num_envs):
    with open(args_cli.orig_config) as f:
        params = yaml.safe_load(f)["train"]["params"]
    params["seed"] = args_cli.seed
    params["load_checkpoint"] = True
    params["load_path"] = args_cli.checkpoint
    c = params["config"]
    c["name"] = "00_eval_expert"; c["device_name"] = c["device"] = "cuda:0"; c["multi_gpu"] = False
    c["num_actors"] = num_envs; c["clip_actions"] = False; c["max_epochs"] = 1; c["max_frames"] = 100
    c["expl_coef_block_size"] = max(1, num_envs // 6); c["minibatch_size"] = num_envs
    if "central_value_config" in c:
        c["central_value_config"]["minibatch_size"] = num_envs
    c["train_dir"] = LOG_DIR; c.setdefault("player", {})
    c["player"]["games_num"] = 128; c["player"]["deterministic"] = True
    return {"params": params}


def make_cfg():
    cfg = HammerEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.pretrained_compat = True
    cfg.eval_append_expl_coef = True
    cfg.domain_randomization = False
    cfg.use_tolerance_curriculum = False
    cfg.success_steps = 1; cfg.success_tolerance = 0.01; cfg.max_consecutive_successes = 0
    cfg.episode_length_s = args_cli.max_ep_steps / 60.0
    cfg.reset_dof_pos_noise_arm = 0.0; cfg.reset_dof_pos_noise_fingers = 0.0
    cfg.reset_position_noise_x = cfg.reset_position_noise_y = cfg.reset_position_noise_z = 0.0
    cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_2"] = 1.571 - math.radians(10)
    cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_4"] = 1.376 + math.radians(10)
    cfg.per_env_camera = False
    cfg.use_fixed_goal_trajectory = False; cfg.use_tighten_goals = True
    cfg.randomize_layout = True; cfg.physical_screw = True
    cfg.screw_contact_clearance = -0.04
    cfg.terminate_on_nail_driven = args_cli.success_joint
    cfg.terminate_on_screw_rotated = None
    cfg.sim.physx.gpu_collision_stack_size = 2 ** 30
    return cfg


def main():
    torch.manual_seed(args_cli.seed)
    N = args_cli.num_envs
    agent_cfg = build_agent_cfg(N)
    env = gym.make(GYM_ID, cfg=make_cfg(), render_mode=None)
    env = RlGamesVecEnvWrapper(env, "cuda:0", math.inf, math.inf)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    runner = Runner(IsaacAlgoObserver()); runner.load(agent_cfg); runner.reset()
    player = runner.create_player(); player.restore(args_cli.checkpoint); player.has_batch_dimension = True
    base = env.unwrapped

    obs = env.reset(); o = obs["obs"]
    if player.is_rnn:
        player.init_rnn()
    successes = attempts = steps = 0
    print(f"[expert] rollout: N={N} envs, target {args_cli.episodes} episodes", flush=True)
    while attempts < args_cli.episodes and steps < args_cli.max_steps:
        a = player.get_action(o, is_deterministic=True)
        obs, _, done, _ = env.step(a); o = obs["obs"]
        done_b = done.bool()
        nd = base.nail_driven
        idx = torch.nonzero(done_b).flatten().tolist()
        if idx:
            attempts += len(idx)
            successes += int(nd[done_b].sum().item())
        if player.is_rnn and bool(done_b.any()):
            for s in player.states:
                s[:, done_b, :] = 0.0
        steps += 1
        if steps % 100 == 0:
            print(f"[expert] step={steps} episodes={attempts} success={successes} "
                  f"rate={successes/max(1,attempts):.0%}", flush=True)
    print(f"\n[expert] DONE: {successes}/{attempts} episodes succeeded "
          f"(success_rate {successes/max(1,attempts):.1%}) over {steps} control steps", flush=True)
    env.close(); app.close()


if __name__ == "__main__":
    main()
