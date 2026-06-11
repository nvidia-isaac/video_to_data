"""Train the SimToolReal vertical slice with rl_games (Isaac Lab).

Self-contained launcher: registers the task, builds + wraps the env, runs rl_games.
Swap the stock rl_games for the SAPG fork by pip-installing the fork into this venv;
the Runner/A2CAgent API matches, and SAPG flags live in the agent yaml.

Run: ./isaaclab.sh -p scripts/train.py --headless --num_envs 4096 --max_iterations 200
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-ClawHammer-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent_cfg", type=str, default="rl_games_ppo_cfg.yaml", help="agent yaml filename under agents/")
parser.add_argument(
    "--domain_randomization",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="enable domain randomization (default: on; use --no-domain_randomization to disable)",
)
parser.add_argument("--run_name", type=str, default=None, help="override the rl_games run name (must start with '00_' for SAPG)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import simtoolreal_lab.tasks  # noqa: E402, F401  (registers the gym task)
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import importlib  # noqa: E402

AGENTS_DIR = (
    "/home/cning/simtoolreal_isaaclab/simtoolreal_lab/tasks/simtoolreal/agents"
)
LOG_DIR = "/home/cning/simtoolreal_isaaclab/logs/simtoolreal"


def resolve_env_cfg(task: str):
    """Load the env-cfg class registered for `task` (so each task gets its own cfg)."""
    entry = gym.spec(task).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def main():
    env_cfg = resolve_env_cfg(args_cli.task)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.domain_randomization = args_cli.domain_randomization

    with open(os.path.join(AGENTS_DIR, args_cli.agent_cfg)) as f:
        agent_cfg = yaml.safe_load(f)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # build + wrap env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register with rl_games
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # finalize agent cfg
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["seed"] = args_cli.seed
    if args_cli.run_name is not None:
        agent_cfg["params"]["config"]["name"] = args_cli.run_name

    # auto-scale SAPG structure to the env count (matches original ratios at any scale):
    # 6 blocks, minibatch = batch/4. Also keeps OOM-fallback env counts consistent.
    cfg_c = agent_cfg["params"]["config"]
    n_envs = env.unwrapped.num_envs
    cfg_c["minibatch_size"] = (cfg_c["horizon_length"] * n_envs) // 4
    if "expl_coef_block_size" in cfg_c:
        assert n_envs % 6 == 0, f"num_envs {n_envs} must be divisible by 6 (SAPG blocks)"
        cfg_c["expl_coef_block_size"] = n_envs // 6
    if "central_value_config" in cfg_c:
        cfg_c["central_value_config"]["minibatch_size"] = cfg_c["minibatch_size"]
    agent_cfg["params"]["config"]["train_dir"] = LOG_DIR
    if args_cli.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations

    os.makedirs(LOG_DIR, exist_ok=True)
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False})

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
