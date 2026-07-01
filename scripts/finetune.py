"""Finetune the ORIGINAL SimToolReal pretrained policy in an Isaac Lab env on generated goals.

MODULAR: works for ANY registered task and ANY goal generator.
  --task            which env to finetune in        (e.g. Isaac-SimToolReal-Screwdriver043-Direct-v0)
  --goal_generator  open-loop goal-generator module  (overrides the env cfg's `goal_generator_module`;
                    any module exporting TOOL/BLADE/TIP/T/compute_goals_batch). OR
  --responsive      use the closed-loop responsive generator instead of an open-loop module.

It loads the pretrained weights into the matching network (built from the ORIGINAL config.yaml so the
state_dict aligns, exactly like deploy_pretrained.py) and CONTINUES training (finetune) on the env's
per-env goal trajectory. The env is put in training mode (NOT demo): reset noise + domain randomization
+ tolerance curriculum on, randomized layouts, kinematic screw (fast). `use_tighten_goals=True` makes
the env generate + advance the per-env goals during training so the reward trains "follow the trajectory."

Run (3072 envs = 6 SAPG blocks x 512; drop if OOM):
  cd IsaacLab && (venv) OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
    ~/simtoolreal_isaaclab/scripts/finetune.py --headless \
    --task Isaac-SimToolReal-Screwdriver043-Direct-v0 --num_envs 3072 --max_iterations 500
"""

import argparse
import math

from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"
DEFAULT_CONFIG = f"{ORIG_REPO}/pretrained_policy/config.yaml"
DEFAULT_CKPT = f"{ORIG_REPO}/pretrained_policy/model.pth"
LOG_DIR = "/home/cning/simtoolreal_isaaclab/logs/finetune"

parser = argparse.ArgumentParser()
# --- what to finetune, where ---
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-Screwdriver043-Direct-v0",
                    help="any registered Isaac Lab task id")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="pretrained model.pth to finetune from")
parser.add_argument("--orig_config", type=str, default=DEFAULT_CONFIG,
                    help="original pretrained config.yaml (network arch; built so the checkpoint state_dict loads)")
# --- goal generation (modular) ---
parser.add_argument("--goal_generator", type=str, default=None,
                    help="override the env's open-loop goal-generator MODULE (exports TOOL/BLADE/TIP/T/compute_goals_batch)")
parser.add_argument("--responsive", action="store_true",
                    help="use the closed-loop responsive generator instead of the open-loop module")
parser.add_argument("--reward_module", type=str, default="simtoolreal_lab.tasks.screwdriver043.reward_tip",
                    help="pluggable reward-augmentation MODULE (augment_reward(env)->reward_add,gate); e.g. the cross-slot TIP gate+bonus. '' to disable.")
parser.add_argument("--goal_noise", type=str, default="simtoolreal_lab.tasks.screwdriver043.goal_noise",
                    help="pluggable goal-pose NOISE schedule MODULE (sigma_schedule(T,phase_counts)); large early -> ~0 at insert/rotate. '' to disable.")
# --- training knobs ---
parser.add_argument("--num_envs", type=int, default=3072, help="must be divisible by 6 (SAPG blocks)")
parser.add_argument("--max_iterations", type=int, default=3000, help="ADDITIONAL finetune epochs beyond the checkpoint (~tip curriculum 10->2mm needs ~3000 at the default interval)")
parser.add_argument("--learning_rate", type=float, default=None, help="override base LR (default: keep the original 1e-4)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run_name", type=str, default=None, help="rl_games run name (auto: '00_ft_<task>'); MUST start with a number for SAPG")
parser.add_argument("--max_successes", type=int, default=None, help="override max_consecutive_successes (episode = this many goals; default: env cfg)")
parser.add_argument("--curriculum_interval", type=int, default=None,
                    help="override tolerance_curriculum_interval (control-steps between tip-tolerance anneal checks; cfg default 3000 ~= 187 iters/step -> ~3000 iters for 10->2mm; lower it for a faster anneal)")
# --- env knobs ---
parser.add_argument("--domain_randomization", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--randomize_layout", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--physical_screw", action="store_true", help="train with the physical revolute screw (slower; default kinematic)")
parser.add_argument("--spawn_screw", action=argparse.BooleanOptionalAction, default=False,
                    help="spawn the passive screw + thread_test during training (default OFF: reward is pure tool pose-reaching, goals only need screw_head_world). --spawn_screw to train under the screw's contact.")
parser.add_argument("--no_pretrained_compat", action="store_true", help="DON'T force the original obs/action convention (advanced)")
parser.add_argument("--agent_cfg", type=str, default=None,
                    help="optional rl_games agent yaml under agents/ to use INSTEAD of the original config.yaml (must match the checkpoint arch)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
import importlib  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import simtoolreal_lab.tasks  # noqa: E402, F401  (registers the gym tasks)
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from rl_games.common import a2c_common as _a2c  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402


def _actor_only_restore(self, weights, set_epoch=True):
    """Transfer-learning restore: load the pretrained ACTOR (policy + obs normalization) only.

    Deliberately SKIP the asymmetric critic (the port's privileged critic obs differs from the
    pretrained's: 155 vs 162 dims), the optimizer, epoch, and per-env env_state (num_envs differs).
    The critic + optimizer re-learn from scratch -- standard for finetuning across a task/scale change.
    """
    self.set_weights(weights)   # actor model + running_mean_std (dims match -> loads cleanly)
    print("[finetune] restored pretrained ACTOR weights only "
          "(critic + optimizer re-learn; epoch reset to 0).", flush=True)


# patched for finetuning across the critic-obs/num_envs mismatch (see above)
_a2c.A2CBase.set_full_state_weights = _actor_only_restore

AGENTS_DIR = "/home/cning/simtoolreal_isaaclab/simtoolreal_lab/tasks/simtoolreal/agents"  # only if --agent_cfg given


def resolve_env_cfg(task: str):
    entry = gym.spec(task).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def build_agent_cfg(num_envs: int, n_blocks: int = 6) -> dict:
    """Agent cfg whose network MATCHES the checkpoint, with TRAINING params for finetuning.

    Default: build from the ORIGINAL config.yaml's train.params (network untouched -> state_dict loads),
    add load_checkpoint, and set the SAPG/PPO training config to the requested env count. With
    --agent_cfg, use that yaml instead (it must match the checkpoint architecture)."""
    if args_cli.agent_cfg:
        with open(os.path.join(AGENTS_DIR, args_cli.agent_cfg)) as f:
            params = yaml.safe_load(f)["params"]
    else:
        with open(args_cli.orig_config) as f:
            params = yaml.safe_load(f)["train"]["params"]

    params["seed"] = args_cli.seed
    params["load_checkpoint"] = True            # restore weights (+ optimizer/mean-std/epoch) -> finetune
    params["load_path"] = args_cli.checkpoint

    c = params["config"]
    # SAPG parses int(name.split('_')[0]); derive a readable name from the task (".../-Name-Direct-v0")
    _parts = args_cli.task.split("-")
    c["name"] = args_cli.run_name or ("00_ft_" + (_parts[-3] if len(_parts) >= 3 else args_cli.task).lower())
    c["device_name"] = "cuda:0"; c["device"] = "cuda:0"; c["multi_gpu"] = False
    c["num_actors"] = num_envs
    c["clip_actions"] = False                   # env clamps actions itself; rl_games rescale would NaN on inf bounds
    c["train_dir"] = LOG_DIR
    # actor-only restore resets the epoch to 0, so max_epochs == the finetune budget directly
    c["max_epochs"] = args_cli.max_iterations
    # SAPG structure scaled to the env count (matches the original ratios at any scale)
    horizon = c.get("horizon_length", 16)
    c["minibatch_size"] = (horizon * num_envs) // 4
    if "expl_coef_block_size" in c:
        assert num_envs % n_blocks == 0, f"num_envs {num_envs} must be divisible by {n_blocks} (SAPG blocks)"
        c["expl_coef_block_size"] = num_envs // n_blocks
    if "central_value_config" in c:
        c["central_value_config"]["minibatch_size"] = c["minibatch_size"]
        if args_cli.learning_rate is not None:
            c["central_value_config"]["learning_rate"] = args_cli.learning_rate
    if args_cli.learning_rate is not None:
        c["learning_rate"] = args_cli.learning_rate
    # resolve a few OmegaConf ${...} summary interpolations the Runner touches
    c["defer_summaries_sec"] = 5
    c["summaries_interval_sec_min"] = 5
    c["summaries_interval_sec_max"] = 300
    return {"params": params}


def main():
    torch.manual_seed(args_cli.seed)

    # ---- env cfg: training mode + the chosen goal generator ----
    env_cfg = resolve_env_cfg(args_cli.task)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.domain_randomization = args_cli.domain_randomization
    if not args_cli.no_pretrained_compat:
        env_cfg.pretrained_compat = True        # original obs/action convention -> matches the checkpoint
        # NB: do NOT set eval_append_expl_coef here -- that's for EVAL (env appends the fixed exploit
        # coef). During SAPG TRAINING the algorithm appends the exploration coef itself (obs 140 -> 141).
    # train ON the goal trajectory (these flags only exist on the screwdriver-family cfgs)
    if hasattr(env_cfg, "randomize_layout"):
        env_cfg.randomize_layout = args_cli.randomize_layout
    if args_cli.responsive and hasattr(env_cfg, "responsive_goals"):
        env_cfg.responsive_goals = True          # closed-loop generator (no per-env open-loop goals)
    elif hasattr(env_cfg, "use_tighten_goals"):
        env_cfg.use_tighten_goals = True         # generate + advance the open-loop goals during training
    if args_cli.goal_generator and hasattr(env_cfg, "goal_generator_module"):
        env_cfg.goal_generator_module = args_cli.goal_generator
    # pluggable reward augmentation (tip gate+bonus) + goal-pose noise schedule (training diversity)
    if args_cli.reward_module and hasattr(env_cfg, "reward_module"):
        env_cfg.reward_module = args_cli.reward_module
    if args_cli.goal_noise and hasattr(env_cfg, "goal_noise_module"):
        env_cfg.goal_noise_module = args_cli.goal_noise
    if hasattr(env_cfg, "physical_screw"):
        env_cfg.physical_screw = args_cli.physical_screw
    if hasattr(env_cfg, "spawn_passive_screw"):
        env_cfg.spawn_passive_screw = args_cli.spawn_screw   # default False -> train screw-free
    # SDF colliders (screwdriver, + screw if physical) generate many contacts; size the GPU contact
    # stack to the env count (~the 043 SDF screwdriver needs it even at small N). Reduce --num_envs if OOM.
    # gpu_collision_stack_size is a USD uint32 (max ~4.29e9): scale with env count but CAP with
    # headroom, or large-N runs overflow ("Type mismatch ... expected 'unsigned int', got 'long'").
    # Screw-free is SDF-screwdriver vs convex hand/ground (cheap); the original IsaacGym ran 24576
    # envs on ~8M contact pairs, so 3 GiB is generous here. Reduce --num_envs if PhysX drops contacts.
    # uint32 max is ~4.29e9; the 3072-env startup-settling spike peaks ~3.7e9, so cap near (not at) the
    # ceiling to cover it without overflow.
    per_env = 12_000_000 if args_cli.physical_screw else 7_000_000
    est = max(2 ** 28, args_cli.num_envs * per_env)
    env_cfg.sim.physx.gpu_collision_stack_size = int(min(est, 4_200_000_000))
    if args_cli.max_successes is not None:
        env_cfg.max_consecutive_successes = args_cli.max_successes
    env_cfg.demo_mode = False
    env_cfg.use_fixed_goal_trajectory = True     # base loads a trajectory; per-env goals override it
    # --- train==eval consistency + the tolerance design (see cross-slot finetune notes) ---
    # (1) score success on FIXED-SIZE keypoints, like the original (fixedSizeKeypointReward used in
    #     TRAINING too) and like eval -> the trained metric == the eval metric.
    if hasattr(env_cfg, "fixed_size_success"):
        env_cfg.fixed_size_success = True
    # (2) keypoint tolerance FIXED at 0.01 (the eval/target value). The pretrained policy already
    #     reaches it, so skip the from-scratch 0.075->0.01 curriculum (which can teach sloppy tracking).
    env_cfg.success_tolerance = 0.01
    env_cfg.use_tolerance_curriculum = False
    # (3) the TIP tolerance gets its OWN curriculum (reward_tip: 10mm -> 2mm), success-gated and
    #     independent of the (now fixed) keypoint tolerance. (successSteps stays the cfg default 10,
    #     matching the original TRAINING.) pretrained_object_scale (2.5,0.75,0.75) is set in the cfg.
    if hasattr(env_cfg, "tip_tol_curriculum"):
        env_cfg.tip_tol_curriculum = True
    if args_cli.curriculum_interval is not None:
        env_cfg.tolerance_curriculum_interval = args_cli.curriculum_interval  # faster/slower tip anneal
    # (4b) the episode must be long enough to walk the WHOLE trajectory -- otherwise the late
    #      rotate/tighten goals NEVER train (the cfg default 600 steps / max_consecutive_successes=50
    #      ends at ~goal 50). Cap at the trajectory length (one full tighten per episode -> resets to a
    #      new layout) and give enough steps to complete it (success_steps per goal + travel margin).
    #      This mirrors what eval/--demo does (max_consecutive_successes = #goals).
    if getattr(env_cfg, "use_tighten_goals", False) and hasattr(env_cfg, "goal_generator_module"):
        _T = int(importlib.import_module(env_cfg.goal_generator_module).T)
        if args_cli.max_successes is None:
            env_cfg.max_consecutive_successes = _T
        env_cfg.episode_length_s = (_T * env_cfg.success_steps * 2 + 120) / 60.0
        print(f"[finetune] trajectory T={_T} -> max_consecutive_successes={env_cfg.max_consecutive_successes}, "
              f"episode_length_s={env_cfg.episode_length_s:.1f}s "
              f"(~{int(env_cfg.episode_length_s * 60)} control steps)", flush=True)

    gen = ("responsive" if args_cli.responsive else
           (args_cli.goal_generator or getattr(env_cfg, "goal_generator_module", "<env default>")))
    print(f"[finetune] task={args_cli.task} | goals={gen} | num_envs={args_cli.num_envs} "
          f"| physical_screw={args_cli.physical_screw} | ckpt={args_cli.checkpoint}", flush=True)
    print(f"[finetune] reward_module={getattr(env_cfg,'reward_module',None)} | "
          f"goal_noise_module={getattr(env_cfg,'goal_noise_module',None)}", flush=True)

    # ---- agent cfg (network matches the checkpoint) ----
    agent_cfg = build_agent_cfg(args_cli.num_envs)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"].get("env", {}).get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"].get("env", {}).get("clip_actions", math.inf)

    # ---- build + wrap env, register with rl_games ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper",
                    lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    os.makedirs(LOG_DIR, exist_ok=True)
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    # the fork's _restore only fires when 'checkpoint' is in the run args (params.load_path is for the
    # player only). Pass it here so the pretrained weights are actually loaded for finetuning.
    runner.run({"train": True, "play": False, "checkpoint": args_cli.checkpoint})

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
