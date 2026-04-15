# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--eval_video_only",
    action="store_true",
    default=False,
    help="Record video during eval only; suppress interval-based and curriculum-triggered training videos.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--zero-actor",
    action="store_true",
    default=False,
    help="Make the last layer of the actor network a zero layer.",
)
parser.add_argument(
    "--eval_episodes_per_save",
    type=int,
    default=100,
    help="Number of completed episodes to collect for eval after each checkpoint save (0 to disable).",
)
parser.add_argument(
    "--set-std",
    type=float,
    default=None,
    help="Std of the policy network regardless of the checkpoint.",
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--ray-proc-id",
    "-rid",
    type=int,
    default=None,
    help="Automatically configured by Ray integration, otherwise None.",
)
parser.add_argument(
    "--scene_config",
    type=str,
    default=None,
    help="Path to the scene configuration file.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Motion file to load.",
)
parser.add_argument(
    "--wandb_id",
    type=str,
    default=None,
    help="Wandb run ID to resume from.",
)
parser.add_argument(
    "--use_primitive_urdfs",
    action="store_true",
    default=False,
    help="Use primitive URDFs for the robot.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [
            r".\isaaclab.bat",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    else:
        cmd = [
            "./isaaclab.sh",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import time
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
from download_from_wandb import download_run

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import robotic_grounding.tasks  # noqa: F401
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from robotic_grounding.tasks.v2p_whole_body.utils import WandbVideoUploader

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class _EvalCallback:
    """Runs inference episodes after each checkpoint save and logs completion stats to wandb.

    Triggered by monkey-patching runner.save().  After a warm-up phase that
    waits until every env has reset at least once (so we only measure clean
    episodes), it collects `eval_episodes` completed episodes in inference mode
    and logs:

        eval/completion_ratio_mean   – mean fraction of trajectory completed
        eval/completion_ratio_std    – std of the above
        eval/full_completion_pct     – % of episodes that completed ≥99% of traj

    If the training env is wrapped with RecordVideo (i.e. --video is set),
    it also triggers a video recording during the evaluation rollout.
    """

    def __init__(self, runner, env, eval_episodes: int, log_video: bool) -> None:
        import re as _re
        self._re = _re
        self.runner = runner
        self.env = env  # RslRlVecEnvWrapper
        self.eval_episodes = eval_episodes
        self.log_video = log_video

        isaac_env = env.unwrapped
        self._cmd = isaac_env.command_manager.get_term(
            "dual_hands_object_tracking_command"
        )
        self._warmup_steps = getattr(
            self._cmd.cfg, "virtual_object_control_decay_steps", 20
        )
        self._traj_len = self._cmd.retargeted_horizon
        self._isaac_env = isaac_env

        # Find the RecordVideo wrapper so eval videos can be redirected to videos/eval
        self._record_video_env = None
        self._eval_video_folder = None
        self._train_video_folder = None
        self._logged_eval_videos: set = set()
        if log_video:
            w = getattr(env, "env", None)
            while w is not None:
                if hasattr(w, "video_folder") and hasattr(w, "start_recording"):
                    self._record_video_env = w
                    self._train_video_folder = w.video_folder
                    self._eval_video_folder = os.path.join(
                        os.path.dirname(w.video_folder), "eval"
                    )
                    os.makedirs(self._eval_video_folder, exist_ok=True)
                    break
                w = getattr(w, "env", None)

    def __call__(self, path: str, *args, **kwargs) -> None:
        import wandb

        if wandb.run is None:
            return

        match = self._re.search(r"model_(\d+)\.pt", os.path.basename(path))
        iteration = int(match.group(1)) if match else None

        device = self.env.unwrapped.device
        num_envs = self.env.unwrapped.num_envs

        policy = self.runner.get_inference_policy(device=device)
        try:
            policy_nn = self.runner.alg.policy
        except AttributeError:
            policy_nn = self.runner.alg.actor_critic

        obs = self.env.get_observations()

        # --- Warm-up: step until every env has completed at least one episode ---
        # This ensures the episodes we collect started from a genuine reset, not
        # mid-episode when the callback fired.
        env_reset_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
        while not env_reset_once.all():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = self.env.step(actions)
                policy_nn.reset(dones)
            env_reset_once |= dones.bool()

        # --- Collect eval_episodes clean episodes + drain recording ---
        # Allocate state tensors before the try so they're accessible after the
        # finally (for the stats section).  These assignments cannot raise.
        traj_len = self._cmd.retargeted_horizon  # re-read in case init was stale
        my_ep_len = torch.zeros(num_envs, dtype=torch.float32, device=device)
        # tracking_length captured at the start of each env's fresh episode
        my_ep_max_len = torch.zeros(num_envs, dtype=torch.float32, device=device)
        env_first_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        completed: list[tuple[float, float]] = []

        try:
            # --- Trigger video recording for the upcoming eval rollout ---
            # Redirect RecordVideo to videos/eval so the clip is logged separately
            # from training videos (uploaded as eval/video rather than train/video).
            # Done inside try so the finally always restores video_folder, even if
            # the eval loop or drain throws (e.g. CUDA OOM, moviepy write error).
            if self.log_video:
                if self._record_video_env is not None:
                    self._record_video_env.video_folder = self._eval_video_folder
                self._isaac_env.eval_video_trigger_pending = True

            # After warmup each env's episode is at some arbitrary midpoint.  We
            # discard each env's first post-warmup episode (the "tail"), then count
            # only full episodes that started fresh.  tracking_length is captured at
            # the START of each fresh episode so the per-episode denominator is
            # correct regardless of random start offsets.
            while len(completed) < self.eval_episodes:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = self.env.step(actions)
                    policy_nn.reset(dones)

                my_ep_len += 1
                done_mask = dones.bool()
                if done_mask.any():
                    done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
                    for i in done_ids:
                        if env_first_done[i]:
                            # Fresh episode — record (ep_len, tracking_length at start).
                            completed.append((my_ep_len[i].item(), my_ep_max_len[i].item()))
                        else:
                            # Tail of a mid-episode that pre-dated the warmup end;
                            # mark done so the NEXT episode is recorded.
                            env_first_done[i] = True
                        my_ep_len[i] = 0.0
                        # Capture tracking_length for the new episode that just started.
                        my_ep_max_len[i] = self._cmd.tracking_lengths[i].float()

            # Drain: keep stepping until the in-progress eval recording finishes.
            # Cap at video_length * 3 steps to avoid an infinite loop if the
            # recording state somehow gets stuck.
            if self.log_video and self._record_video_env is not None:
                _drain_limit = (
                    self._record_video_env.video_length
                    if self._record_video_env.video_length != float("inf")
                    else 600
                ) * 3
                _drain_steps = 0
                while self._record_video_env.recording:
                    if _drain_steps >= _drain_limit:
                        print(
                            f"[eval] WARNING: drain loop hit {_drain_limit}-step limit "
                            "while waiting for eval recording to finish; aborting drain."
                        )
                        break
                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, _, dones, _ = self.env.step(actions)
                        policy_nn.reset(dones)
                    _drain_steps += 1

        finally:
            # Always restore the train video folder and clear any unconsumed pending
            # trigger, regardless of whether an exception occurred above.
            if self.log_video:
                self._isaac_env.video_trigger_pending = False
                self._isaac_env.eval_video_trigger_pending = False
                if self._record_video_env is not None:
                    self._record_video_env.video_folder = self._train_video_folder

        # Log eval video directly from the main thread so it is committed reliably.
        # Background-thread commit=False is not guaranteed to flush before the run ends.
        if self.log_video and self._eval_video_folder:
            import glob as _glob
            new_videos = sorted(
                [f for f in _glob.glob(os.path.join(self._eval_video_folder, "*.mp4"))
                 if f not in self._logged_eval_videos],
                key=os.path.getmtime,
            )
            if new_videos:
                src_path = new_videos[-1]
                model_step = iteration if iteration is not None else 0
                dst_name = f"video_eval_{model_step}.mp4"
                dst_path = os.path.join(self._eval_video_folder, dst_name)
                if src_path != dst_path:
                    os.rename(src_path, dst_path)
                log_video_data = {"eval/video": wandb.Video(dst_path, format="mp4")}
                if iteration is not None:
                    wandb.log(log_video_data, step=iteration, commit=False)
                else:
                    wandb.log(log_video_data, commit=False)
                self._logged_eval_videos.update(new_videos)  # mark all seen as tracked, not just the logged one
                self._logged_eval_videos.add(dst_path)       # also track renamed destination

        # --- Compute stats ---
        # ratio = fraction of the episode's own tracking_length that was
        # completed after the per-episode VOC warmup (warmup_steps).
        data = completed[: self.eval_episodes]
        print(
            f"[eval] iter={iteration}  traj_len={traj_len}  "
            f"warmup_steps={self._warmup_steps}  "
            f"sample ep_lens={[round(e[0]) for e in data[:5]]}"
        )
        ratios = [
            min(max(e[0] - self._warmup_steps, 0), e[1])
            / max(e[1], 1)
            for e in data
        ]
        n_full = sum(1 for r in ratios if r >= 0.99)
        mean_r = sum(ratios) / len(ratios)
        std_r = (
            sum((x - mean_r) ** 2 for x in ratios) / max(len(ratios) - 1, 1)
        ) ** 0.5
        full_pct = 100.0 * n_full / len(data)

        # --- Log to wandb ---
        log_data = {
            "eval/completion_ratio_mean": mean_r,
            "eval/completion_ratio_std": std_r,
            "eval/full_completion_pct": full_pct,
        }
        if iteration is not None:
            wandb.log(log_data, step=iteration, commit=False)
        else:
            wandb.log(log_data, commit=False)

        print(
            f"[eval] iter={iteration}  "
            f"ratio={mean_r:.3f}±{std_r:.3f}  full={n_full}/{len(data)} ({full_pct:.0f}%)"
        )

        # Refresh runner's internal obs so the next collect_rollouts starts consistently
        if hasattr(self.runner, "obs"):
            self.runner.obs = obs


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )

    scene_config = None
    # Apply scene config: motion_file (from Hydra override) takes priority,
    # then --scene_config YAML, then the env_cfg default.
    if args_cli.motion_file is not None:
        env_cfg.motion_file = args_cli.motion_file
    if hasattr(env_cfg, "motion_file") and env_cfg.motion_file is not None:
        scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
        apply_scene_config(
            env_cfg, scene_config, use_primitive_urdfs=args_cli.use_primitive_urdfs
        )
    elif args_cli.scene_config is not None:
        env_cfg.scene_config_path = args_cli.scene_config
        scene_config = SceneConfig.from_yaml(args_cli.scene_config)
        apply_scene_config(
            env_cfg, scene_config, use_primitive_urdfs=args_cli.use_primitive_urdfs
        )

    # set max iterations
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # check for invalid combination of CPU device with distributed training
    if (
        args_cli.distributed
        and args_cli.device is not None
        and "cpu" in args_cli.device
    ):
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if os.path.isabs(agent_cfg.load_checkpoint) or os.path.exists(
            agent_cfg.load_checkpoint
        ):
            # allow path to be directly specified
            resume_path = os.path.abspath(agent_cfg.load_checkpoint)
        elif args_cli.wandb_id is not None:
            resume_path = download_run(args_cli.wandb_id)
            agent_cfg.load_checkpoint = resume_path
        else:
            # get checkpoint from log directory
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )

    # wrap for video recording
    if args_cli.video:
        _base_env = env  # capture reference before RecordVideo wrapping
        # Mutable reference so _video_step_trigger can see the RecordVideo wrapper
        # after it is created below (closures capture variables, not values).
        _record_video_ref = [None]

        def _video_step_trigger(step: int) -> bool:
            # Use .unwrapped to reach the actual Isaac env: gym.make() wraps it in
            # OrderEnforcing, and setting an attribute on that wrapper would create a
            # shadow that permanently masks subsequent writes to the inner Isaac env.
            _isaac_env = _base_env.unwrapped
            # Eval-triggered recording — always honored, even with --eval_video_only.
            eval_pending = getattr(_isaac_env, "eval_video_trigger_pending", False)
            if eval_pending:
                _isaac_env.eval_video_trigger_pending = False
                return True
            # Training-triggered recording (curriculum pre-decay + interval).
            # Suppressed entirely by --eval_video_only.
            if args_cli.eval_video_only:
                return False
            pending = getattr(_isaac_env, "video_trigger_pending", False)
            if pending:
                _isaac_env.video_trigger_pending = False
                return True
            # Only fire the interval trigger when not already recording, to avoid
            # prematurely cutting an in-progress eval recording.
            _rv = _record_video_ref[0]
            if _rv is not None and _rv.recording:
                return False
            return step % args_cli.video_interval == 0

        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": _video_step_trigger,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        _record_video_ref[0] = env  # fill the reference now that the wrapper exists

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # Reset the std of the policy network
    if args_cli.set_std is not None:
        with torch.no_grad():
            runner.alg.policy.std.fill_(args_cli.set_std)

    # set the actor network to zero if requested
    if args_cli.zero_actor:
        torch.nn.init.zeros_(runner.alg.policy.actor[-1].weight)
        torch.nn.init.zeros_(runner.alg.policy.actor[-1].bias)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # setup eval callback: after each checkpoint save, run inference episodes and log stats
    if args_cli.eval_episodes_per_save > 0:
        _eval_cb = _EvalCallback(
            runner=runner,
            env=env,
            eval_episodes=args_cli.eval_episodes_per_save,
            log_video=args_cli.video,
        )
        _orig_save = runner.save

        def _save_with_eval(path: str, *a, **kw):
            _orig_save(path, *a, **kw)
            try:
                _eval_cb(path)
            except Exception as exc:
                print(f"[eval] WARNING: eval callback raised an exception: {exc}")

        runner.save = _save_with_eval

        # Fire one eval immediately at startup (after wandb is initialised).
        # _prepare_logging_writer is idempotent; calling it here lets the eval
        # log before the first save_interval checkpoint.
        _orig_learn = runner.learn

        def _learn_with_startup_eval(num_learning_iterations, **kw):
            runner._prepare_logging_writer()
            try:
                _eval_cb("model_startup.pt")
            except Exception as exc:
                print(f"[eval] WARNING: startup eval failed: {exc}")
            return _orig_learn(num_learning_iterations, **kw)

        runner.learn = _learn_with_startup_eval

    # setup video uploader for wandb (train folder only; eval videos logged directly by _EvalCallback)
    video_uploader = None
    if args_cli.video and agent_cfg.logger == "wandb":
        video_folder = os.path.join(log_dir, "videos", "train")
        os.makedirs(video_folder, exist_ok=True)
        video_uploader = WandbVideoUploader(
            video_folder,
            check_interval=60.0,
            num_steps_per_env=agent_cfg.num_steps_per_env,
            wandb_key="train/video",
        )
        video_uploader.start()

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # stop video uploader
    if video_uploader is not None:
        video_uploader.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
