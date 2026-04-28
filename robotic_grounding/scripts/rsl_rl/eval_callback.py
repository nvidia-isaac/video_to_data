"""Eval callback utility for RSL-RL training."""

import os


class EvalCallback:
    """Runs inference episodes after each checkpoint save and logs completion stats to wandb.

    Two eval passes are run per checkpoint:

      Pass A (from_start): every reset goes to tc=0; measures whether the policy
          can complete the full trajectory from scratch.
          Logs: eval/completion_ratio_mean, eval/completion_ratio_std,
                eval/full_completion_pct

      Pass B (random): resets use training behaviour (random tc); measures what
          fraction of the trajectory the policy has learned to follow regardless
          of starting position.
          Logs: eval/completion_ratio_mean_random, eval/completion_ratio_std_random

    Both passes run a warm-up phase that waits until every env has reset at least
    once, then collect `eval_episodes` completed episodes in inference mode.

    If the training env is wrapped with RecordVideo (i.e. --video is set),
    a video is captured during pass A only (genuine from-frame-0 rollout).
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

    def _collect_episodes(
        self,
        policy,
        policy_nn,
        obs,
        from_start: bool,
        log_video: bool,
    ):
        """Run one eval pass; returns (completed_list, final_obs).

        Pass A (from_start=True): always_reset_to_first_frame=True, logs eval video.
        Pass B (from_start=False): training reset behaviour (random tc), no video.
        The caller's finally block is responsible for restoring the original value.
        """
        import torch

        device = self.env.unwrapped.device
        num_envs = self.env.unwrapped.num_envs

        self._cmd.cfg.always_reset_to_first_frame = from_start

        my_ep_len = torch.zeros(num_envs, dtype=torch.float32, device=device)
        my_ep_max_len = torch.zeros(num_envs, dtype=torch.float32, device=device)
        env_first_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        completed: list[tuple[float, float]] = []

        # Warmup: step until every env has completed at least one episode so all
        # collected episodes start from a genuine post-reset state.
        env_reset_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
        while not env_reset_once.all():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = self.env.step(actions)
                policy_nn.reset(dones)
            dones = (
                dones.clone()
            )  # detach inference tensor before use outside inference_mode
            env_reset_once |= dones.bool()

        if log_video and self._record_video_env is not None:
            self._record_video_env.video_folder = self._eval_video_folder

        _video_triggered = False
        while len(completed) < self.eval_episodes:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = self.env.step(actions)
                policy_nn.reset(dones)
            dones = (
                dones.clone()
            )  # detach inference tensor before use outside inference_mode

            my_ep_len += 1
            done_mask = dones.bool()
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
                for i in done_ids:
                    if env_first_done[i]:
                        completed.append((my_ep_len[i].item(), my_ep_max_len[i].item()))
                    else:
                        # Tail episode pre-dating warmup end; discard, record next.
                        env_first_done[i] = True
                        if log_video and not _video_triggered:
                            self._isaac_env.eval_video_trigger_pending = True
                            _video_triggered = True
                    my_ep_len[i] = 0.0
                    my_ep_max_len[i] = self._cmd.tracking_lengths[i].float()

        # Drain: keep stepping until the in-progress recording finishes.
        if log_video and self._record_video_env is not None:
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

        return completed, obs

    def _compute_stats(
        self, completed: list[tuple[float, float]]
    ) -> tuple[float, float, float, int]:
        """Return (mean_ratio, std_ratio, full_pct, n_full) from a completed-episodes list."""
        data = completed[: self.eval_episodes]
        ratios = [
            min(max(e[0] - self._warmup_steps, 0), max(e[1] - self._warmup_steps, 1))
            / max(e[1] - self._warmup_steps, 1)
            for e in data
        ]
        n_full = sum(1 for r in ratios if r >= 0.99)
        mean_r = sum(ratios) / len(ratios)
        std_r = (
            sum((x - mean_r) ** 2 for x in ratios) / max(len(ratios) - 1, 1)
        ) ** 0.5
        full_pct = 100.0 * n_full / len(data)
        return mean_r, std_r, full_pct, n_full

    def __call__(self, path: str, *args, **kwargs) -> None:
        import copy as _copy

        import wandb

        if wandb.run is None:
            return

        match = self._re.search(r"model_(\d+)\.pt", os.path.basename(path))
        iteration = int(match.group(1)) if match else None

        device = self.env.unwrapped.device

        policy = self.runner.get_inference_policy(device=device)
        try:
            policy_nn = self.runner.alg.policy
        except AttributeError:
            policy_nn = self.runner.alg.actor_critic

        obs = self.env.get_observations()
        traj_len = self._cmd.retargeted_horizon

        # ── Save training state that eval steps would contaminate ──────────── #
        _saved_episode_sums = {
            k: v.clone()
            for k, v in self._isaac_env.reward_manager._episode_sums.items()
        }
        _saved_metrics = {k: v.clone() for k, v in self._cmd.metrics.items()}
        _saved_cws_buf = (
            _copy.copy(self._cmd._cws_reward_step_buf)
            if hasattr(self._cmd, "_cws_reward_step_buf")
            else None
        )
        _orig_reset_to_first = self._cmd.cfg.always_reset_to_first_frame

        completed_start: list[tuple[float, float]] = []
        completed_random: list[tuple[float, float]] = []

        try:
            # Pass A: from-start (tc=0) — also captures eval video.
            completed_start, obs = self._collect_episodes(
                policy, policy_nn, obs, from_start=True, log_video=self.log_video
            )
            # Pass B: random reset — measures coverage regardless of start position.
            completed_random, obs = self._collect_episodes(
                policy, policy_nn, obs, from_start=False, log_video=False
            )

        finally:
            self._cmd.cfg.always_reset_to_first_frame = _orig_reset_to_first

            # ── Restore training state saved before eval ───────────────────── #
            # Use assignment rather than copy_() because env.step() inside
            # inference_mode may have replaced dict entries with inference tensors;
            # inplace copy_ on those outside inference_mode raises an error.
            for k, v in _saved_episode_sums.items():
                if k in self._isaac_env.reward_manager._episode_sums:
                    self._isaac_env.reward_manager._episode_sums[k] = v.clone()
            for k, v in _saved_metrics.items():
                if k in self._cmd.metrics:
                    self._cmd.metrics[k] = v.clone()
            if _saved_cws_buf is not None and hasattr(
                self._cmd, "_cws_reward_step_buf"
            ):
                self._cmd._cws_reward_step_buf.clear()
                self._cmd._cws_reward_step_buf.extend(_saved_cws_buf)

            try:
                self.runner.alg.train_mode()
            except AttributeError:
                pass

            self._isaac_env.extras["log"] = dict()

            # Signal curriculum that eval has finished so deferred decay can fire.
            self._isaac_env.pre_decay_eval_pending = False

            # Always restore the train video folder and clear any unconsumed pending
            # trigger, regardless of whether an exception occurred above.
            if self.log_video:
                self._isaac_env.video_trigger_pending = False
                self._isaac_env.eval_video_trigger_pending = False
                if self._record_video_env is not None:
                    self._record_video_env.video_folder = self._train_video_folder

        # Log eval video (pass A only).
        if self.log_video and self._eval_video_folder:
            import glob as _glob

            new_videos = sorted(
                [
                    f
                    for f in _glob.glob(os.path.join(self._eval_video_folder, "*.mp4"))
                    if f not in self._logged_eval_videos
                ],
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
                self._logged_eval_videos.update(new_videos)
                self._logged_eval_videos.add(dst_path)

        # ── Stats and logging: pass A (from-start) ────────────────────────── #
        if completed_start:
            mean_r, std_r, full_pct, n_full = self._compute_stats(completed_start)
            n = len(completed_start[: self.eval_episodes])
            print(
                f"[eval/from_start] iter={iteration}  traj_len={traj_len}  "
                f"warmup={self._warmup_steps}  "
                f"sample ep_lens={[round(e[0]) for e in completed_start[:5]]}\n"
                f"  ratio={mean_r:.3f}±{std_r:.3f}  full={n_full}/{n} ({full_pct:.0f}%)"
            )
            log_data: dict = {
                "eval/completion_ratio_mean": mean_r,
                "eval/completion_ratio_std": std_r,
                "eval/full_completion_pct": full_pct,
            }
            if iteration is not None:
                wandb.log(log_data, step=iteration, commit=False)
            else:
                wandb.log(log_data, commit=False)

        # ── Stats and logging: pass B (random reset) ──────────────────────── #
        if completed_random:
            mean_r_rnd, std_r_rnd, _, _ = self._compute_stats(completed_random)
            print(
                f"[eval/random] iter={iteration}  "
                f"sample ep_lens={[round(e[0]) for e in completed_random[:5]]}\n"
                f"  ratio={mean_r_rnd:.3f}±{std_r_rnd:.3f}"
            )
            log_data_rnd: dict = {
                "eval/completion_ratio_mean_random": mean_r_rnd,
                "eval/completion_ratio_std_random": std_r_rnd,
            }
            if iteration is not None:
                wandb.log(log_data_rnd, step=iteration, commit=False)
            else:
                wandb.log(log_data_rnd, commit=False)

        # Refresh runner's internal obs so the next collect_rollouts starts consistently.
        if hasattr(self.runner, "obs"):
            self.runner.obs = obs
