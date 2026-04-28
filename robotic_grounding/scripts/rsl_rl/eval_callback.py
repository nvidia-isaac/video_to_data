"""Eval callback utility for RSL-RL training."""

import os


class EvalCallback:
    """Runs inference episodes after each checkpoint save and logs completion stats to wandb.

    Triggered by monkey-patching runner.save().  Temporarily forces
    always_reset_to_first_frame=True so every collected episode starts from
    the beginning of the trajectory (tc=0), making full_completion_pct a clean
    measure of whether the policy can complete the full trajectory from scratch.

    After a warm-up phase that waits until every env has reset at least once,
    it collects `eval_episodes` completed episodes in inference mode and logs:

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
            env_reset_once |= dones.bool()

        if log_video and self._record_video_env is not None:
            self._record_video_env.video_folder = self._eval_video_folder

        _video_triggered = False
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
        import torch
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

        # ── Save training state that eval steps would contaminate ──────────── #
        # All three are restored unconditionally in the finally block so that
        # training resumes with exactly the state it had before the checkpoint.
        import copy as _copy

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

        # Force all resets to start from the first trajectory frame for the
        # duration of eval; restored unconditionally in the finally block below.
        _orig_reset_to_first = self._cmd.cfg.always_reset_to_first_frame
        self._cmd.cfg.always_reset_to_first_frame = True

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
            # --- Trigger video recording for the upcoming eval rollout ---
            # Redirect RecordVideo to videos/eval so the clip is logged separately
            # from training videos (uploaded as eval/video rather than train/video).
            # Done inside try so the finally always restores video_folder, even if
            # the eval loop or drain throws (e.g. CUDA OOM, moviepy write error).
            if self.log_video and self._record_video_env is not None:
                self._record_video_env.video_folder = self._eval_video_folder

            # After warmup each env's episode is at some arbitrary midpoint.  We
            # discard each env's first post-warmup episode (the "tail"), then count
            # only full episodes that started fresh.  tracking_length is captured at
            # the START of each fresh episode so the per-episode denominator is
            # correct regardless of random start offsets.
            #
            # Video recording is triggered when the first tail ends — at that moment
            # the env resets to tc=0 (always_reset_to_first_frame=True), so the
            # recording captures a genuine from-frame-0 fresh episode.
            _video_triggered = False
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
                            completed.append(
                                (my_ep_len[i].item(), my_ep_max_len[i].item())
                            )
                        else:
                            # Tail of a mid-episode that pre-dated the warmup end;
                            # mark done so the NEXT episode is recorded.
                            env_first_done[i] = True
                            # The env just reset to tc=0; trigger recording now so
                            # the video captures a fresh from-frame-0 episode.
                            if self.log_video and not _video_triggered:
                                self._isaac_env.eval_video_trigger_pending = True
                                _video_triggered = True
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
            self._cmd.cfg.always_reset_to_first_frame = _orig_reset_to_first

            # ── Restore training state saved before eval ───────────────────── #
            # Restoring (not flushing) avoids both spikes AND dips:
            # • _episode_sums: eval steps accumulated inference-mode rewards;
            #   restore pre-eval sums so the next training episode completion logs
            #   the correct partial-episode reward, not zero or eval-quality values.
            #   (The previous fix called reward_manager.log() which does not exist —
            #   AttributeError silently aborted the entire finally block every time.)
            for k, v in _saved_episode_sums.items():
                if k in self._isaac_env.reward_manager._episode_sums:
                    self._isaac_env.reward_manager._episode_sums[k].copy_(v)
            # • self._cmd.metrics: _update_metrics() runs inside command_manager.compute()
            #   which is called AFTER _reset_idx in each env.step(). The first training
            #   episode reset therefore logs metrics from the *last eval step* (inference-
            #   mode, better performance) → spike in every Metrics/* key.
            for k, v in _saved_metrics.items():
                if k in self._cmd.metrics:
                    self._cmd.metrics[k].copy_(v)
            # • _cws_reward_step_buf: deque(maxlen=200) filled entirely with eval values
            #   → contact_wrench_support_reward_cv wrong for 200 training steps.
            if _saved_cws_buf is not None and hasattr(
                self._cmd, "_cws_reward_step_buf"
            ):
                self._cmd._cws_reward_step_buf.clear()
                self._cmd._cws_reward_step_buf.extend(_saved_cws_buf)

            # Restore policy to train mode.  get_inference_policy() calls
            # alg.eval_mode(); RSL-RL only calls train_mode() once at the start of
            # learn(), so without this the network stays in eval mode for all
            # subsequent training iterations after the first checkpoint.
            try:
                self.runner.alg.train_mode()
            except AttributeError:
                pass

            # Clear stale extras["log"]: the last eval-step reset left episode
            # metrics from inference mode in this dict; it persists until the next
            # training reset overwrites it, so the RSL-RL logger would process it.
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

        # Log eval video directly from the main thread so it is committed reliably.
        # Background-thread commit=False is not guaranteed to flush before the run ends.
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
                self._logged_eval_videos.update(
                    new_videos
                )  # mark all seen as tracked, not just the logged one
                self._logged_eval_videos.add(dst_path)  # also track renamed destination

        # --- Compute stats ---
        # ratio = fraction of the post-warmup trajectory that was completed.
        # Both numerator and denominator exclude warmup_steps so a perfectly
        # completing episode always gives ratio = 1.0.
        data = completed[: self.eval_episodes]
        print(
            f"[eval] iter={iteration}  traj_len={traj_len}  "
            f"warmup_steps={self._warmup_steps}  "
            f"sample ep_lens={[round(e[0]) for e in data[:5]]}"
        )
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
