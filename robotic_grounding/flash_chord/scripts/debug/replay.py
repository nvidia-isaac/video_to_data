# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a task parquet with a configured embodiment and zero-residual action.

Two modes:
    physical (default): contact on, soft gains -- shows the real grasp contact
    kinematic: contact off, stiff gains -- clean kinematic-fidelity replay

    python scripts/debug/replay.py "task.parquet='<.../robot_name=sharpa_wave>'"
    python scripts/debug/replay.py replay=kinematic action=residual_hand_pose_stiff \
      collision=kinematic embodiment=sharpa_hands_stiff
    python scripts/debug/replay.py --config-name=sonic_replay "task.parquet='<.../robot_name=g1>'"
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.visualization.markers import log_configured_markers
from flash_chord.visualization.viewer import ViewerApp, ViewerConfig


class Replay(ViewerApp):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        cfg = args.config
        embodiment = instantiate_typed(cfg.embodiment, Embodiment)
        collision = instantiate_typed(cfg.collision, CollisionPolicy)
        replay_cfg = instantiate_typed(cfg.replay, ReplayConfig)
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=replay_cfg.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min,
            object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
            motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
            motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
        )
        self.reference = setup.reference
        scene = setup.scene
        self.scene = scene
        self.runner = ReplayRunner(scene, self.reference, replay_cfg, device=cfg.device)
        self.runner.reset(0)
        self.model = scene.model
        self.state = self.runner.state_0
        self.frame = 0
        self._shown_frame = 0  # the reference frame currently in `self.state` (for markers)
        self.num_frames = self.reference.num_frames
        self._start_paused = bool(cfg.get("start_paused", False))
        self._requested_mode = None
        self._playing_mode = None if self._start_paused else "replay"
        self._replay_button = None
        self._reference_button = None
        self.viewer.set_model(self.model)
        if self._start_paused:
            print("Compiling replay kernels before enabling Viser playback...", flush=True)
            self.runner.step(0)
            self.runner.reset(0)
            self.state = self.runner.state_0
            self.frame = 0
            self._shown_frame = 0
            self.sim_time = 0.0
            print("Replay kernels are ready; playback remains paused at frame 0.", flush=True)
            self._install_playback_controls()
        action_details = f", action={type(self.runner.action).__name__}, action_dim={self.runner.action.action_dim}"
        encoder = getattr(self.runner.action, "encoder_session", None)
        decoder = getattr(self.runner.action, "decoder_session", None)
        if encoder is not None and decoder is not None:
            action_details += f", encoder={encoder.provider}, decoder={decoder.provider}"
        print(
            f"replay: {self.num_frames} frames, "
            f"robot_collision={collision.robot_scope}, "
            f"control={replay_cfg.sim.fps:g} Hz, physics={replay_cfg.sim.physics_fps:g} Hz, "
            f"motion_speed={cfg.task.motion_speed:g}, cone={replay_cfg.sim.cone}, "
            f"impratio={replay_cfg.sim.impratio:g}, VOC={replay_cfg.voc_scale:g}"
            f"{action_details}"
        )

    def _install_playback_controls(self) -> None:
        server = getattr(self.viewer, "_server", None)
        gui = getattr(server, "gui", None)
        if gui is None or not hasattr(gui, "add_button"):
            raise ValueError("start_paused requires the Viser backend's interactive GUI")
        rate = self.runner.config.sim.fps
        self._replay_button = gui.add_button(
            "Run configured replay",
            hint=f"Run all {self.num_frames} command frames with physics at {rate:g} Hz control.",
            color="green",
        )
        self._reference_button = gui.add_button(
            "Show kinematic reference",
            hint=f"Show all {self.num_frames} command frames at {rate:g} Hz without physics.",
            color="blue",
        )

        @self._replay_button.on_click
        def _request_replay(_event) -> None:
            self._request_playback("replay")

        @self._reference_button.on_click
        def _request_reference(_event) -> None:
            self._request_playback("reference")

        print("Playback is paused at frame 0; choose a replay mode in Viser.")

    def _request_playback(self, mode: str) -> None:
        if self._playing_mode is not None or self._requested_mode is not None:
            return
        self._requested_mode = mode
        self._set_controls_disabled(True)

    def _set_controls_disabled(self, disabled: bool) -> None:
        if self._replay_button is not None:
            self._replay_button.disabled = disabled
        if self._reference_button is not None:
            self._reference_button.disabled = disabled

    def _start_playback(self, mode: str) -> None:
        self.runner.reset(0)
        self.state = self.runner.state_0
        self.sim_time = 0.0
        self.frame = 0
        self._shown_frame = 0
        self._playing_mode = mode
        self._set_controls_disabled(True)
        print(f"Full {mode} started from frame 0 ({self.num_frames} frames at {self.runner.config.sim.fps:g} Hz).")

    def _finish_playback(self) -> None:
        mode = self._playing_mode
        self._playing_mode = None
        self._set_controls_disabled(False)
        print(f"Full {mode} reached frame {self._shown_frame} and is paused at the final pose.")

    def step(self) -> None:
        if self._playing_mode is None:
            if self._requested_mode is None:
                return
            requested_mode = self._requested_mode
            self._requested_mode = None
            self._start_playback(requested_mode)

        f = self.frame
        if self._playing_mode == "replay":
            self.runner.step(f)
            self.sim_time = self.runner.sim_time
        else:
            self.runner.reset(f)
            self.sim_time += self.runner.frame_dt
        self.state = self.runner.state_0
        self._shown_frame = f
        self.frame = f + 1
        if self.frame >= self.num_frames:
            if self._start_paused:
                self._finish_playback()
            else:  # preserve the original generic replay behavior
                self.frame = 0
                self.runner.reset(0)

    def render(self) -> None:
        self.pace_realtime()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        log_configured_markers(
            self.viewer,
            self.args.config.markers,
            self.runner.scene,
            self.reference,
            self._shown_frame,
            self.state,
            self.runner.contacts,
            self.runner.model,
            self.runner.device,
        )
        self.viewer.end_frame()


@hydra.main(version_base="1.3", config_path="../../src/flash_chord/configs", config_name="replay")
def main(cfg: DictConfig) -> None:
    """Launch a replay using one composed config and no experiment-specific argparse layer."""
    viewer = instantiate_typed(cfg.viewer, ViewerConfig)
    Replay.launch_configured(cfg, viewer, device=cfg.device)


if __name__ == "__main__":
    main()
