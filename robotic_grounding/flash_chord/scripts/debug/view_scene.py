# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a scene (embodiment + a sequence's objects) and display it at reference frame 0.

    python scripts/debug/view_scene.py --parquet <.../robot_name=sharpa_wave>
"""

import flash_chord.embodiments.sharpa_hands
import flash_chord.embodiments.vega_sharpa  # noqa: F401  (register)
from flash_chord.embodiments.base import get_embodiment
from flash_chord.lifecycle.reset import reset_scene_to_frame
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.visualization.viewer import ViewerApp


class ViewScene(ViewerApp):
    """Embodiment + the sequence's objects (objects at reference frame 0; robot at default pose)."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        embodiment = get_embodiment(args.embodiment)()
        setup = setup_scene(
            parquet=args.parquet,
            control_fps=None,
            motion_speed=1.0,
            embodiment=embodiment,
            collision=CollisionPolicy(),
            world_count=1,
            include_support=not args.no_support,
            decompose_objects=True,
        )
        reference = setup.reference
        scene = setup.scene
        self.model = scene.model
        self.state = self.model.state()
        reset_scene_to_frame(scene, reference, args.frame, self.state)
        self.viewer.set_model(self.model)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--embodiment", default="sharpa_hands")
        parser.add_argument("--parquet", required=True, help="reference parquet dir/file")
        parser.add_argument("--no-support", action="store_true", help="skip the support surface")
        parser.add_argument("--frame", type=int, default=0, help="reference frame to pose the scene to")


if __name__ == "__main__":
    ViewScene.launch()
