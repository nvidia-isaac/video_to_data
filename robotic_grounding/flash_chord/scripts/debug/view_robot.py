# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load an embodiment and display it at its default pose in the Newton viewer.

python scripts/debug/view_robot.py --embodiment sharpa_hands
python scripts/debug/view_robot.py --embodiment dexmate_sharpa --viewer gl
"""

import newton

# import the concrete embodiments so they register in EMBODIMENT_REGISTRY
import flash_chord.embodiments.g1_dex3
import flash_chord.embodiments.sharpa_hands  # noqa: F401
import flash_chord.embodiments.vega_sharpa  # noqa: F401
from flash_chord.embodiments.base import EMBODIMENT_REGISTRY, get_embodiment
from flash_chord.visualization.viewer import ViewerApp


class ViewRobot(ViewerApp):
    """Build one copy of an embodiment and show it at its default joint pose (static)."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        embodiment = get_embodiment(args.embodiment)()
        builder = newton.ModelBuilder()
        embodiment.build(builder)
        self.model = builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.set_model(self.model)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--embodiment",
            default="sharpa_hands",
            help=f"registered embodiment name; one of {sorted(EMBODIMENT_REGISTRY)}",
        )


if __name__ == "__main__":
    ViewRobot.launch()
