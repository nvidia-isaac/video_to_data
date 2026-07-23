from __future__ import annotations

from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from inpainting.parallel_jaw_renderer.bundle import load_robot_bundle
from inpainting.parallel_jaw_renderer.gripper import map_aperture_trajectory
from inpainting.parallel_jaw_renderer.kinematics import (
    KinematicsError,
    KinematicsResult,
    build_world_tcp_targets,
    validate_kinematics_quality,
)
from inpainting.parallel_jaw_renderer.render import (
    RENDER_METADATA_SCHEMA,
    build_frame_configuration,
)
from inpainting.robot_renderer.backend import (
    RENDER_METADATA_SCHEMA as ESTABLISHED_SCHEMA,
)
from inpainting.parallel_jaw_renderer.tests.helpers import make_bundle


class KinematicsAndRenderPureTests(unittest.TestCase):
    def test_metadata_schema_matches_existing_compositor_contract(self) -> None:
        self.assertEqual(RENDER_METADATA_SCHEMA, ESTABLISHED_SCHEMA)

    def test_semantic_rotation_is_right_composed(self) -> None:
        semantic = np.eye(4)[None, ...]
        rotation = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=float)
        inputs = SimpleNamespace(
            left_world_semantic=semantic,
            right_world_semantic=semantic,
        )
        bundle = SimpleNamespace(
            semantic_target_to_tcp_rotation={"left": rotation, "right": rotation}
        )
        targets = build_world_tcp_targets(inputs, bundle)
        np.testing.assert_allclose(targets["left"][0, :3, :3], rotation)

    def test_quality_gates_position_orientation_and_joint_step(self) -> None:
        kwargs = {
            "position_residuals_m": np.asarray((0.001, 0.002)),
            "orientation_residuals_deg": np.asarray((1.0, 2.0)),
            "joint_values": np.asarray(((0.0, 0.0), (0.1, 0.1))),
            "max_position_residual_m": 0.01,
            "max_orientation_residual_deg": 20.0,
            "max_joint_step_rad": 0.4,
        }
        result = validate_kinematics_quality(**kwargs)
        self.assertAlmostEqual(result[0], 0.002)
        with self.assertRaisesRegex(KinematicsError, "orientation residual"):
            validate_kinematics_quality(
                **{**kwargs, "orientation_residuals_deg": np.asarray((1.0, 21.0))}
            )
        with self.assertRaisesRegex(KinematicsError, "joint step"):
            validate_kinematics_quality(
                **{
                    **kwargs,
                    "joint_values": np.asarray(((0.0, 0.0), (0.5, 0.1))),
                }
            )

    def test_frame_configuration_commands_no_mimic_followers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = load_robot_bundle(
                make_bundle(Path(temporary)), require_visual_assets=False
            )
            left = map_aperture_trajectory(
                np.asarray((0.04,)),
                side="left",
                spec=bundle.gripper_mapping,
                render_inspection=bundle.render_inspection,
            )
            right = map_aperture_trajectory(
                np.asarray((0.04,)),
                side="right",
                spec=bundle.gripper_mapping,
                render_inspection=bundle.render_inspection,
            )
            kinematics = KinematicsResult(
                T_world_hub=np.eye(4),
                T_world_robot_root=np.eye(4),
                arm_joint_names=("left_arm", "right_arm"),
                arm_joint_values=np.asarray(((0.1, -0.1),), dtype=np.float32),
                max_position_residual_m=0.0,
                p95_position_residual_m=0.0,
                max_orientation_residual_deg=0.0,
                p95_orientation_residual_deg=0.0,
                max_joint_step_rad=0.0,
                external_ik={},
                ik_constructor_kwargs={},
            )
            configuration = build_frame_configuration(
                bundle, kinematics, left, right, 0
            )
            self.assertNotIn("left_mimic", configuration)
            self.assertNotIn("right_mimic", configuration)
            self.assertEqual(
                set(configuration),
                set(bundle.render_inspection.independent_joint_names),
            )


if __name__ == "__main__":
    unittest.main()
