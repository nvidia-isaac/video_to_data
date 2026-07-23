from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from inpainting.parallel_jaw_renderer.bundle import (
    BundleError,
    GripperMappingSpec,
    JointLimit,
    UrdfInspection,
    load_robot_bundle,
)
from inpainting.parallel_jaw_renderer.gripper import map_aperture_trajectory
from inpainting.parallel_jaw_renderer.tests.helpers import make_bundle


class BundleAndGripperTests(unittest.TestCase):
    def test_bundle_normalizes_root_transform_and_excludes_mimics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = make_bundle(Path(temporary))
            bundle = load_robot_bundle(path, require_visual_assets=False)
            self.assertEqual(
                set(bundle.render_inspection.mimic_joint_names),
                {"left_mimic", "right_mimic"},
            )
            self.assertNotIn(
                "left_mimic", bundle.render_inspection.independent_joint_names
            )
            T_world_hub = np.eye(4)
            T_world_hub[0, 3] = 10.0
            T_world_root = bundle.world_robot_root(T_world_hub)
            self.assertAlmostEqual(T_world_root[0, 3], 9.0)

    def test_bundle_rejects_provenance_record_in_path_field_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = make_bundle(Path(temporary))
            payload = json.loads(path.read_text())
            payload["render_urdf"] = {"path": "render.urdf", "sha256": "0" * 64}
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(BundleError, "must be a path string"):
                load_robot_bundle(path, require_visual_assets=False)

    def test_galbot_nonlinear_mapping_matches_source_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = load_robot_bundle(
                make_bundle(Path(temporary)), require_visual_assets=False
            )
            maximum = 2.0 * (0.026 - 0.0062 + 0.045 * np.sin(1.2465))
            result = map_aperture_trajectory(
                np.asarray((0.0, maximum, maximum + 0.1)),
                side="left",
                spec=bundle.gripper_mapping,
                render_inspection=bundle.render_inspection,
            )
            expected_closed = 1.2465 - np.arcsin((-(0.026 - 0.0062)) / 0.045)
            self.assertAlmostEqual(result.values[0, 0], expected_closed, places=8)
            self.assertAlmostEqual(result.values[1, 0], 0.0, places=6)
            self.assertEqual(result.report["clipped_frame_count"], 1)
            self.assertEqual(result.names, ("left_gripper",))

    def test_yam_mirrored_prismatic_mapping_drives_both_fingers(self) -> None:
        limits = {
            "left_finger": JointLimit("prismatic", -0.0475, 0.0),
            "right_finger": JointLimit("prismatic", -0.0475, 0.0),
        }
        inspection = UrdfInspection(
            path=Path("/synthetic/yam.urdf"),
            links=(),
            joints=tuple(limits),
            independent_joint_names=tuple(limits),
            mimic_joint_names=(),
            joint_limits=limits,
            visual_mesh_paths=(),
        )
        spec = GripperMappingSpec(
            kind="mirrored_prismatic",
            joint_names={
                "left": ("left_finger", "right_finger"),
                "right": ("left_finger", "right_finger"),
            },
            params={
                "closed_aperture_m": 0.0,
                "open_aperture_m": 0.095,
                "closed_joint_position_m": 0.0,
                "open_joint_position_m": -0.0475,
            },
        )
        result = map_aperture_trajectory(
            np.asarray((0.0, 0.0475, 0.095, 0.2)),
            side="left",
            spec=spec,
            render_inspection=inspection,
        )
        np.testing.assert_allclose(
            result.values,
            (
                (0.0, 0.0),
                (-0.02375, -0.02375),
                (-0.0475, -0.0475),
                (-0.0475, -0.0475),
            ),
        )
        self.assertEqual(result.report["clipped_frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
