from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from inpainting.robot_assets.yam_usd_to_urdf import (
    ArmModel,
    BUNDLE_SCHEMA,
    EXPECTED_BODY_NAMES,
    FINGER_JOINTS,
    JointSpec,
    MANIFEST_NAME,
    RENDER_URDF_NAME,
    IK_URDF_NAME,
    TCP_OFFSET_LINK_6,
    YAM_MOUNT_FORWARD_X_M,
    YAM_MOUNT_ROLL_DEG,
    YAM_SEMANTIC_TARGET_TO_TCP_ROTATION,
    YAM_T_HUB_ROBOT_ROOT,
    YAM_T_ROBOT_ROOT_HUB,
    _build_urdf,
    convert_yam_usd,
    sha256_file,
)


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _joint(
    name: str,
    parent: str,
    child: str,
    *,
    joint_type: str = "revolute",
    axis: tuple[float, float, float] | None = (0.0, 0.0, 1.0),
    lower: float | None = -1.0,
    upper: float | None = 1.0,
) -> JointSpec:
    return JointSpec(
        name=name,
        parent=parent,
        child=child,
        joint_type=joint_type,
        parent_from_child_zero=_matrix_tuple(np.eye(4)),
        axis_child=axis,
        lower=lower,
        upper=upper,
        effort=10.0 if joint_type != "fixed" else None,
        velocity=100.0 if joint_type != "fixed" else None,
        source_axis="X" if joint_type != "fixed" else None,
        source_lower=math.degrees(lower) if joint_type == "revolute" else lower,
        source_upper=math.degrees(upper) if joint_type == "revolute" else upper,
        authored_zero_pose_max_abs_residual=0.0,
    )


def _synthetic_model() -> ArmModel:
    joints: list[JointSpec] = []
    parent = "arm"
    for index in range(1, 7):
        child = f"link_{index}"
        joints.append(
            _joint(
                f"joint{index}",
                parent,
                child,
                lower=math.radians(-150.0) if index == 1 else -1.0,
                upper=math.radians(175.0) if index == 1 else 1.0,
            )
        )
        parent = child
    for finger in sorted(FINGER_JOINTS):
        joints.append(
            _joint(
                finger,
                "link_6",
                finger,
                joint_type="prismatic",
                axis=(1.0, 0.0, 0.0),
                lower=-0.0475,
                upper=0.0,
            )
        )
    return ArmModel(
        bodies=EXPECTED_BODY_NAMES,
        visuals=(),
        joints=tuple(joints),
        meters_per_unit=1.0,
        finger_open_position=-0.0475,
        finger_closed_position=0.0,
        closed_aperture_m=0.002,
        open_aperture_m=0.095,
        fingertip_keypoints={
            "left_finger": (-0.088, 0.025, -0.045),
            "right_finger": (-0.025, -0.088, -0.045),
        },
    )


def _joint_element(root: ET.Element, name: str) -> ET.Element:
    result = root.find(f"./joint[@name='{name}']")
    if result is None:
        raise AssertionError(f"missing joint {name}")
    return result


class YamUrdfBuilderTests(unittest.TestCase):
    def test_import_keeps_usd_and_mesh_dependencies_lazy(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import inpainting.robot_assets.yam_usd_to_urdf; "
                    "assert 'pxr' not in sys.modules; "
                    "assert 'trimesh' not in sys.modules"
                ),
            ],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_render_and_ik_variants_have_exact_bimanual_structure(self) -> None:
        model = _synthetic_model()
        render = ET.fromstring(_build_urdf(model, fixed_open_fingers=False))
        ik = ET.fromstring(_build_urdf(model, fixed_open_fingers=True))

        render_moving = [
            joint.attrib["name"]
            for joint in render.findall("joint")
            if joint.attrib["type"] != "fixed"
        ]
        ik_moving = [
            joint.attrib["name"]
            for joint in ik.findall("joint")
            if joint.attrib["type"] != "fixed"
        ]
        self.assertEqual(
            render_moving,
            [
                *(f"left_joint{index}" for index in range(1, 7)),
                "left_left_finger",
                "left_right_finger",
                *(f"right_joint{index}" for index in range(1, 7)),
                "right_left_finger",
                "right_right_finger",
            ],
        )
        self.assertEqual(
            ik_moving,
            [
                *(f"left_joint{index}" for index in range(1, 7)),
                *(f"right_joint{index}" for index in range(1, 7)),
            ],
        )

        left_mount = _joint_element(render, "left_mount").find("origin")
        right_mount = _joint_element(render, "right_mount").find("origin")
        assert left_mount is not None and right_mount is not None
        self.assertEqual(
            tuple(float(value) for value in left_mount.attrib["xyz"].split()),
            (0.0, 0.305, 0.0),
        )
        self.assertEqual(
            tuple(float(value) for value in right_mount.attrib["xyz"].split()),
            (0.0, -0.305, 0.0),
        )

        limit = _joint_element(render, "left_joint1").find("limit")
        assert limit is not None
        self.assertAlmostEqual(float(limit.attrib["lower"]), math.radians(-150.0))
        self.assertAlmostEqual(float(limit.attrib["upper"]), math.radians(175.0))

        render_finger = _joint_element(render, "left_left_finger")
        ik_finger = _joint_element(ik, "left_left_finger")
        self.assertEqual(render_finger.attrib["type"], "prismatic")
        self.assertEqual(ik_finger.attrib["type"], "fixed")
        ik_origin = ik_finger.find("origin")
        assert ik_origin is not None
        self.assertEqual(
            tuple(float(value) for value in ik_origin.attrib["xyz"].split()),
            (-0.0475, 0.0, 0.0),
        )

        for side in ("left", "right"):
            tcp = _joint_element(ik, f"{side}_tcp_joint")
            origin = tcp.find("origin")
            assert origin is not None
            self.assertEqual(
                tuple(float(value) for value in origin.attrib["xyz"].split()),
                TCP_OFFSET_LINK_6,
            )
            self.assertEqual(tcp.find("parent").attrib["link"], f"{side}_link_6")
            self.assertEqual(tcp.find("child").attrib["link"], f"{side}_tcp")

    def test_fixed_parallel_jaw_retarget_calibration_is_exact_and_rigid(self) -> None:
        root_from_hub = np.asarray(YAM_T_ROBOT_ROOT_HUB, dtype=np.float64)
        hub_from_root = np.asarray(YAM_T_HUB_ROBOT_ROOT, dtype=np.float64)
        np.testing.assert_allclose(
            root_from_hub @ hub_from_root,
            np.eye(4),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            hub_from_root @ root_from_hub,
            np.eye(4),
            atol=1e-12,
        )
        self.assertAlmostEqual(
            hub_from_root[0, 3],
            YAM_MOUNT_FORWARD_X_M,
        )
        recovered_roll = math.degrees(
            math.atan2(hub_from_root[2, 1], hub_from_root[1, 1])
        )
        self.assertAlmostEqual(recovered_roll, YAM_MOUNT_ROLL_DEG)
        self.assertAlmostEqual(np.linalg.det(hub_from_root[:3, :3]), 1.0)

        np.testing.assert_array_equal(
            np.asarray(YAM_SEMANTIC_TARGET_TO_TCP_ROTATION["left"]),
            np.diag([-1.0, -1.0, 1.0]),
        )
        np.testing.assert_array_equal(
            np.asarray(YAM_SEMANTIC_TARGET_TO_TCP_ROTATION["right"]),
            np.eye(3),
        )
        for rotation in YAM_SEMANTIC_TARGET_TO_TCP_ROTATION.values():
            matrix = np.asarray(rotation, dtype=np.float64)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(matrix), 1.0)


@unittest.skipUnless(
    os.environ.get("YAMLAB_TEST_REPOSITORY"),
    "set YAMLAB_TEST_REPOSITORY to run the pinned-asset integration test",
)
class YamPinnedAssetIntegrationTests(unittest.TestCase):
    def test_pinned_asset_converts_and_matches_renderer_bundle_contract(self) -> None:
        repository = Path(os.environ["YAMLAB_TEST_REPOSITORY"]).resolve()
        source = repository / "yamlab/robot/yam/arm/yam.usd"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            result = convert_yam_usd(
                source_usd=source,
                input_repository=repository,
                output_dir=output,
                runtime_validation=True,
            )
            self.assertEqual(result.render_urdf.name, RENDER_URDF_NAME)
            self.assertEqual(result.ik_urdf.name, IK_URDF_NAME)
            self.assertEqual(result.manifest.name, MANIFEST_NAME)

            manifest = json.loads(result.manifest.read_text())
            self.assertEqual(manifest["schema_version"], BUNDLE_SCHEMA)
            self.assertEqual(manifest["render_urdf"], RENDER_URDF_NAME)
            self.assertEqual(manifest["ik_urdf"], IK_URDF_NAME)
            self.assertEqual(manifest["tcp_frames"], {
                "left": "left_tcp",
                "right": "right_tcp",
            })
            np.testing.assert_allclose(
                manifest["T_robot_root_hub"],
                YAM_T_ROBOT_ROOT_HUB,
                atol=1e-12,
            )
            self.assertEqual(
                manifest["semantic_target_to_tcp_rotation"],
                {
                    side: [list(row) for row in rotation]
                    for side, rotation in (
                        YAM_SEMANTIC_TARGET_TO_TCP_ROTATION.items()
                    )
                },
            )
            calibration = manifest["asset_provenance"]["conversion"][
                "parallel_jaw_retarget_calibration"
            ]
            self.assertEqual(
                calibration["mount"]["T_hub_robot_root"],
                [list(row) for row in YAM_T_HUB_ROBOT_ROOT],
            )
            self.assertEqual(
                calibration["semantic_target_to_tcp_rotation"],
                manifest["semantic_target_to_tcp_rotation"],
            )
            self.assertEqual(
                calibration["evaluation_context"]["tracker_conditions"],
                ["ground_truth", "v2d", "phantom"],
            )
            self.assertEqual(len(manifest["arm_joint_names"]), 12)
            self.assertEqual(
                manifest["gripper_mapping"]["kind"], "mirrored_prismatic"
            )
            params = manifest["gripper_mapping"]["params"]
            self.assertGreater(params["open_aperture_m"], params["closed_aperture_m"])
            self.assertAlmostEqual(params["closed_aperture_m"], 0.002004, places=5)
            self.assertAlmostEqual(params["open_aperture_m"], 0.094901, places=5)
            self.assertEqual(
                manifest["fixed_root_posture"]["joint_values"], {}
            )

            for record in manifest["asset_provenance"]["output_files"]:
                payload = output / record["path"]
                self.assertEqual(payload.stat().st_size, record["bytes"])
                self.assertEqual(sha256_file(payload), record["sha256"])

            from inpainting.parallel_jaw_renderer.bundle import load_robot_bundle

            bundle = load_robot_bundle(result.manifest)
            self.assertEqual(len(bundle.render_inspection.independent_joint_names), 16)
            self.assertEqual(len(bundle.ik_inspection.independent_joint_names), 12)
            self.assertEqual(len(bundle.render_inspection.visual_mesh_paths), 17)


if __name__ == "__main__":
    unittest.main()
