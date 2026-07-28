from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np

from inpainting.robot_renderer.assets import (
    AssetError,
    inspect_urdf,
    validate_named_joint_trajectory,
)


class AssetTests(unittest.TestCase):
    def _one_joint_urdf(self, root: Path, mesh_ref: str = "mesh.stl") -> Path:
        mesh = root / "mesh.stl"
        mesh.write_text("solid mesh\nendsolid mesh\n")
        urdf = root / "robot.urdf"
        urdf.write_text(
            '<robot name="r"><link name="base"><visual><geometry>'
            f'<mesh filename="{mesh_ref}"/></geometry></visual></link>'
            '<link name="tip"/><joint name="joint" type="revolute">'
            '<parent link="base"/><child link="tip"/>'
            '<limit lower="-1" upper="1" effort="1" velocity="1"/>'
            '</joint></robot>'
        )
        return urdf

    def test_inspects_real_mesh_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection = inspect_urdf(self._one_joint_urdf(root), asset_root=root)
            self.assertEqual(inspection.actuated_joint_names, ("joint",))
            validate_named_joint_trajectory(
                np.array(((-1.0,), (1.0,))),
                np.array(("joint",)),
                inspection,
                label="test",
            )
            with self.assertRaisesRegex(AssetError, "above URDF upper"):
                validate_named_joint_trajectory(
                    np.array(((1.1,),)),
                    np.array(("joint",)),
                    inspection,
                    label="test",
                )

    def test_rejects_lfs_pointer_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            urdf = self._one_joint_urdf(root)
            (root / "mesh.stl").write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 20\n"
            )
            with self.assertRaisesRegex(AssetError, "Git LFS pointer"):
                inspect_urdf(urdf, asset_root=root)

    def test_rejects_mesh_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "assets"
            root.mkdir()
            outside = parent / "outside.stl"
            outside.write_text("solid x\nendsolid x\n")
            urdf = self._one_joint_urdf(root, "../outside.stl")
            with self.assertRaisesRegex(AssetError, "escapes asset root"):
                inspect_urdf(urdf, asset_root=root)

    def test_gltf_external_buffers_and_textures_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            urdf = self._one_joint_urdf(root, "mesh.gltf")
            (root / "mesh.gltf").write_text(
                '{"asset":{"version":"2.0"},'
                '"buffers":[{"uri":"mesh.bin","byteLength":4}],'
                '"images":[{"uri":"texture.png"}]}'
            )
            (root / "mesh.bin").write_bytes(b"mesh")
            with self.assertRaisesRegex(AssetError, "texture.png.*missing"):
                inspect_urdf(urdf, asset_root=root)
            (root / "texture.png").write_bytes(b"png")
            inspection = inspect_urdf(urdf, asset_root=root)
            self.assertEqual({path.name for path in inspection.mesh_paths}, {
                "mesh.gltf", "mesh.bin", "texture.png"
            })
            details = inspection.as_dict(asset_root=root)
            self.assertEqual(details["urdf_file"]["path"], "robot.urdf")
            self.assertEqual(
                [entry["path"] for entry in details["referenced_asset_files"]],
                ["mesh.bin", "mesh.gltf", "texture.png"],
            )
            self.assertEqual(
                details["urdf_file"]["sha256"],
                hashlib.sha256(urdf.read_bytes()).hexdigest(),
            )
            for entry in details["referenced_asset_files"]:
                self.assertEqual(
                    entry["sha256"],
                    hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
