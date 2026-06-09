"""Convert a DexToolBench object URDF -> USD with CONVEX DECOMPOSITION collision.

Mirrors IsaacLab/scripts/tools/convert_urdf.py but sets
``collider_type="convex_decomposition"`` so the collision shape is decomposed into
multiple convex pieces (a closer fit than a single convex hull). This matches the
original DexToolBench objects (need_vhacd=True) and captures concavities — e.g. the
claw hammer's claw/head and thin screwdriver/marker — which improves grasp fidelity.

Run:
  cd IsaacLab && (venv) OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
      ~/simtoolreal_isaaclab/scripts/convert_object.py <input.urdf> <output.usd> --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert an object URDF to USD with convex decomposition collision.")
parser.add_argument("input", type=str, help="Path to the input URDF file.")
parser.add_argument("output", type=str, help="Path to store the output USD file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402


def main():
    urdf_path = os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    dest = os.path.abspath(args_cli.output)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest),
        usd_file_name=os.path.basename(dest),
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        collider_type="convex_decomposition",  # <-- the change vs the default convex_hull
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
            target_type="position",
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[convex_decomposition] Generated USD: {converter.usd_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
