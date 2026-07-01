"""Convert the Vega + dual-Sharpa humanoid URDF -> USD (mirrors the IIWA+Sharpa conversion).

Settings match assets/usd/iiwa14_left_sharpa/config.yaml (fix_base, merge_fixed_joints=False,
convex_hull colliders, self_collision off, instanceable) so the Vega robot drops into the
SimToolReal pipeline the same way the IIWA robot does. The Vega tasks use the LEFT arm + LEFT
Sharpa hand (the right arm/hand + torso are held at their default pose by the env's actuators).

Run:
  source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
    ~/simtoolreal_isaaclab/scripts/convert_vega_sharpa.py --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

REPO = "/home/cning/simtoolreal_isaaclab"
# DETAILED-visual URDF: arm/body visual meshes repointed from the decimated Open3D *.obj (~171 verts)
# to the full-resolution CAD *.dae (~27k verts). Collision is unchanged (convex hulls) so physics is
# unaffected -- only the rendered geometry is higher fidelity. (The plain *.obj URDF is kept alongside.)
DEFAULT_IN = f"{REPO}/assets/urdf/vega_sharpa/vega_sharpa_reduced_detailed.urdf"
DEFAULT_OUT = f"{REPO}/assets/usd/vega_sharpa/robot.usd"

parser = argparse.ArgumentParser(description="Convert the Vega+Sharpa robot URDF to USD.")
parser.add_argument("input", nargs="?", default=DEFAULT_IN, help="Input URDF path.")
parser.add_argument("output", nargs="?", default=DEFAULT_OUT, help="Output USD path.")
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
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest),
        usd_file_name=os.path.basename(dest),
        # mirror the IIWA+Sharpa conversion (assets/usd/iiwa14_left_sharpa/config.yaml)
        fix_base=True,                 # fixed-base humanoid manipulator (no balancing)
        merge_fixed_joints=False,      # keep named links (L_arm_l7 palm, left_*_DP fingertips, ...)
        # self-contained USD (NOT the instanceable base/physics split): the humanoid's instanceable
        # visual proxies produced broken `/visuals/<link>` cross-references (noisy recompose warnings);
        # a single-file conversion avoids them. Only one robot instance per env here, so no mem win lost.
        make_instanceable=False,
        link_density=0.0,
        force_usd_conversion=True,
        collider_type="convex_hull",
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
            target_type="position",
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[convert_vega_sharpa] Generated USD: {converter.usd_path}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
