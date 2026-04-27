"""Run ego hand reconstruction pipeline (ViPE camera estimation + Dyn-HaMR).

Delegates to the vendored ``run_reconstruction.sh`` so pipeline logic stays in
sync with upstream IsaacTeleop.

Prerequisites:
    - Both Docker images must be built (``python -m v2d_ego_hand_reconstruction.docker.build``).
    - MANO_RIGHT.pkl and BMC/*.npy must be placed in *output_dir*
      (see ``vendor/doc/quickstart.md`` for download instructions).
"""

import os
import shutil
import subprocess

from v2d_ego_hand_reconstruction.docker._config import VENDOR_DIR


def run_reconstruction(
    video_input: str,
    output_dir: str,
    weights_dir: str,
) -> None:
    """Run the full ego hand reconstruction pipeline.

    Args:
        video_input: Local file path, or ``s3://`` / ``swift://`` URL.
        output_dir: Directory for all pipeline outputs.
        weights_dir: Directory containing MANO_RIGHT.pkl and BMC/ folder.
    """
    output_dir = os.path.abspath(output_dir)
    weights_dir = os.path.abspath(weights_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Copy weights into output_dir so the container can find them.
    # Symlinking doesn't work because the target is outside the container's mount.
    mano_src = os.path.join(weights_dir, "MANO_RIGHT.pkl")
    mano_dst = os.path.join(output_dir, "MANO_RIGHT.pkl")
    if not os.path.exists(mano_dst):
        shutil.copy2(mano_src, mano_dst)

    bmc_src = os.path.join(weights_dir, "BMC")
    bmc_dst = os.path.join(output_dir, "BMC")
    if not os.path.exists(bmc_dst):
        shutil.copytree(bmc_src, bmc_dst)

    # Resolve local paths to absolute so the vendored script can find them
    if not video_input.startswith(("s3://", "swift://")):
        video_input = os.path.abspath(video_input)

    subprocess.run(
        [os.path.join(VENDOR_DIR, "scripts", "run_reconstruction.sh"), video_input],
        env={**os.environ, "OUTPUTS_DIR": output_dir},
        check=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video_input",
        required=True,
        help="Local path or s3:///swift:// URL",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--weights_dir", required=True, help="Directory with MANO_RIGHT.pkl and BMC/")
    args = parser.parse_args()
    run_reconstruction(args.video_input, args.output_dir, args.weights_dir)
