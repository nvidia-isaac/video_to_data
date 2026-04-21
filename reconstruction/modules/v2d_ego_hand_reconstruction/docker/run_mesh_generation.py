"""Export per-track MANO hand meshes and joint trajectories from Dyn-HaMR results.

Delegates to the vendored ``run_mesh_generation.sh`` so mesh export logic stays
in sync with upstream IsaacTeleop.

Prerequisites:
    - Dyn-HaMR Docker image must be built (``python -m v2d_ego_hand_reconstruction.docker.build``).
    - MANO_RIGHT.pkl must be placed in *output_dir*.
"""

import os
import subprocess

from v2d_ego_hand_reconstruction.docker._config import VENDOR_DIR


def run_mesh_generation(
    output_dir: str,
    *,
    phase: str = "smooth_fit",
    gpu: int = 0,
    no_temporal_smooth: bool = False,
    no_smooth_trans: bool = False,
) -> None:
    """Run hand mesh export over every Dyn-HaMR run under *output_dir*.

    Args:
        output_dir: Directory containing Dyn-HaMR
            ``logs/.../smooth_fit/*_world_results.npz``.
        phase: Phase subdir under each run dir (default: ``smooth_fit``).
        gpu: GPU index inside the container.
        no_temporal_smooth: Disable OneEuro smoothing on poses.
        no_smooth_trans: Smooth pose only, not translation.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    env = {
        **os.environ,
        "OUTPUTS_DIR": output_dir,
        "PHASE": phase,
        "GPU": str(gpu),
        "NO_TEMPORAL_SMOOTH": "1" if no_temporal_smooth else "0",
        "NO_SMOOTH_TRANS": "1" if no_smooth_trans else "0",
    }

    subprocess.run(
        [os.path.join(VENDOR_DIR, "scripts", "run_mesh_generation.sh")],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, help="Output directory (contains logs/)")
    parser.add_argument("--phase", default="smooth_fit")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_temporal_smooth", action="store_true")
    parser.add_argument("--no_smooth_trans", action="store_true")
    args = parser.parse_args()
    run_mesh_generation(
        args.output_dir,
        phase=args.phase,
        gpu=args.gpu,
        no_temporal_smooth=args.no_temporal_smooth,
        no_smooth_trans=args.no_smooth_trans,
    )
