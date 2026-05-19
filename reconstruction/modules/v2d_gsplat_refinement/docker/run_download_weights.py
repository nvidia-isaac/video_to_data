"""No-op weights downloader.

This module has no model weights of its own — gsplat ships its CUDA kernels
in the Docker image, manotorch loads MANO assets from the v2d_hamer weights
directory (caller passes ``--mano_assets_root``). Kept as a stub so the
build_containers.sh / weights-download orchestration is uniform across modules.
"""
from __future__ import annotations


def run_download(weights_path: str | None = None) -> None:
    print(f"v2d_gsplat_refinement has no weights to download (got: {weights_path}).")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights_path", default=None)
    args = p.parse_args()
    run_download(args.weights_path)
