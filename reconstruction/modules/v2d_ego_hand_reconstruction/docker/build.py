"""Build Docker images for ego hand reconstruction (ViPE + Dyn-HaMR).

Delegates to the vendored shell scripts so build flags stay in sync with
upstream IsaacTeleop.
"""

import subprocess
from pathlib import Path

from v2d_ego_hand_reconstruction.docker._config import VENDOR_DIR


def _run_vendor_script(script: str, *args: str) -> None:
    vendor = Path(VENDOR_DIR)
    if not vendor.is_dir():
        raise FileNotFoundError(
            f"{vendor} not found. Run sync.sh first to fetch vendored sources."
        )
    subprocess.run(
        [str(vendor / "docker" / script), *args],
        check=True,
    )


def build_vipe() -> None:
    """Build the ViPE camera estimation image."""
    _run_vendor_script("vipe.sh", "build")


def build_dynhamr() -> None:
    """Build the Dyn-HaMR hand reconstruction image."""
    _run_vendor_script("dynhamr.sh", "build")


def build_all() -> None:
    """Build both ViPE and Dyn-HaMR images."""
    build_vipe()
    build_dynhamr()


if __name__ == "__main__":
    build_all()
