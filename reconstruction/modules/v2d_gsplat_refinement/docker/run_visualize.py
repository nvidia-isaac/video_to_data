"""Interactive viewer for a trained gsplat checkpoint.

Docker orchestration around ``v2d.gsplat_refinement.lib.visualize``. Runs the
container in interactive mode with X11 forwarding so OpenCV's ``imshow``
window appears on the host display. Tested on Linux hosts; for remote/SSH
sessions you'll need to forward X11 (``ssh -X``) before invoking.

If you see ``cv2.error: ... could not connect to display`` make sure
``$DISPLAY`` is set on the host and run::

    xhost +local:docker     # one-off, allows containers to talk to your X server

Usage:
    python -m v2d.gsplat_refinement.docker.run_visualize \\
        --checkpoint        data/clean/03_undist/output_wilor/refine_checkpoint.pt \\
        --intrinsics_path   data/clean/03_undist/output_wilor/intrinsics_stable.json \\
        --mano_assets_root  data/weights/wilor/pretrained_models
"""

import argparse
import os
import subprocess

from v2d.gsplat_refinement.docker._config import IMAGE_NAME, MODULES_DIR


def run_visualize(
    checkpoint: str,
    intrinsics_path: str,
    mano_assets_root: str,
    width: int | None = None,
    height: int | None = None,
    fps: float = 30.0,
    dev: bool = False,
) -> None:
    checkpoint       = os.path.abspath(checkpoint)
    intrinsics_path  = os.path.abspath(intrinsics_path)
    mano_assets_root = os.path.abspath(mano_assets_root)

    # One bind mount per distinct host directory.
    ckpt_dir  = os.path.dirname(checkpoint)
    intr_dir  = os.path.dirname(intrinsics_path)
    mano_dir  = mano_assets_root

    # Container-side paths (claim the bind names by argument order).
    dir_mounts: dict[str, str] = {}
    def _bind(host_dir: str, mount_name: str) -> str:
        if host_dir not in dir_mounts:
            dir_mounts[host_dir] = f"/data/{mount_name}"
        return dir_mounts[host_dir]
    ckpt_mount = _bind(ckpt_dir, "checkpoint")
    intr_mount = _bind(intr_dir, "intrinsics")
    mano_mount = _bind(mano_dir, "mano")
    ckpt_in_c  = f"{ckpt_mount}/{os.path.basename(checkpoint)}"
    intr_in_c  = f"{intr_mount}/{os.path.basename(intrinsics_path)}"

    display = os.environ.get("DISPLAY", "")
    if not display:
        raise RuntimeError(
            "No $DISPLAY in the environment. The viewer needs an X11 display "
            "to forward into the container. On a local Linux session this is "
            "set automatically; over SSH use `ssh -X`. You may also need to "
            "run `xhost +local:docker` once on the host."
        )

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-e", f"DISPLAY={display}",
        "-e", "QT_X11_NO_MITSHM=1",
        # X11 socket + Xauthority for non-root user access.
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
    ]
    xauth = os.environ.get("XAUTHORITY")
    if xauth and os.path.exists(xauth):
        cmd += ["-v", f"{xauth}:{xauth}:ro", "-e", f"XAUTHORITY={xauth}"]

    for host_dir, container_dir in dir_mounts.items():
        cmd += ["-v", f"{host_dir}:{container_dir}"]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]

    cmd += [IMAGE_NAME, "python", "-m", "v2d.gsplat_refinement.lib.visualize",
            "--checkpoint",       ckpt_in_c,
            "--intrinsics_path",  intr_in_c,
            "--mano_assets_root", mano_mount]
    if width  is not None: cmd += ["--width",  str(int(width))]
    if height is not None: cmd += ["--height", str(int(height))]
    if fps is not None:    cmd += ["--fps",    str(float(fps))]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive gsplat-checkpoint viewer")
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--intrinsics_path",  required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--width",  type=int,   default=None)
    parser.add_argument("--height", type=int,   default=None)
    parser.add_argument("--fps",    type=float, default=30.0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_visualize(
        checkpoint       = args.checkpoint,
        intrinsics_path  = args.intrinsics_path,
        mano_assets_root = args.mano_assets_root,
        width            = args.width,
        height           = args.height,
        fps              = args.fps,
        dev              = args.dev,
    )
