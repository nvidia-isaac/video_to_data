from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_tracks_interpolate(
    aligned_dir: str,
    masks_dir: str,
    output_dir: str,
    betas: str = "fixed",
    max_gap_frames: int = 100000,
    extrapolate: bool = True,
    dev: bool = False,
) -> None:
    extra_args: dict = {
        "betas":          betas,
        "max_gap_frames": max_gap_frames,
    }
    if not extrapolate:
        extra_args["no_extrapolate"] = True
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.tracks_interpolate",
        inputs={
            "aligned_dir": aligned_dir,
            "masks_dir":   masks_dir,
        },
        outputs={"output_dir": output_dir},
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gap-fill per-track aligned hand records (SLERP rotations, linear cam_t)"
    )
    parser.add_argument("--aligned_dir",    required=True)
    parser.add_argument("--masks_dir",      required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--betas",          default="fixed", choices=("fixed", "interp"))
    parser.add_argument("--max_gap_frames", type=int, default=100000)
    parser.add_argument("--no_extrapolate", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_tracks_interpolate(
        aligned_dir    = args.aligned_dir,
        masks_dir      = args.masks_dir,
        output_dir     = args.output_dir,
        betas          = args.betas,
        max_gap_frames = args.max_gap_frames,
        extrapolate    = not args.no_extrapolate,
        dev            = args.dev,
    )
