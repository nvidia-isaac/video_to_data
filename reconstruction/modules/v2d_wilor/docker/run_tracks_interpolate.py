from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_tracks_interpolate(
    aligned_dir: str,
    masks_dir: str,
    output_dir: str,
    betas: str = "fixed",
    max_gap_frames: int = 15,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.tracks_interpolate",
        inputs={
            "aligned_dir": aligned_dir,
            "masks_dir":   masks_dir,
        },
        outputs={"output_dir": output_dir},
        extra_args={
            "betas":          betas,
            "max_gap_frames": max_gap_frames,
        },
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
    parser.add_argument("--max_gap_frames", type=int, default=15)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_tracks_interpolate(
        aligned_dir    = args.aligned_dir,
        masks_dir      = args.masks_dir,
        output_dir     = args.output_dir,
        betas          = args.betas,
        max_gap_frames = args.max_gap_frames,
        dev            = args.dev,
    )
