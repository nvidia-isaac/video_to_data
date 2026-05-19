from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_hands_video(
    frames_dir: str,
    wilor_dir: str,
    mano_assets_root: str,
    output_path: str,
    fps: float = 30.0,
    alpha: float = 0.55,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.render_hands_video",
        inputs={
            "frames_dir":       frames_dir,
            "wilor_dir":        wilor_dir,
            "mano_assets_root": mano_assets_root,
        },
        outputs={"output_path": output_path},
        extra_args={"fps": fps, "alpha": alpha},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render projected WiLoR MANO meshes onto source frames")
    parser.add_argument("--frames_dir",       required=True)
    parser.add_argument("--wilor_dir",        required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_path",      required=True)
    parser.add_argument("--fps",   type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_hands_video(
        frames_dir       = args.frames_dir,
        wilor_dir        = args.wilor_dir,
        mano_assets_root = args.mano_assets_root,
        output_path      = args.output_path,
        fps              = args.fps,
        alpha            = args.alpha,
        dev              = args.dev,
    )
