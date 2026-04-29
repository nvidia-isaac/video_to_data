from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_dynhamr_video(
    world_results_path: str,
    frames_folder: str,
    mano_assets_root: str,
    output_path: str,
    fps: float = 25.0,
    start: int = 0,
    end: int | None = None,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.render_dynhamr_video",
        inputs={
            "world_results":    world_results_path,
            "frames_folder":    frames_folder,
            "mano_assets_root": mano_assets_root,
        },
        outputs={"output": output_path},
        extra_args={
            "fps":   fps,
            "start": start,
            "end":   end,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
        env={"PYOPENGL_PLATFORM": "egl"},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_results_path",  required=True)
    parser.add_argument("--frames_folder",        required=True)
    parser.add_argument("--mano_assets_root",     required=True)
    parser.add_argument("--output_path",          required=True)
    parser.add_argument("--fps",   type=float, default=25.0)
    parser.add_argument("--start", type=int,   default=0)
    parser.add_argument("--end",   type=int,   default=None)
    parser.add_argument("--dev",   action="store_true")
    args = parser.parse_args()
    run_render_dynhamr_video(
        world_results_path = args.world_results_path,
        frames_folder      = args.frames_folder,
        mano_assets_root   = args.mano_assets_root,
        output_path        = args.output_path,
        fps                = args.fps,
        start              = args.start,
        end                = args.end,
        dev                = args.dev,
    )
