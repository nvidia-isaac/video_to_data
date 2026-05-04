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
    use_trans_aligned: bool = True,
    object_mesh_path: str | None = None,
    object_poses_dir: str | None = None,
    dev: bool = False,
) -> None:
    extra: dict = {"fps": fps, "start": start, "end": end}
    if not use_trans_aligned:
        extra["no_trans_aligned"] = True
    inputs = {
        "world_results":    world_results_path,
        "frames_folder":    frames_folder,
        "mano_assets_root": mano_assets_root,
    }
    if object_mesh_path is not None:
        inputs["object_mesh_path"] = object_mesh_path
    if object_poses_dir is not None:
        inputs["object_poses_dir"] = object_poses_dir
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.render_dynhamr_video",
        inputs=inputs,
        outputs={"output": output_path},
        extra_args=extra,
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
    parser.add_argument("--use_trans_aligned",  dest="use_trans_aligned",
                        action="store_true",  default=True)
    parser.add_argument("--no_trans_aligned",   dest="use_trans_aligned",
                        action="store_false")
    parser.add_argument("--object_mesh_path",  default=None)
    parser.add_argument("--object_poses_dir",  default=None)
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
        use_trans_aligned  = args.use_trans_aligned,
        object_mesh_path   = args.object_mesh_path,
        object_poses_dir   = args.object_poses_dir,
        dev                = args.dev,
    )
