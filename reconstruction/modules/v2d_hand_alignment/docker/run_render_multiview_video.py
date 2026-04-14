from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_multiview_video(
    mesh_path: str,
    poses_dir: str,
    hand_mesh_path: str,
    intrinsics_path: str,
    output_path: str,
    fps: float = 30.0,
    panel_w: int = 388,
    panel_h: int = 516,
    start: int = 0,
    end: int | None = None,
    frames_folder: str | None = None,
    dev: bool = False,
) -> None:
    inputs = {
        "mesh_path": mesh_path,
        "poses_dir": poses_dir,
        "hand_mesh_path": hand_mesh_path,
        "intrinsics_path": intrinsics_path,
    }
    if frames_folder is not None:
        inputs["frames_folder"] = frames_folder
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.render_multiview_video",
        inputs=inputs,
        outputs={"output_path": output_path},
        extra_args={
            "fps": fps,
            "panel_w": panel_w,
            "panel_h": panel_h,
            "start": start,
            "end": end,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
        env={"PYOPENGL_PLATFORM": "egl"},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--poses_dir", required=True)
    parser.add_argument("--hand_mesh_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--panel_w", type=int, default=388)
    parser.add_argument("--panel_h", type=int, default=516)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--frames_folder", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_multiview_video(
        args.mesh_path, args.poses_dir, args.hand_mesh_path, args.intrinsics_path,
        args.output_path,
        fps=args.fps,
        panel_w=args.panel_w,
        panel_h=args.panel_h,
        start=args.start,
        end=args.end,
        frames_folder=args.frames_folder,
        dev=args.dev,
    )
