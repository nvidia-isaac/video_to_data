import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_ekf_smoothing(
    poses_dir: str,
    mesh_path: str,
    intrinsics_path: str,
    weights_dir: str,
    output_dir: str,
    masks_folder: str = None,
    process_noise_xy: float = 0.005,
    process_noise_z: float = 0.005,
    process_noise_r: float = 0.01,
    measurement_noise_xy: float = 0.02,
    measurement_noise_z: float = 0.1,
    measurement_noise_r: float = 0.05,
    min_iou: float = 0.1,
    dev: bool = False,
) -> None:
    weights_abs = os.path.abspath(weights_dir)
    weights_container = f"/data/weights_dir/{os.path.basename(weights_abs)}"
    inputs = {
        "poses_dir": poses_dir,
        "mesh_path": mesh_path,
        "intrinsics_path": intrinsics_path,
        "weights_dir": weights_dir,
    }
    if masks_folder is not None:
        inputs["masks_folder"] = masks_folder
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.run_ekf_smoothing",
        inputs=inputs,
        outputs={"output_dir": output_dir},
        extra_args={
            "process_noise_xy":     process_noise_xy,
            "process_noise_z":      process_noise_z,
            "process_noise_r":      process_noise_r,
            "measurement_noise_xy": measurement_noise_xy,
            "measurement_noise_z":  measurement_noise_z,
            "measurement_noise_r":  measurement_noise_r,
            "min_iou":              min_iou,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"FOUNDATIONPOSE_WEIGHTS_DIR": weights_container},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESKF + RTS pose smoother in Docker")
    parser.add_argument("--poses_dir",             required=True)
    parser.add_argument("--mesh_path",             required=True)
    parser.add_argument("--intrinsics_path",       required=True)
    parser.add_argument("--weights_dir",           required=True)
    parser.add_argument("--output_dir",            required=True)
    parser.add_argument("--masks_folder",          default=None)
    parser.add_argument("--process_noise_xy",      type=float, default=0.005)
    parser.add_argument("--process_noise_z",       type=float, default=0.005)
    parser.add_argument("--process_noise_r",       type=float, default=0.01)
    parser.add_argument("--measurement_noise_xy",  type=float, default=0.02)
    parser.add_argument("--measurement_noise_z",   type=float, default=0.1)
    parser.add_argument("--measurement_noise_r",   type=float, default=0.05)
    parser.add_argument("--min_iou",               type=float, default=0.1)
    parser.add_argument("--dev",                   action="store_true")
    args = parser.parse_args()
    run_ekf_smoothing(
        args.poses_dir,
        args.mesh_path,
        args.intrinsics_path,
        args.weights_dir,
        args.output_dir,
        masks_folder=args.masks_folder,
        process_noise_xy=args.process_noise_xy,
        process_noise_z=args.process_noise_z,
        process_noise_r=args.process_noise_r,
        measurement_noise_xy=args.measurement_noise_xy,
        measurement_noise_z=args.measurement_noise_z,
        measurement_noise_r=args.measurement_noise_r,
        min_iou=args.min_iou,
        dev=args.dev,
    )
