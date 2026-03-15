from v2d.docker.container import run_in_container
from v2d.nlf.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_smpl_depth(
    smpl_params_path: str,
    intrinsics_path: str,
    output_depth_folder: str,
    output_mask_folder: str,
    weights_dir: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.nlf.lib.render_smpl_depth",
        inputs={"smpl_params_path": smpl_params_path, "intrinsics_path": intrinsics_path, "weights_dir": weights_dir},
        outputs={"output_depth_folder": output_depth_folder, "output_mask_folder": output_mask_folder},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL depth rendering in Docker")
    parser.add_argument("--smpl_params_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_depth_folder", required=True)
    parser.add_argument("--output_mask_folder", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_smpl_depth(
        args.smpl_params_path, args.intrinsics_path,
        args.output_depth_folder, args.output_mask_folder,
        args.weights_dir, dev=args.dev,
    )
