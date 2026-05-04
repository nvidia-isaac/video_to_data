from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_align_world_results(
    input_hand_data: str,
    depth_dir: str,
    depth_intrinsics: str,
    mano_model_dir: str,
    output_hand_data: str,
    object_masks_dir: str | None = None,
    object_poses_dir: str | None = None,
    smooth_sigma: float = 5.0,
    dev: bool = False,
) -> None:
    inputs: dict = {
        "input_hand_data":  input_hand_data,
        "depth_dir":        depth_dir,
        "depth_intrinsics": depth_intrinsics,
        "mano_model_dir":   mano_model_dir,
    }
    if object_masks_dir is not None:
        inputs["object_masks_dir"] = object_masks_dir
    if object_poses_dir is not None:
        inputs["object_poses_dir"] = object_poses_dir

    outputs: dict = {"output_hand_data": output_hand_data}

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.align_world_results",
        inputs=inputs,
        outputs=outputs,
        extra_args={"smooth_sigma": smooth_sigma if smooth_sigma > 0 else None},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Align DynHaMR world_results.npz to monocular depth."
    )
    parser.add_argument("--input_hand_data",  required=True,
                        help="DynHaMR world_results.npz")
    parser.add_argument("--depth_dir",        required=True,
                        help="Folder of depth PNGs")
    parser.add_argument("--depth_intrinsics", required=True,
                        help="Depth intrinsics JSON {fx,fy,cx,cy,width,height}")
    parser.add_argument("--mano_model_dir",   required=True,
                        help="Dir containing MANO_RIGHT.pkl (or models/ subdir)")
    parser.add_argument("--output_hand_data", required=True,
                        help="Output path for world_results_aligned.npz")
    parser.add_argument("--object_masks_dir", default=None,
                        help="Per-frame object mask PNGs (SAM2, optional)")
    parser.add_argument("--object_poses_dir", default=None,
                        help="Per-frame FoundationPose Transform3d JSON files (optional)")
    parser.add_argument("--smooth_sigma",     type=float, default=5.0,
                        help="Gaussian sigma (frames) for temporal smoothing. 0=disable.")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_world_results(
        input_hand_data  = args.input_hand_data,
        depth_dir        = args.depth_dir,
        depth_intrinsics = args.depth_intrinsics,
        mano_model_dir   = args.mano_model_dir,
        output_hand_data = args.output_hand_data,
        object_masks_dir = args.object_masks_dir,
        object_poses_dir = args.object_poses_dir,
        smooth_sigma     = args.smooth_sigma,
        dev              = args.dev,
    )
