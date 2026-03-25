import os
from v2d.docker.container import run_in_container
from v2d.sam3d.docker._config import IMAGE_NAME, MODULES_DIR

def run_image_to_mesh(
    image_path: str,
    mask_path: str,
    mesh_path: str,
    transform_path: str,
    intrinsics_path: str,
    weights_dir: str,
    seed: int = None,
    stage1_only: bool = False,
    with_mesh_postprocess: bool = False,
    with_texture_baking: bool = False,
    with_layout_postprocess: bool = False,
    use_vertex_color: bool = True,
    stage1_inference_steps: int = None,
    dev: bool = False,
) -> None:
    weights_abs = os.path.abspath(weights_dir)
    weights_container = f"/data/weights_dir/{os.path.basename(weights_abs)}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam3d.lib.image_to_mesh",
        inputs={"image_path": image_path, "mask_path": mask_path, "weights_dir": weights_dir},
        outputs={"mesh_path": mesh_path, "transform_path": transform_path, "intrinsics_path": intrinsics_path},
        extra_args={
            "seed": seed,
            "stage1_only": stage1_only,
            "with_mesh_postprocess": with_mesh_postprocess,
            "with_texture_baking": with_texture_baking,
            "with_layout_postprocess": with_layout_postprocess,
            "use_vertex_color": use_vertex_color,
            "stage1_inference_steps": stage1_inference_steps,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"TORCH_HOME": f"{weights_container}/torch_home", "HF_HOME": f"{weights_container}/hf_home"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process image to mesh using SAM3D")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to input mask")
    parser.add_argument("--mesh_path", type=str, required=True, help="Output path for mesh (.glb or .obj)")
    parser.add_argument("--transform_path", type=str, required=True, help="Output path for transform JSON")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for intrinsics JSON")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to weights directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--stage1_only", action="store_true", help="Only run stage 1")
    parser.add_argument("--with_mesh_postprocess", action="store_true", help="Enable mesh postprocessing")
    parser.add_argument("--with_texture_baking", action="store_true", help="Enable texture baking")
    parser.add_argument("--with_layout_postprocess", action="store_true", help="Enable layout postprocessing")
    parser.add_argument("--use_vertex_color", action="store_true", default=True, help="Use vertex color")
    parser.add_argument("--stage1_inference_steps", type=int, default=None, help="Stage 1 inference steps")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_image_to_mesh(
        args.image_path, args.mask_path, args.mesh_path,
        args.transform_path, args.intrinsics_path, args.weights_dir,
        seed=args.seed, stage1_only=args.stage1_only,
        with_mesh_postprocess=args.with_mesh_postprocess,
        with_texture_baking=args.with_texture_baking,
        with_layout_postprocess=args.with_layout_postprocess,
        use_vertex_color=args.use_vertex_color,
        stage1_inference_steps=args.stage1_inference_steps,
        dev=args.dev,
    )
