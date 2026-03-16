from v2d.docker.container import run_in_container
from v2d.sam3d.docker._config import IMAGE_NAME, MODULES_DIR

def run_render_debug_image(
    image_path: str,
    mesh_path: str,
    transform_path: str,
    intrinsics_path: str,
    output_image_path: str,
    num_vertices_to_use: int = 5000,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam3d.lib.render_debug_image",
        inputs={"image_path": image_path, "mesh_path": mesh_path, "transform_path": transform_path, "intrinsics_path": intrinsics_path},
        outputs={"output_image_path": output_image_path},
        extra_args={"num_vertices_to_use": num_vertices_to_use},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render debug image of a mesh")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the mesh")
    parser.add_argument("--transform_path", type=str, required=True, help="Path to the transform")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Path to the intrinsics")
    parser.add_argument("--output_image_path", type=str, required=True, help="Path to the output image")
    parser.add_argument("--num_vertices_to_use", type=int, default=5000, help="Number of vertices to use")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_render_debug_image(
        args.image_path, args.mesh_path, args.transform_path,
        args.intrinsics_path, args.output_image_path,
        num_vertices_to_use=args.num_vertices_to_use, dev=args.dev,
    )
