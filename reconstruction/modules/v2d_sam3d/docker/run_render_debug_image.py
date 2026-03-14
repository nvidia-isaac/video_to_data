import subprocess
import os

IMAGE_NAME = "v2d_sam3d"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))

def run_render_debug_image(
    image_path: str,
    mesh_path: str,
    transform_path: str,
    intrinsics_path: str,
    output_image_path: str,
    num_vertices_to_use: int = 5000,
    dev: bool = False,
) -> None:
    image_path = os.path.abspath(image_path)
    mesh_path = os.path.abspath(mesh_path)
    transform_path = os.path.abspath(transform_path)
    intrinsics_path = os.path.abspath(intrinsics_path)
    output_image_path = os.path.abspath(output_image_path)

    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    mesh_dir = os.path.dirname(mesh_path)
    mesh_name = os.path.basename(mesh_path)
    transform_dir = os.path.dirname(transform_path)
    transform_name = os.path.basename(transform_path)
    intrinsics_dir = os.path.dirname(intrinsics_path)
    intrinsics_name = os.path.basename(intrinsics_path)
    output_dir = os.path.dirname(output_image_path)
    output_name = os.path.basename(output_image_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{image_dir}:/data/image",
        "-v", f"{mesh_dir}:/data/mesh",
        "-v", f"{transform_dir}:/data/transform",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{output_dir}:/data/output",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.sam3d.lib.render_debug_image",
        "--image_path", f"/data/image/{image_name}",
        "--mesh_path", f"/data/mesh/{mesh_name}",
        "--transform_path", f"/data/transform/{transform_name}",
        "--intrinsics_path", f"/data/intrinsics/{intrinsics_name}",
        "--output_image_path", f"/data/output/{output_name}",
        "--num_vertices_to_use", str(num_vertices_to_use),
    ]
    subprocess.run(cmd, check=True)


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
