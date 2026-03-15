from v2d.docker.container import run_in_container
from v2d.mesh.docker._config import IMAGE_NAME, MODULES_DIR


def run_mesh_render_mask(
    mesh_path: str,
    intrinsics_path: str,
    output_mask_path: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mesh.lib.run_mesh_render_mask",
        inputs={"mesh": mesh_path, "intrinsics": intrinsics_path},
        outputs={"output_mask": output_mask_path},
        dev=dev,
        modules_dir=MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a silhouette mask of a mesh (via Docker)")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output_mask", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_render_mask(args.mesh, args.intrinsics, args.output_mask, dev=args.dev)
