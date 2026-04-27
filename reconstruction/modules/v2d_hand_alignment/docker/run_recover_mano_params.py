from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_recover_mano_params(
    aligned_path: str,
    world_results_path: str,
    hand_mesh_traj_path: str,
    output_path: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.recover_mano_params",
        inputs={
            "aligned_path":        aligned_path,
            "world_results_path":  world_results_path,
            "hand_mesh_traj_path": hand_mesh_traj_path,
        },
        outputs={"output_path": output_path},
        extra_args={},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_path",        required=True)
    parser.add_argument("--world_results_path",  required=True)
    parser.add_argument("--hand_mesh_traj_path", required=True)
    parser.add_argument("--output_path",         required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_recover_mano_params(
        args.aligned_path,
        args.world_results_path,
        args.hand_mesh_traj_path,
        args.output_path,
        dev=args.dev,
    )
