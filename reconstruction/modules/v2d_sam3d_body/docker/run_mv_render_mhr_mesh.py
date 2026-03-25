from v2d.docker.container import run_in_container
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR


def run_mv_render_mhr_mesh(
    data_path: str,
    config_path: str | None = None,
    dev: bool = False,
) -> None:
    inputs = {"data_path": data_path}
    if config_path is not None:
        inputs["config_path"] = config_path

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam3d_body.lib.mv_render_mhr_mesh",
        inputs=inputs,
        outputs={},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render MHR mesh overlay for all cameras")
    parser.add_argument("--data_path", type=str, required=True, help="Root data directory")
    parser.add_argument("--config_path", type=str, default=None, help="Path to mv_config.yaml")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_render_mhr_mesh(
        data_path=args.data_path,
        config_path=args.config_path,
        dev=args.dev,
    )
