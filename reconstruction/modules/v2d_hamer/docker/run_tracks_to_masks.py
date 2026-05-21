from v2d.docker.container import run_in_container
from v2d.hamer.docker._config import IMAGE_NAME, MODULES_DIR


def run_tracks_to_masks(
    tracks_dir: str,
    intrinsics_path: str,
    mano_assets_root: str,
    output_dir: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hamer.lib.tracks_to_masks",
        inputs={
            "tracks_dir":       tracks_dir,
            "intrinsics_path":  intrinsics_path,
            "mano_assets_root": mano_assets_root,
        },
        outputs={"output_dir": output_dir},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render MANO silhouette masks from a HaMeR-style tracks dir"
    )
    parser.add_argument("--tracks_dir",       required=True)
    parser.add_argument("--intrinsics_path",  required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_tracks_to_masks(
        tracks_dir       = args.tracks_dir,
        intrinsics_path  = args.intrinsics_path,
        mano_assets_root = args.mano_assets_root,
        output_dir       = args.output_dir,
        dev              = args.dev,
    )
