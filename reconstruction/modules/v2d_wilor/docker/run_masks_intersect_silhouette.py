from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_masks_intersect_silhouette(
    wilor_dir: str,
    masks_dir: str,
    tracks_path: str,
    output_dir: str,
    mano_assets_root: str,
    dilation_pixels: int = 20,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.masks_intersect_silhouette",
        inputs={
            "wilor_dir":        wilor_dir,
            "masks_dir":        masks_dir,
            "tracks_path":      tracks_path,
            "mano_assets_root": mano_assets_root,
        },
        outputs={"output_dir": output_dir},
        extra_args={"dilation_pixels": dilation_pixels},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Refine SAM2 hand masks by intersecting with dilated MANO silhouettes."
    )
    parser.add_argument("--wilor_dir",        required=True)
    parser.add_argument("--masks_dir",        required=True)
    parser.add_argument("--tracks_path",      required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--dilation_pixels",  type=int, default=20)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_masks_intersect_silhouette(
        wilor_dir        = args.wilor_dir,
        masks_dir        = args.masks_dir,
        tracks_path      = args.tracks_path,
        output_dir       = args.output_dir,
        mano_assets_root = args.mano_assets_root,
        dilation_pixels  = args.dilation_pixels,
        dev              = args.dev,
    )
