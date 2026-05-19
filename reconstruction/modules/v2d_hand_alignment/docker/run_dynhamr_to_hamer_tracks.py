from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_dynhamr_to_hamer_tracks(
    input_npz: str,
    output_dir: str,
    intrinsics_path: str | None = None,
    left_id: int = 2,
    right_id: int = 3,
    dev: bool = False,
) -> None:
    """Convert DynHaMR world_results.npz → per-frame v2d_hamer-style tracks.

    Output structure:
        output_dir/<left_id>/<frame:06d>.json   (left hand records)
        output_dir/<right_id>/<frame:06d>.json  (right hand records)

    The per-frame JSON places each hand in DynHaMR's per-frame camera
    coordinates so v2d_hamer.lib.align_hands can consume the tracks just
    like real HaMeR detections — bypassing ViPE's world frame entirely.
    """
    inputs: dict[str, str] = {
        "input_npz": input_npz,
    }
    if intrinsics_path is not None:
        inputs["intrinsics_path"] = intrinsics_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.dynhamr_to_hamer_tracks",
        inputs=inputs,
        outputs={"output_dir": output_dir},
        extra_args={
            "left_id":  left_id,
            "right_id": right_id,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Convert DynHaMR world_results.npz → v2d_hamer-style tracks"
    )
    p.add_argument("--input_npz",       required=True)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--intrinsics_path", default=None)
    p.add_argument("--left_id",  type=int, default=2)
    p.add_argument("--right_id", type=int, default=3)
    p.add_argument("--dev",      action="store_true")
    args = p.parse_args()
    run_dynhamr_to_hamer_tracks(
        input_npz       = args.input_npz,
        output_dir      = args.output_dir,
        intrinsics_path = args.intrinsics_path,
        left_id         = args.left_id,
        right_id        = args.right_id,
        dev             = args.dev,
    )
