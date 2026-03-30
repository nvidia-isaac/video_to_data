import argparse
import urllib.request
from pathlib import Path

VITDET_VARIANTS = {
    "b": {
        "url": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_b/f325358525/model_final_435fa9.pkl",
        "file": "model_final_435fa9.pkl",
    },
    "l": {
        "url": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_l/f328021305/model_final_1a9f28.pkl",
        "file": "model_final_1a9f28.pkl",
    },
    "h": {
        "url": "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
        "file": "model_final_f05665.pkl",
    },
}


def download_weights(output_dir: str, model_sizes: list[str] | None = None):
    if model_sizes is None:
        model_sizes = ["b"]

    for size in model_sizes:
        if size not in VITDET_VARIANTS:
            raise ValueError(f"Unknown model size '{size}', expected one of {list(VITDET_VARIANTS)}")

        variant = VITDET_VARIANTS[size]
        dest_dir = Path(output_dir) / f"cascade_mask_rcnn_vitdet_{size}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / variant["file"]

        if dest_file.exists():
            print(f"ViTDet-{size.upper()} checkpoint already exists at {dest_file}")
            continue

        print(f"Downloading ViTDet-{size.upper()} checkpoint to {dest_file}...")
        urllib.request.urlretrieve(variant["url"], dest_file)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ViTDet checkpoints")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument(
        "--model_sizes", type=str, nargs="+", default=["b"], choices=["b", "l", "h"],
        help="Which model sizes to download (default: b)",
    )
    args = parser.parse_args()
    download_weights(args.output_dir, args.model_sizes)
