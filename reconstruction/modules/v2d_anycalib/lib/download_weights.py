"""Download AnyCalib model weights from Hugging Face."""
import argparse
import os
import subprocess

# Hugging Face repo for the general distortion-aware AnyCalib checkpoint.
# Variants: anycalib_pinhole, anycalib_radial, anycalib_gen, anycalib_dist.
DEFAULT_MODEL_ID = "javrtg/anycalib_gen"


def download_anycalib(output_dir: str | None = None, model_id: str = DEFAULT_MODEL_ID) -> None:
    if output_dir is None:
        output_dir = os.environ.get("CHECKPOINT_DIR")
        if output_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["hf", "download", model_id, "--local-dir", output_dir], check=True)
    print(f"AnyCalib checkpoint ({model_id}) downloaded to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AnyCalib checkpoint")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="HF repo id (default: javrtg/anycalib_gen)")
    args = parser.parse_args()
    download_anycalib(output_dir=args.output_dir, model_id=args.model_id)
