"""Pre-fetch WiLoR weights into ``<weights_dir>/pretrained_models/``.

The pipeline auto-downloads its four files (mano_mean_params.npz,
MANO_RIGHT.pkl, wilor_final.ckpt, detector.pt) on first instantiation; the
``wilor_pretrained_dir`` kwarg redirects them out of the package dir into a
user-writable location. Running this once primes the cache so subsequent
runs are offline.
"""

import argparse
import os


def run_download(weights_dir: str) -> None:
    os.makedirs(weights_dir, exist_ok=True)
    print(f"  Downloading WiLoR weights → {weights_dir}/pretrained_models/")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = WiLorHandPose3dEstimationPipeline(
        device=device, dtype=torch.float16, wilor_pretrained_dir=weights_dir,
    )
    print(f"  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights_dir", required=True)
    args = parser.parse_args()
    run_download(args.weights_dir)
