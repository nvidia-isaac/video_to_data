#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Download model weights required by run_ego_reconstruction.py.
#
# Run from reconstruction/ or repo root after building the matching containers.
# SAM3D requires HF_TOKEN or a prior `huggingface-cli login` for gated access.
# DynHaMR/MANO assets still require manual setup under data/weights/hand.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODE="all"
if [[ $# -gt 0 ]]; then
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    *)
      echo "Usage: $0 [--mode all|dynhamr_prompt|hamer_prompt|hamer_mesh]" >&2
      exit 2
      ;;
  esac
fi
if [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--mode all|dynhamr_prompt|hamer_prompt|hamer_mesh]" >&2
  exit 2
fi

case "$MODE" in
  all|dynhamr_prompt|hamer_prompt|hamer_mesh) ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [--mode all|dynhamr_prompt|hamer_prompt|hamer_mesh]" >&2
    exit 2
    ;;
esac

run_common() {
  python -m v2d.moge.docker.run_download_weights --output_dir data/weights/moge
  python -m v2d.sam2.docker.run_download_weights --output_dir data/weights/sam2
  python -m v2d.foundation_pose.docker.run_download_weights --output_dir data/weights/foundation_pose
  python -m v2d.anycalib.docker.run_download_weights --output_dir data/weights/anycalib
}

run_prompt_object() {
  python -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
  python -m v2d.sam3d.docker.run_download_weights --output_dir data/weights/sam3d
}

run_hamer() {
  python -m v2d.wilor.docker.run_download_weights --weights_dir data/weights/wilor
  python -m v2d.hamer.docker.run_download_weights --weights_dir data/weights/hamer
}

run_optional_new() {
  python -m v2d.droid_slam.docker.run_download_weights --output_dir data/weights/droid_slam
  python -m v2d.geocalib.docker.run_download_weights --output_dir data/weights/geocalib
  python -m v2d.gsplat_refinement.docker.run_download_weights --weights_path data/weights/gsplat_refinement
}

run_common
case "$MODE" in
  all)
    run_prompt_object
    run_hamer
    run_optional_new
    ;;
  dynhamr_prompt)
    run_prompt_object
    ;;
  hamer_prompt)
    run_prompt_object
    run_hamer
    run_optional_new
    ;;
  hamer_mesh)
    # Object masks still come from --object_prompt. SAM3D is not needed when a mesh is supplied.
    python -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
    run_hamer
    run_optional_new
    ;;
esac

cat <<'EOF'

Weight download complete.

Manual DynHaMR/MANO setup is still required for --hand_tracking dynhamr:
  data/weights/hand/models/MANO_RIGHT.pkl
  data/weights/hand/BMC/*.npy
EOF
