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
# Run download wrappers from the same interpreter environment as run_e2e.sh.
PYTHON_BIN="${PYTHON:-python3}"

MODE="all"
# FoundationPose model-license acceptance must always be explicit.
ACCEPT_NVIDIA_MODEL_EULA=false
usage() {
  echo "Usage: $0 [--mode all|dynhamr_prompt|hamer_prompt|hamer_mesh|hawor_prompt|hawor_mesh] [--accept-nvidia-model-eula]" >&2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    --accept-nvidia-model-eula)
      ACCEPT_NVIDIA_MODEL_EULA=true
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

case "$MODE" in
  all|dynhamr_prompt|hamer_prompt|hamer_mesh|hawor_prompt|hawor_mesh) ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

run_common() {
  local -a foundation_pose_args=(--output_dir data/weights/foundation_pose)
  "$PYTHON_BIN" -m v2d.moge.docker.run_download_weights --output_dir data/weights/moge
  "$PYTHON_BIN" -m v2d.sam2.docker.run_download_weights --output_dir data/weights/sam2
  if [[ "$ACCEPT_NVIDIA_MODEL_EULA" == true ]]; then
    foundation_pose_args+=(--accept_nvidia_model_eula)
  fi
  "$PYTHON_BIN" -m v2d.foundation_pose.docker.run_download_weights "${foundation_pose_args[@]}"
  "$PYTHON_BIN" -m v2d.anycalib.docker.run_download_weights --output_dir data/weights/anycalib
}

run_prompt_object() {
  "$PYTHON_BIN" -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
  "$PYTHON_BIN" -m v2d.sam3d.docker.run_download_weights --output_dir data/weights/sam3d
}

run_hamer() {
  "$PYTHON_BIN" -m v2d.wilor.docker.run_download_weights --weights_dir data/weights/wilor
  "$PYTHON_BIN" -m v2d.hamer.docker.run_download_weights --weights_dir data/weights/hamer
}

run_hawor() {
  "$PYTHON_BIN" -m v2d.hawor.docker.run_download_weights --weights_dir data/weights/hawor
}

run_optional_new() {
  "$PYTHON_BIN" -m v2d.droid_slam.docker.run_download_weights --output_dir data/weights/droid_slam
  "$PYTHON_BIN" -m v2d.geocalib.docker.run_download_weights --output_dir data/weights/geocalib
  "$PYTHON_BIN" -m v2d.gsplat_refinement.docker.run_download_weights --weights_path data/weights/gsplat_refinement
}

run_common
case "$MODE" in
  all)
    run_prompt_object
    run_hamer
    run_hawor
    run_optional_new
    ;;
  dynhamr_prompt)
    run_prompt_object
    ;;
  # Keep HaMeR modes independent so they do not require HAWOR images or weights.
  hamer_prompt)
    run_prompt_object
    run_hamer
    run_optional_new
    ;;
  hawor_prompt)
    run_prompt_object
    run_hamer
    run_hawor
    run_optional_new
    ;;
  hamer_mesh)
    # Object masks still come from --object_prompt. SAM3D is not needed when a mesh is supplied.
    "$PYTHON_BIN" -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
    run_hamer
    run_optional_new
    ;;
  hawor_mesh)
    # Object masks still come from --object_prompt. SAM3D is not needed when a mesh is supplied.
    "$PYTHON_BIN" -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
    run_hamer
    run_hawor
    run_optional_new
    ;;
esac

cat <<'EOF'

Weight download complete.

Manual DynHaMR/MANO setup is still required for --hand_tracking dynhamr:
  data/weights/hand/models/MANO_RIGHT.pkl
  data/weights/hand/BMC/*.npy
EOF
