#!/usr/bin/env bash
# Local LOAD stage: raw dataset -> {dataset}_loaded parquet (MANO FK), via the
# reconstruction `v2d_task_library_loader` image. This is the upstream of the
# retarget pipeline (run_retarget_local.sh consumes the {dataset}_loaded output).
#
# IMPORTANT: this runs on the HOST (the loader has its OWN docker image, separate
# from robotic-grounding). It does NOT run inside the robotic-grounding container.
# The MANO .pkl models (GPL-adjacent, license-gated) are mounted at runtime.
#
# Usage:
#   MANO_DIR=/data/mano BUILD=1 bash run_load_local.sh taco     # build image first time
#   MANO_DIR=/data/mano bash run_load_local.sh arctic
#
# Required env:
#   MANO_DIR   dir with manotorch layout: MANO_DIR/models/MANO_{LEFT,RIGHT}.pkl + MANO_DIR/BMC/*.npy
#              (download from mano.is.tue.mpg.de — see docs/SETUP.md)
# Optional env:
#   RAW_DIR    raw dataset root            (default: <HMD>/<dataset>/dataset)
#   OBJ_DIR    object_assets root w/ urdfs/<ds> + meshes/<ds>
#              (default: the in-repo assets dir; populate per the dataset's doc in docs/)
#   OUT        output dir                  (default: <HMD>/<dataset>/<dataset>_loaded)
#   HMD        data root                   (default: in-repo assets/human_motion_data)
#   DEVICE     (default cuda:0)   BUILD=1 to (re)build the loader image first
#   SEQUENCE_PATTERN / MAX_SEQUENCES  passthrough sampling
set -uo pipefail

DATASET=${1:?usage: MANO_DIR=<dir> bash run_load_local.sh <taco|hot3d|arctic|oakink2|h2o|grab|dexycb>}
: "${MANO_DIR:?set MANO_DIR=<dir with models/MANO_*.pkl + BMC/> (see docs/SETUP.md)}"
DEVICE=${DEVICE:-cuda:0}

# Monorepo root = parent of robotic_grounding; the loader lives in reconstruction/.
RG_DIR="$(dirname "$(readlink -f "$0")")"
MONO="$(dirname "${RG_DIR}")"
RECON="${MONO}/reconstruction"
[ -d "${RECON}/modules/v2d_task_library_loader" ] || {
    echo "ERROR: reconstruction loader not found at ${RECON}/modules/v2d_task_library_loader" >&2; exit 1; }

HMD=${HMD:-${RG_DIR}/source/robotic_grounding/robotic_grounding/assets/human_motion_data}
RAW_DIR=${RAW_DIR:-${HMD}/${DATASET}/dataset}
OBJ_DIR=${OBJ_DIR:-${RG_DIR}/source/robotic_grounding/robotic_grounding/assets}
OUT=${OUT:-${HMD}/${DATASET}/${DATASET}_loaded}

echo "============================================================"
echo "load-local: ${DATASET}"
echo "  raw     : ${RAW_DIR}"
echo "  mano    : ${MANO_DIR}"
echo "  objects : ${OBJ_DIR}  (expects urdfs/${DATASET} + meshes/${DATASET})"
echo "  output  : ${OUT}"
echo "============================================================"
[ -d "${RAW_DIR}" ] || { echo "ERROR: raw dataset not at ${RAW_DIR} — download it first (see docs)." >&2; exit 1; }
[ -f "${MANO_DIR}/models/MANO_RIGHT.pkl" ] || echo "WARN: ${MANO_DIR}/models/MANO_RIGHT.pkl not found — check MANO layout."

cd "${RECON}"

# Build the loader image once (or when BUILD=1).
if [ "${BUILD:-0}" = "1" ] || ! docker image inspect v2d_task_library_loader >/dev/null 2>&1; then
    echo "=== Building v2d_task_library_loader image ==="
    # v2d-docker is a LOCAL monorepo package (not on PyPI), and the loader's
    # docker package depends on it by name. Install both (+ v2d_common) editable
    # in ONE pip invocation so the local dep resolves — installing the loader
    # docker pkg alone makes pip try (and fail) to fetch v2d-docker from an index.
    pip install -e modules/v2d_common \
                -e modules/v2d_docker \
                -e modules/v2d_task_library_loader/docker
    python -m v2d.task_library_loader.docker.build
fi

FILTER=()
[ -n "${SEQUENCE_PATTERN:-}" ] && FILTER+=(--sequence_pattern "${SEQUENCE_PATTERN}")
[ -n "${MAX_SEQUENCES:-}" ] && FILTER+=(--max_sequences "${MAX_SEQUENCES}")

echo "=== Running loader (raw -> ${DATASET}_loaded) ==="
# NOTE: run_loader.py mounts --human_motion_data_dir as the container's
# $HUMAN_MOTION_DATA_DIR, and each loader appends "<dataset>/dataset" itself
# (e.g. taco reads $HUMAN_MOTION_DATA_DIR/taco/dataset). So we pass the HMD
# ROOT here, not RAW_DIR (=<HMD>/<dataset>/dataset) — otherwise the loader
# would look in <HMD>/<dataset>/dataset/<dataset>/dataset.
python modules/v2d_task_library_loader/docker/run_loader.py \
    --dataset "${DATASET}" \
    --mano_model_dir "${MANO_DIR}" \
    --human_motion_data_dir "${HMD}" \
    --object_assets_dir "${OBJ_DIR}" \
    --output_dir "${OUT}" \
    --device "${DEVICE}" --save "${FILTER[@]}"

echo ""
echo "Done. Loaded parquet -> ${OUT}"
echo "Next: bash run_retarget_local.sh ${DATASET}"
