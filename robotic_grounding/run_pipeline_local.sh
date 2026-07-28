#!/usr/bin/env bash
# One-command per-dataset orchestrator (run INSIDE the robotic-grounding container):
#   delegate to run_retarget_local.sh (URDF gen -> retarget -> support -> vis/dummy -> assess).
#
# Object meshes are NOT fetched here. Download them from each dataset's public source and
# place them per docs/SETUP.md (§4/§8):
#   - object_assets datasets (arctic / taco / hot3d / oakink2): object_assets/meshes/<ds>/
#   - grab / h2o / dexycb: meshes ship inside the raw dataset (nothing to place)
# The URDF generation that consumes the meshes runs inside run_retarget_local.sh.
#
# The upstream LOAD stage (raw -> {dataset}_loaded) is NOT here — it uses a separate image
# and runs on the HOST. If {dataset}_loaded is missing, run_retarget_local.sh tells you to
# run run_load_local.sh first. See docs/SETUP.md.
#
# Usage:
#   bash run_pipeline_local.sh taco
#   WITH_VIS=1 bash run_pipeline_local.sh hot3d
#   (all run_retarget_local.sh env vars pass through: WITH_VIS, WITH_DUMMY, JOBS, SEQUENCE_PATTERN, MAX_SEQUENCES, HMD, ...)
set -uo pipefail

DATASET=${1:?usage: bash run_pipeline_local.sh <taco|hot3d|arctic|oakink2|h2o|grab|dexycb>}
cd "$(dirname "$(readlink -f "$0")")"

case "${DATASET}" in
    arctic|taco|oakink2|hot3d)
        echo "=== [assets] ${DATASET}: object meshes must already be present under object_assets/meshes/${DATASET}/ (download per docs/SETUP.md §6/§8) ==="
        ;;
    *)
        echo "=== [assets] ${DATASET}: object meshes ship inside the raw dataset — nothing to place ==="
        ;;
esac

echo "=== [pipeline] delegating to run_retarget_local.sh ${DATASET} ==="
exec bash run_retarget_local.sh "${DATASET}"
