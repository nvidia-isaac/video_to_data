#!/usr/bin/env bash
# Local equivalent of workflow/retarget.yaml — run the retargeting pipeline for
# one dataset on this machine, starting from the {dataset}_loaded parquet (the
# LOAD stage runs separately; see run_load_local.sh / docs/SETUP.md).
#
# Stages (mirrors retarget.yaml):
#   1.6  segment_to_atomic        (HOT3D only — long seqs -> atomic clips)
#   1.5  generate_rigid_urdfs     (object STL + rigid URDF, one-shot)
#   2    run_retarget (IK)        {dataset}_loaded -> {dataset}_processed
#   3    reconstruct_support_surfaces -> reconstructed_stage/
#   4    vis_retargeted           -> {dataset}_html/   (opt-in: WITH_VIS=1)
#   5    dummy_agent (Isaac play) -> {dataset}_videos/ (opt-in: WITH_DUMMY=1)
#   6    data_assessor            -> {dataset}_quality.csv + {dataset}_processed_invalid/
#
# Run inside the robotic-grounding Docker container (./workflow/run.sh start).
#
# Usage:
#   bash run_retarget_local.sh taco
#   WITH_VIS=1 bash run_retarget_local.sh hot3d
#   SEQUENCE_PATTERN='.*screw.*' MAX_SEQUENCES=5 WITH_VIS=1 WITH_DUMMY=1 bash run_retarget_local.sh taco
#
# Tunables (env): HMD (data root), ROBOT (sharpa_wave), WITH_VIS, WITH_DUMMY,
#   WITH_SUPPORT (default 1), SEQUENCE_PATTERN, MAX_SEQUENCES, JOBS, TIMEOUT, GPUS.
set -uo pipefail

DATASET=${1:?usage: bash run_retarget_local.sh <taco|hot3d|arctic|oakink2|h2o|grab|dexycb>}
ROBOT=${ROBOT:-sharpa_wave}
WITH_SUPPORT=${WITH_SUPPORT:-1}
WITH_VIS=${WITH_VIS:-0}        # viser .viser + MP4 (heavy for full datasets)
WITH_DUMMY=${WITH_DUMMY:-0}    # Isaac dummy_agent playback (very heavy)
JOBS=${JOBS:-4}                # dummy_agent concurrency (RAM-bound, ~10GB/proc)
TIMEOUT=${TIMEOUT:-600}        # per-clip dummy_agent timeout (s)

cd "$(dirname "$(readlink -f "$0")")"

# Data root: env HUMAN_MOTION_DATA_DIR, else the in-repo assets dir.
HMD=${HMD:-${HUMAN_MOTION_DATA_DIR:-source/robotic_grounding/robotic_grounding/assets/human_motion_data}}
DS_ROOT="${HMD}/${DATASET}"
LOADED="${DS_ROOT}/${DATASET}_loaded"
PROCESSED="${DS_ROOT}/${DATASET}_processed"
HTML_DIR="${DS_ROOT}/${DATASET}_html"
VIDEO_DIR="${DS_ROOT}/${DATASET}_videos"
LOG_DIR="${DS_ROOT}/dummy_agent_logs"

# Sequence filtering, passed through to every stage (all use add_sequence_filter_args).
FILTER=()
[ -n "${SEQUENCE_PATTERN:-}" ] && FILTER+=(--sequence_pattern "${SEQUENCE_PATTERN}")
[ -n "${MAX_SEQUENCES:-}" ] && FILTER+=(--max_sequences "${MAX_SEQUENCES}")

# GPUs to round-robin the dummy_agent over (auto-detect all).
if [ -z "${GPUS:-}" ]; then
    _n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1); [ "${_n:-0}" -lt 1 ] && _n=1
    GPUS=$(seq -s, 0 $((_n - 1)))
fi
IFS=',' read -ra GPU_ARR <<< "${GPUS}"; NGPU=${#GPU_ARR[@]}

echo "============================================================"
echo "retarget-local: ${DATASET}  (robot=${ROBOT})"
echo "  loaded : ${LOADED}"
echo "  vis=${WITH_VIS} dummy=${WITH_DUMMY} support=${WITH_SUPPORT}  filter='${SEQUENCE_PATTERN:-} ${MAX_SEQUENCES:-}'"
echo "============================================================"
[ -d "${LOADED}" ] || { echo "ERROR: ${LOADED} not found — run the LOAD stage first (run_load_local.sh ${DATASET})." >&2; exit 1; }

# --- Stage 1.6: HOT3D segmentation (atomic clips) --------------------------
RETARGET_INPUT="${LOADED}"
if [ "${DATASET}" = "hot3d" ]; then
    SEGMENTED="${DS_ROOT}/${DATASET}_loaded_segmented"
    echo "=== [1.6] Segment ${DATASET} -> ${SEGMENTED} ==="
    python scripts/segment_to_atomic.py --input_dir "${LOADED}" --output_dir "${SEGMENTED}" "${FILTER[@]}"
    RETARGET_INPUT="${SEGMENTED}"
fi

# --- Stage 1.5: object STL + rigid URDF (one-shot for the dataset) ----------
echo "=== [1.5] Generate rigid URDFs for ${DATASET} ==="
python scripts/generate_rigid_urdfs.py --dataset "${DATASET}" || true

# --- Stage 2: IK retargeting ------------------------------------------------
echo "=== [2] Retarget (IK) -> ${PROCESSED} ==="
python scripts/retarget/run_retarget.py \
    --dataset "${DATASET}" --robot "${ROBOT}" \
    --input_dir "${RETARGET_INPUT}" --output_dir "${PROCESSED}" \
    --device cuda:0 --save "${FILTER[@]}"

# --- Stage 3: support surfaces ----------------------------------------------
if [ "${WITH_SUPPORT}" = "1" ]; then
    echo "=== [3] Reconstruct support surfaces ==="
    python scripts/reconstruct_support_surfaces.py \
        --input_dir "${RETARGET_INPUT}" --dataset "${DATASET}" "${FILTER[@]}" || true
fi

# --- Stage 4: viser HTML + MP4 (opt-in) -------------------------------------
if [ "${WITH_VIS}" = "1" ]; then
    echo "=== [4] Render viser HTML + MP4 -> ${HTML_DIR} ==="
    MP4_ARG=(--save_mp4)
    [ "${DATASET}" = "oakink2" ] && MP4_ARG=()  # long seqs OOM pyrender
    python scripts/retarget/vis_retargeted.py \
        --input_dir "${PROCESSED}" --save_html "${MP4_ARG[@]}" \
        --html_dir "${HTML_DIR}" "${FILTER[@]}" || true
fi

# --- Stage 5: Isaac dummy_agent playback (opt-in, sharded) ------------------
if [ "${WITH_DUMMY}" = "1" ]; then
    echo "=== [5] dummy_agent playback (JOBS=${JOBS}, GPUs=${GPUS}) ==="
    mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"
    mapfile -t SEQ_DIRS < <(ls -d "${PROCESSED}"/sequence_id=*/robot_name=* 2>/dev/null)
    running=0; idx=0
    for seq_dir in "${SEQ_DIRS[@]}"; do
        gpu="${GPU_ARR[$((idx % NGPU))]}"; idx=$((idx + 1))
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            sid=$(basename "$(dirname "${seq_dir}")"); sid=${sid#sequence_id=}
            [ -f "${seq_dir}/.dummy_agent_ok" ] && exit 0   # resume
            tmp=/tmp/da_${sid}; rm -rf "${tmp}"
            # Detach from TTY (</dev/null) + render to /tmp, else Kit shutdown
            # hangs under `timeout`; move the finished MP4 onto the dataset.
            timeout --signal=KILL "${TIMEOUT}" python scripts/rsl_rl/dummy_agent.py \
                --task Sharpa-V2P-v0-Play --motion_file "${seq_dir}" \
                --num_envs 1 --headless \
                --record_video --output_dir "${tmp}" --video_length 300 \
                --success_marker "${seq_dir}/.dummy_agent_ok" \
                </dev/null >"${LOG_DIR}/${sid}.log" 2>&1 || true
            [ -f "${tmp}/rl-video-step-0.mp4" ] && mv "${tmp}/rl-video-step-0.mp4" "${VIDEO_DIR}/${sid}.mp4"
            rm -rf "${tmp}"
        ) &
        running=$((running + 1))
        if [ "${running}" -ge "${JOBS}" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
    done
    wait
fi

# --- Stage 6: data-quality filter -------------------------------------------
echo "=== [6] Data quality filter ==="
CHECKS_ARG=()
[ "${WITH_DUMMY}" != "1" ] && CHECKS_ARG=(--disable dummy_agent_success)
python scripts/data_assessor.py \
    --input_dir "${PROCESSED}" --reject \
    --output_reject "${DS_ROOT}/${DATASET}_rejected.txt" \
    --output_csv "${DS_ROOT}/${DATASET}_quality.csv" \
    "${CHECKS_ARG[@]}" "${FILTER[@]}" || true

echo ""
echo "Done ${DATASET}. processed=${PROCESSED}  quality=${DS_ROOT}/${DATASET}_quality.csv"
[ "${WITH_VIS}" = "1" ] && echo "  serve: python visualizer/serve.py --port 8081 --html-dir ${HTML_DIR}"
