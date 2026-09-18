#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch N sharded taco data-gen containers on ONE GPU in parallel, then merge.
#
# Each shard is a SEPARATE container (own Kit cache -> no kvdb lock; disjoint output seq
# dirs -> no recorder file-lock) processing seqs[i::N]. This overlaps the per-task Isaac
# startup (the batch bottleneck) across shards for a ~N x wall-clock speedup.
#
# Host-RAM bound (~6.4 GB + ~21 GB base per instance; swap disabled): 4 is safe, 5 tight,
# 6+ risks OOM. VRAM (~6 GB/instance on the 49 GB A6000) is NOT the limit.
#
# Usage (from robotic_grounding/):
#   scripts/launch_sharded_datagen.sh [N_SHARDS=4]
#   OUT_ROOT=datasets/Taco_Datagen_Sharded scripts/launch_sharded_datagen.sh 4
#
# Batch config (NUM_ENVS/NUM_EPISODES/VOC_*/DISABLE_TERMINATIONS) lives in
# scripts/batch_taco_datagen.py. Progress: tail datasets/batch_shard*.log. Final report:
# $OUT_ROOT/SUMMARY.md (written by the merge step after all shards exit).
set -euo pipefail

SHARDS="${1:-4}"
IMAGE="robotic-grounding:latest"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-datasets/Taco_Datagen_Sharded}"
REPO_PARENT="$(cd "$(dirname "$0")/../.." && pwd)"   # repository checkout
CACHE_ROOT="${HOME}/.cache/robotic-grounding"

CRED="${HOME}/.config/osmo/css_credential.yaml"
q() { grep -E "^\s*$1:" "$CRED" | head -1 | sed -E 's/^[^"]*"([^"]*)".*/\1/'; }
CSS_ACCESS_KEY="$(q access_key_id)"; CSS_SECRET_KEY="$(q access_key)"
CSS_ENDPOINT_URL="$(q endpoint)"; CSS_REGION="$(q region)"

HOST_UID="$(id -u)"; HOST_GID="$(id -g)"; HOST_USER="$(id -un)"
[[ "$HOST_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] || HOST_USER="user"

setup_container_dir() {  # $1 = container name -> writes passwd/group/kit dirs, echoes dir
  local cdir="$CACHE_ROOT/$1"
  mkdir -p "$cdir/kit-data" "$cdir/kit-cache" "$cdir/kit-logs"
  cat > "$cdir/passwd" <<EOF
root:x:0:0:root:/root:/bin/bash
isaac-sim:x:1234:1234::/isaac-sim:/bin/bash
${HOST_USER}:x:${HOST_UID}:${HOST_GID}:${HOST_USER}:/tmp:/bin/bash
EOF
  cat > "$cdir/group" <<EOF
root:x:0:
isaac-sim:x:1234:${HOST_USER}
${HOST_USER}:x:${HOST_GID}:
EOF
  chmod 0644 "$cdir/passwd" "$cdir/group"
  echo "$cdir"
}

run_container() {  # $1=name $2=extra "-e" env pairs (space sep) $3=inner cmd
  local name="$1" envs="$2" cmd="$3" cdir
  docker rm -f "$name" >/dev/null 2>&1 || true
  cdir="$(setup_container_dir "$name")"
  # shellcheck disable=SC2086
  docker run -d --rm --name "$name" \
    --runtime=nvidia --gpus "device=${GPU}" --network host \
    --user "${HOST_UID}:${HOST_GID}" --group-add 1234 \
    -v "${REPO_PARENT}:/workspace/video_to_data" \
    -v "${cdir}/kit-data:/isaac-sim/kit/data" \
    -v "${cdir}/kit-cache:/isaac-sim/kit/cache" \
    -v "${cdir}/kit-logs:/isaac-sim/kit/logs" \
    -v "${cdir}/passwd:/etc/passwd:ro" -v "${cdir}/group:/etc/group:ro" \
    -e HOME=/tmp -e USER="${HOST_USER}" -e ACCEPT_EULA=Y -e PYTHONUNBUFFERED=1 \
    -e CSS_ACCESS_KEY="${CSS_ACCESS_KEY}" -e CSS_SECRET_KEY="${CSS_SECRET_KEY}" \
    -e CSS_ENDPOINT_URL="${CSS_ENDPOINT_URL}" -e CSS_REGION="${CSS_REGION}" \
    -e OUT_ROOT="${OUT_ROOT}" ${envs} \
    --entrypoint /bin/bash "${IMAGE}" \
    -lc "cd /workspace/video_to_data/robotic_grounding && ${cmd}" >/dev/null
}

echo "[shard] launching ${SHARDS} shards on GPU ${GPU} -> ${OUT_ROOT}"
names=()
for i in $(seq 0 $((SHARDS - 1))); do
  n="rg-shard-${i}"
  run_container "$n" "-e SHARD_INDEX=${i} -e SHARD_COUNT=${SHARDS}" \
    "stdbuf -oL -eL python scripts/batch_taco_datagen.py > datasets/batch_shard${i}.log 2>&1"
  names+=("$n")
  echo "  launched $n (shard ${i}/${SHARDS})"
done

echo "[shard] waiting for all shards to finish (monitor: tail -f datasets/batch_shard*.log)"
docker wait "${names[@]}" >/dev/null || true

echo "[shard] all shards done; merging summaries via SCORE_ONLY"
run_container "rg-merge" "-e SCORE_ONLY=1" "python scripts/batch_taco_datagen.py"
docker wait rg-merge >/dev/null || true
echo "[shard] DONE. Combined report -> ${OUT_ROOT}/SUMMARY.md ; per-shard logs datasets/batch_shard*.log"
