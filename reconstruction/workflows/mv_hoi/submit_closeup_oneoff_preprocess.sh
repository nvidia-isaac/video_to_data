#!/usr/bin/env bash
set -euo pipefail

# Submit one-off closeup MV preprocess jobs with calibration extrinsics and no
# mesh pinning. These runs are outside the MV HOI processing DB bookkeeping.
#
# Usage:
#   ./submit_closeup_oneoff_preprocess.sh
#   DRY_RUN=1 ./submit_closeup_oneoff_preprocess.sh
#
# Optional overrides:
#   CALIB_HTTP_URL=... POOL=... WORKFLOW_PATH=... ./submit_closeup_oneoff_preprocess.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"

POOL="${POOL:-isaac-dev-h100-01}"
WORKFLOW_PATH="${WORKFLOW_PATH:-$REPO_ROOT/reconstruction/workflows/mv_hoi/osmo/mv_preprocess_oneoff.yaml}"
DRY_RUN="${DRY_RUN:-0}"
CALIB_HTTP_URL="${CALIB_HTTP_URL:-https://pdx.s8k.io/v1/AUTH_team-isaac/recordings/v2d/multiview_closeup_test/sc_office_4exo_1/calibration/2026-07-07_14-57-46_calibration_closeup/}"

SEQUENCE_HTTP_URLS=(
  "https://pdx.s8k.io/v1/AUTH_team-isaac/recordings/v2d/multiview_closeup_test/sc_office_4exo_1/data/2026-07-07_15-07-45_rice_cooker_open_close_02/"
  "https://pdx.s8k.io/v1/AUTH_team-isaac/recordings/v2d/multiview_closeup_test/sc_office_4exo_1/data/2026-07-07_15-09-40_scissors_walk_around/"
  "https://pdx.s8k.io/v1/AUTH_team-isaac/recordings/v2d/multiview_closeup_test/sc_office_4exo_1/data/2026-07-07_15-11-30_egg_carton_open_close/"
  "https://pdx.s8k.io/v1/AUTH_team-isaac/recordings/v2d/multiview_closeup_test/sc_office_4exo_1/data/2026-07-07_15-13-10_toy_pressure_cooker_open_close/"
)

to_swift_url() {
  local url="${1%/}/"
  if [[ "$url" == swift://* ]]; then
    printf '%s\n' "$url"
  elif [[ "$url" == https://pdx.s8k.io/v1/* ]]; then
    printf '%s\n' "${url/#https:\/\/pdx.s8k.io\/v1\//swift:\/\/pdx.s8k.io\/}"
  else
    echo "ERROR: unsupported URL format: $1" >&2
    return 1
  fi
}

short_workflow_name() {
  local seq_name="$1"
  local seq_time="${seq_name:11:8}"
  local seq_slug="${seq_name:20:24}"
  seq_slug="${seq_slug//[^A-Za-z0-9_-]/_}"
  printf 'oneoff_pre_%s_%s\n' "$seq_time" "$seq_slug"
}

CALIB_URL="$(to_swift_url "$CALIB_HTTP_URL")"
EXTRINSICS_URL="${CALIB_URL%/}"
EXTRINSICS_URL="${EXTRINSICS_URL/\/calibration\//\/calibration_output\/}/calibrate_extrinsics"

echo "Workflow:   $WORKFLOW_PATH"
echo "Pool:       $POOL"
echo "Extrinsics: $EXTRINSICS_URL"
echo "Dry run:    $DRY_RUN"
echo

for seq_http_url in "${SEQUENCE_HTTP_URLS[@]}"; do
  seq_url="$(to_swift_url "$seq_http_url")"
  seq_name="$(basename "${seq_url%/}")"
  workflow_name="$(short_workflow_name "$seq_name")"
  swift_output_base="${seq_url%/}"
  swift_output_base="${swift_output_base/\/data\//\/data_output\/}"

  cmd=(
    osmo workflow submit "$WORKFLOW_PATH"
    --set
    "workflow_name=$workflow_name"
    "rosbag_url=$seq_url"
    "extrinsics_url=$EXTRINSICS_URL"
    "swift_output_base=$swift_output_base"
    --pool "$POOL"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  echo "=== Submitting $seq_name ==="
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  echo
done
