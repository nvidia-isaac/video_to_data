#!/usr/bin/env bash
set -euo pipefail

# Crontab:
# CRON_TZ=America/Los_Angeles
# 30 * * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/orchestration/publish_status_cron.sh

export PATH=/usr/local/bin:/usr/bin:/bin
: "${HOME:?HOME must be set for private credential env files}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MV_HOI_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKDIR="$(cd -- "$MV_HOI_DIR/../.." && pwd -P)"
STATE_DIR="${MV_HOI_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/v2d-mv-hoi}"
HOST_ENV="${MV_HOI_ENV_FILE:-$STATE_DIR/host.env}"
[ ! -f "$HOST_ENV" ] || source "$HOST_ENV"
LOGDIR="$STATE_DIR/logs"
LOCKFILE="$STATE_DIR/locks/mv_hoi_publish_status.lock"
VENV="${MV_HOI_VENV:-$STATE_DIR/venv}"
DATASET="${1:-${DATASET:-sc_office_4exo_1}}"
CSS_ENV="${S3_ENV:-${CSS_ENV:-}}"
STATUS_PUBLISH_ENV="${STATUS_PUBLISH_ENV:-$HOME/secrets/setup_mv_hoi_status_publish_env.sh}"

mkdir -p "$LOGDIR" "$(dirname "$LOCKFILE")"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) publish_status skipped: already running ===" 2>&1 | tee -a "$LOGDIR/publish_status.log"
  exit 0
fi

if [ -n "$CSS_ENV" ]; then source "$CSS_ENV"; fi
source "$STATUS_PUBLISH_ENV"

cd "$WORKDIR"
source "$VENV/bin/activate"

{
  echo "=== $(date -Is) publish_status start: dataset=$DATASET ==="
  echo "Using CSS env=$CSS_ENV"
  echo "Using status publish env=$STATUS_PUBLISH_ENV"
  python -u workflows/mv_hoi/orchestration/publish_status.py \
    --dataset "$DATASET" \
    --summary-worksheet summary \
    --data-worksheet data_sequence_status \
    --calibration-worksheet calibration_sequence_status
  echo "=== $(date -Is) publish_status done ==="
} 2>&1 | tee -a "$LOGDIR/publish_status.log"
