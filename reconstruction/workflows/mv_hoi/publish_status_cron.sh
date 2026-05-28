#!/usr/bin/env bash
set -euo pipefail

# Crontab:
# CRON_TZ=America/Los_Angeles
# 10 * * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/publish_status_cron.sh

export PATH=/usr/local/bin:/usr/bin:/bin
: "${HOME:?HOME must be set for private credential env files}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKDIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
LOGDIR="$WORKDIR/workflows/mv_hoi/logs"
LOCKFILE="/tmp/mv_hoi_publish_status.lock"
DATASET="${1:-${DATASET:-sc_office_4exo_1}}"
CSS_ENV="${CSS_ENV:-$HOME/secrets/setup_css_env.sh}"
STATUS_PUBLISH_ENV="${STATUS_PUBLISH_ENV:-$HOME/secrets/setup_mv_hoi_status_publish_env.sh}"

mkdir -p "$LOGDIR"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) publish_status skipped: already running ===" 2>&1 | tee -a "$LOGDIR/publish_status.log"
  exit 0
fi

source "$CSS_ENV"
source "$STATUS_PUBLISH_ENV"

cd "$WORKDIR"
source .venv/bin/activate

{
  echo "=== $(date -Is) publish_status start: dataset=$DATASET ==="
  echo "Using CSS env=$CSS_ENV"
  echo "Using status publish env=$STATUS_PUBLISH_ENV"
  python -u workflows/mv_hoi/publish_status.py \
    --dataset "$DATASET" \
    --pipeline mv_calibration \
    --latest \
    --status-worksheet calibration_status \
    --summary-worksheet calibration_summary
  python -u workflows/mv_hoi/publish_status.py \
    --dataset "$DATASET" \
    --pipeline mv_hoi_reconstruction \
    --latest \
    --status-worksheet reconstruction_status \
    --summary-worksheet reconstruction_summary
  echo "=== $(date -Is) publish_status done ==="
} 2>&1 | tee -a "$LOGDIR/publish_status.log"
