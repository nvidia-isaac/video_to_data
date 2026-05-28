#!/usr/bin/env bash
set -euo pipefail

# Crontab:
# CRON_TZ=America/Los_Angeles
# 0 */1 * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/submit_calibration_cron.sh

export PATH=/usr/local/bin:/usr/bin:/bin
: "${HOME:?HOME must be set for private credential env files}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKDIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
LOGDIR="$WORKDIR/workflows/mv_hoi/logs"
LOCKFILE="/tmp/mv_calibration_submit.lock"
CSS_ENV="${CSS_ENV:-$HOME/secrets/setup_css_env.sh}"

mkdir -p "$LOGDIR"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) calibration submit skipped: already running ===" 2>&1 | tee -a "$LOGDIR/submit_calibration.log"
  exit 0
fi

source "$CSS_ENV"

cd "$WORKDIR"
source .venv/bin/activate

{
  echo "=== $(date -Is) calibration submit start ==="
  echo "Using CSS env=$CSS_ENV"
  python -u workflows/mv_hoi/submit.py --dataset sc_office_4exo_1 --pipeline mv_calibration --retry_failed
  echo "=== $(date -Is) calibration submit done ==="
} 2>&1 | tee -a "$LOGDIR/submit_calibration.log"
