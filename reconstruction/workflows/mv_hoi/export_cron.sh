#!/usr/bin/env bash
set -euo pipefail

# Crontab:
# CRON_TZ=America/Los_Angeles
# 30 * * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/export_cron.sh

export PATH=/usr/local/bin:/usr/bin:/bin
: "${HOME:?HOME must be set for private credential env files}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKDIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
LOGDIR="$WORKDIR/workflows/mv_hoi/logs"
LOCKFILE="/tmp/mv_hoi_export.lock"
DATASET="${1:-${DATASET:-sc_office_4exo_1}}"
CSS_ENV="${CSS_ENV:-$HOME/secrets/setup_css_env.sh}"
DATABRICKS_ENV="${DATABRICKS_ENV:-$HOME/secrets/setup_databricks_env.sh}"

mkdir -p "$LOGDIR"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) export skipped: already running ===" 2>&1 | tee -a "$LOGDIR/export.log"
  exit 0
fi

source "$CSS_ENV"
source "$DATABRICKS_ENV"

cd "$WORKDIR"
source .venv/bin/activate

{
  echo "=== $(date -Is) export start: dataset=$DATASET ==="
  echo "Using CSS env=$CSS_ENV"
  echo "Using Databricks env=$DATABRICKS_ENV"
  python -u workflows/mv_hoi/export.py --dataset "$DATASET"
  echo "=== $(date -Is) export done ==="
} 2>&1 | tee -a "$LOGDIR/export.log"
