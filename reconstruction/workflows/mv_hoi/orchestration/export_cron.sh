#!/usr/bin/env bash
set -euo pipefail

# Crontab:
# CRON_TZ=America/Los_Angeles
# 30 * * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/orchestration/export_cron.sh

export PATH=/usr/local/bin:/usr/bin:/bin
: "${HOME:?HOME must be set for private credential env files}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MV_HOI_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKDIR="$(cd -- "$MV_HOI_DIR/../.." && pwd -P)"
STATE_DIR="${MV_HOI_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/v2d-mv-hoi}"
HOST_ENV="${MV_HOI_ENV_FILE:-$STATE_DIR/host.env}"
[ ! -f "$HOST_ENV" ] || source "$HOST_ENV"
LOGDIR="$STATE_DIR/logs"
LOCKFILE="$STATE_DIR/locks/mv_hoi_export.lock"
VENV="${MV_HOI_VENV:-$STATE_DIR/venv}"
DATASET="${1:-${DATASET:-sc_office_4exo_1}}"
CSS_ENV="${S3_ENV:-${CSS_ENV:-}}"
KRATOS_DRS_ENV="${KRATOS_DRS_ENV:-$HOME/secrets/setup_kratos_drs_env.sh}"

clear_kratos_drs_env() {
  unset PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID
  unset PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET
  unset KRATOS_PROFILE KRATOS_AUTH_TYPE KRATOS_NAMESPACE KRATOS_DRS_WAREHOUSE_ID
}

mkdir -p "$LOGDIR" "$(dirname "$LOCKFILE")"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) export skipped: already running ===" 2>&1 | tee -a "$LOGDIR/export.log"
  exit 0
fi

if [ -n "$CSS_ENV" ]; then source "$CSS_ENV"; fi
if [ "${MV_HOI_QC_QUERY_ENABLED:-1}" = "0" ]; then
  echo "WARNING: QC query disabled; exports will remain WAITING_QC"
  clear_kratos_drs_env
elif [ -r "$KRATOS_DRS_ENV" ]; then
  if ! source "$KRATOS_DRS_ENV"; then
    echo "WARNING: QC query credentials unavailable; exports will remain WAITING_QC"
    clear_kratos_drs_env
  fi
else
  echo "WARNING: QC query env not found at $KRATOS_DRS_ENV; exports will remain WAITING_QC"
  clear_kratos_drs_env
fi

cd "$WORKDIR"
source "$VENV/bin/activate"

{
  echo "=== $(date -Is) export start: dataset=$DATASET ==="
  echo "Using CSS env=$CSS_ENV"
  echo "Using QC query env=$KRATOS_DRS_ENV"
  python -u workflows/mv_hoi/orchestration/export.py --dataset "$DATASET"
  echo "=== $(date -Is) export done ==="
} 2>&1 | tee -a "$LOGDIR/export.log"
