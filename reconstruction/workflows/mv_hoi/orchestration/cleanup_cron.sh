#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/bin:/usr/local/bin:/usr/bin:/bin"
: "${HOME:?HOME must be set}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MV_HOI_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKDIR="$(cd -- "$MV_HOI_DIR/../.." && pwd -P)"
STATE_DIR="${MV_HOI_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/v2d-mv-hoi}"
HOST_ENV="${MV_HOI_ENV_FILE:-$STATE_DIR/host.env}"
[ ! -f "$HOST_ENV" ] || source "$HOST_ENV"
LOGDIR="$STATE_DIR/logs"
LOCKFILE="$STATE_DIR/locks/mv_hoi_cleanup.lock"
VENV="${MV_HOI_VENV:-$STATE_DIR/venv}"
CSS_ENV="${CSS_ENV:-$HOME/secrets/setup_css_env.sh}"

mkdir -p "$LOGDIR" "$(dirname "$LOCKFILE")"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) cleanup skipped: already running ===" >> "$LOGDIR/cleanup.log"
  exit 0
fi

source "$CSS_ENV"
cd "$WORKDIR"
source "$VENV/bin/activate"

{
  echo "=== $(date -Is) cleanup start ==="
  campaign="${MV_HOI_CLEANUP_CAMPAIGN:-${MV_HOI_BACKLOG_CAMPAIGN:-}}"
  if [ -z "$campaign" ]; then
    echo "No cleanup campaign configured; skipping"
  else
    python -u workflows/mv_hoi/orchestration/cleanup_intermediates.py \
      --campaign "$campaign" --configured-asynchronous --summary --apply
  fi
  echo "=== $(date -Is) cleanup done ==="
} 2>&1 | tee -a "$LOGDIR/cleanup.log"
