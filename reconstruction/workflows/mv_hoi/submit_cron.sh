#!/usr/bin/env bash
set -euo pipefail

export HOME=/home/dzou
export PATH=/usr/local/bin:/usr/bin:/bin

REPO="$HOME/code/robotics/video_to_data"
WORKDIR="$REPO/reconstruction"
LOGDIR="$WORKDIR/workflows/mv_hoi/logs"
LOCKFILE="/tmp/mv_hoi_submit.lock"

mkdir -p "$LOGDIR"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) submit skipped: already running ===" 2>&1 | tee -a "$LOGDIR/submit.log"
  exit 0
fi

source "$HOME/bin/setup_css_env.sh"

cd "$WORKDIR"
source .venv/bin/activate

{
  echo "=== $(date -Is) submit start ==="
  python -u workflows/mv_hoi/submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --retry_failed
  echo "=== $(date -Is) submit done ==="
} 2>&1 | tee -a "$LOGDIR/submit.log"
