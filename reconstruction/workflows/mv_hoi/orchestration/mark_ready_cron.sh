#!/usr/bin/env bash
set -euo pipefail

# Crontab:
# CRON_TZ=America/Los_Angeles
# 50 * * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/orchestration/mark_ready_cron.sh

export PATH=/usr/local/bin:/usr/bin:/bin
: "${HOME:?HOME must be set for private credential env files}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MV_HOI_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKDIR="$(cd -- "$MV_HOI_DIR/../.." && pwd -P)"
STATE_DIR="${MV_HOI_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/v2d-mv-hoi}"
HOST_ENV="${MV_HOI_ENV_FILE:-$STATE_DIR/host.env}"
[ ! -f "$HOST_ENV" ] || source "$HOST_ENV"
LOGDIR="$STATE_DIR/logs"
LOCKFILE="$STATE_DIR/locks/mv_hoi_mark_ready.lock"
VENV="${MV_HOI_VENV:-$STATE_DIR/venv}"
DATASET="${1:-${DATASET:-sc_office_4exo_1}}"
LOCAL_TZ="${LOCAL_TZ:-America/Los_Angeles}"
BATCH_DATE="${BATCH_DATE:-$(TZ="$LOCAL_TZ" date -d yesterday +%Y%m%d)}"
CSS_ENV="${S3_ENV:-${CSS_ENV:-}}"
HITL_AWS_ENV="${HITL_AWS_ENV:-$HOME/secrets/setup_hitl_aws_env.sh}"

mkdir -p "$LOGDIR" "$(dirname "$LOCKFILE")"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) mark_ready skipped: already running ===" 2>&1 | tee -a "$LOGDIR/mark_ready.log"
  exit 0
fi

if [ -n "$CSS_ENV" ]; then source "$CSS_ENV"; fi

cd "$WORKDIR"
source "$VENV/bin/activate"

# Cron does not load ~/.bashrc. Clear any ambient AWS identity, then source the
# dedicated HITL S3 credentials used by this job.
unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
source "$HITL_AWS_ENV"

read -r HITL_S3_BASE BATCH_NAME <<< "$(
  python - "$DATASET" "$BATCH_DATE" <<'PY'
import sys
import yaml

dataset = sys.argv[1]
batch_date = sys.argv[2]

sys.path.insert(0, "workflows/mv_hoi/orchestration")
from config_utils import RECON_PIPELINE, RECONSTRUCTION_WORKFLOW, get_workflow_cfg

with open("workflows/mv_hoi/config.yaml") as f:
    config = yaml.safe_load(f)

dataset_cfg = config["datasets"][dataset]
workflow_cfg = get_workflow_cfg(dataset_cfg, RECON_PIPELINE, RECONSTRUCTION_WORKFLOW)
base = workflow_cfg["hitl_s3_base"].rstrip("/")
template = workflow_cfg.get("hitl_batch_name_template", "batch_{date}")
print(base, template.format(date=batch_date))
PY
)"

{
  echo "=== $(date -Is) mark_ready start: dataset=$DATASET batch=$BATCH_NAME ==="
  echo "Using CSS env=$CSS_ENV"
  echo "Using HITL AWS env=$HITL_AWS_ENV region=${AWS_DEFAULT_REGION:-unset}"

  if ! listing="$(aws s3 ls "$HITL_S3_BASE/" 2>&1)"; then
    echo "ERROR listing HITL base: $HITL_S3_BASE/"
    echo "$listing"
    exit 1
  fi

  if ! grep -q "PRE ${BATCH_NAME}/" <<< "$listing"; then
    echo "No previous-day batch found, skipping: $HITL_S3_BASE/$BATCH_NAME/"
    echo "=== $(date -Is) mark_ready skipped ==="
    exit 0
  fi

  python -u workflows/mv_hoi/orchestration/mark_ready.py --dataset "$DATASET" --batch "$BATCH_NAME"
  echo "=== $(date -Is) mark_ready done ==="
} 2>&1 | tee -a "$LOGDIR/mark_ready.log"
