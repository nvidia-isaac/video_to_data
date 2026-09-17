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
LOCKFILE="$STATE_DIR/locks/mv_hoi_campaign.lock"
VENV="${MV_HOI_VENV:-$STATE_DIR/venv}"
CSS_ENV="${CSS_ENV:-$HOME/secrets/setup_css_env.sh}"
KRATOS_DRS_ENV="${KRATOS_DRS_ENV:-$HOME/secrets/setup_kratos_drs_env.sh}"

clear_kratos_drs_env() {
  unset PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID
  unset PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET
  unset KRATOS_PROFILE KRATOS_AUTH_TYPE KRATOS_NAMESPACE KRATOS_DRS_WAREHOUSE_ID
}

mkdir -p "$LOGDIR" "$(dirname "$LOCKFILE")"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "=== $(date -Is) campaign skipped: already running ===" | tee -a "$LOGDIR/campaign.log"
  exit 0
fi

source "$CSS_ENV"
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
  echo "=== $(date -Is) campaign start ==="
  campaign_scope="${MV_HOI_CAMPAIGN_SCOPE:-all}"
  case "$campaign_scope" in
    all|revalidation|backlog) ;;
    *)
      echo "Invalid MV_HOI_CAMPAIGN_SCOPE=$campaign_scope; expected all, revalidation, or backlog"
      exit 2
      ;;
  esac
  args=(cycle)
  [ "$campaign_scope" = "backlog" ] || [ -z "${MV_HOI_REVALIDATION_CAMPAIGN:-}" ] || \
    args+=(--revalidation-campaign "$MV_HOI_REVALIDATION_CAMPAIGN")
  [ "$campaign_scope" = "revalidation" ] || [ -z "${MV_HOI_BACKLOG_CAMPAIGN:-}" ] || \
    args+=(--backlog-campaign "$MV_HOI_BACKLOG_CAMPAIGN")
  [ "${MV_HOI_VERIFY_PAYLOAD_HASHES:-0}" != "1" ] || \
    args+=(--verify-payload-hashes)
  if [ "${#args[@]}" -eq 1 ]; then
    echo "No campaign names configured; skipping"
  else
    python -u workflows/mv_hoi/orchestration/campaign_controller.py "${args[@]}"
  fi
  echo "=== $(date -Is) campaign done ==="
} 2>&1 | tee -a "$LOGDIR/campaign.log"
