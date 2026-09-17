#!/usr/bin/env bash
# Source this file to configure Kratos DRS for MV HOI export QC queries.
#
# Usage:
#   source ~/secrets/setup_kratos_drs_env.sh
#
# This repo copy is a template. Keep real credentials in a private copy outside
# git, such as ~/secrets/setup_kratos_drs_env.sh.

export PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID="REPLACE_ME"
export PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET="REPLACE_ME"
export KRATOS_PROFILE="production"
export KRATOS_AUTH_TYPE="service"
export KRATOS_NAMESPACE="REPLACE_ME"
# This is the Kratos DRS warehouse ID, not the Databricks SQL UI warehouse ID.
export KRATOS_DRS_WAREHOUSE_ID="REPLACE_ME"

if [ "$PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID" = "REPLACE_ME" ] || \
   [ "$PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET" = "REPLACE_ME" ] || \
   [ "$KRATOS_NAMESPACE" = "REPLACE_ME" ] || \
   [ "$KRATOS_DRS_WAREHOUSE_ID" = "REPLACE_ME" ]; then
  echo "ERROR: Kratos DRS SSA credentials have not been configured in this script."
  echo ""
  echo "To configure credentials:"
  echo "  1. Create a private copy at ~/secrets/setup_kratos_drs_env.sh"
  echo "  2. Replace the SSA client ID and client secret"
  echo "  3. Source that private file before running workflows/mv_hoi/orchestration/export.py"
  return 1 2>/dev/null || exit 1
fi

echo "Kratos DRS environment configured."
