#!/usr/bin/env bash
# Source this file to set Databricks credentials for MV HOI export QC queries.
#
# Usage:
#   source ~/secrets/setup_databricks_env.sh
#
# This repo copy is a template. Keep real credentials in a private copy outside
# git, such as ~/secrets/setup_databricks_env.sh.

export DATABRICKS_SERVER_HOSTNAME="REPLACE_ME"
export DATABRICKS_HTTP_PATH="REPLACE_ME"
export DATABRICKS_TOKEN="REPLACE_ME"

if [ "${DATABRICKS_SERVER_HOSTNAME}" = "REPLACE_ME" ] || \
   [ "${DATABRICKS_HTTP_PATH}" = "REPLACE_ME" ] || \
   [ "${DATABRICKS_TOKEN}" = "REPLACE_ME" ]; then
  echo "ERROR: Databricks credentials have not been configured in this script."
  echo ""
  echo "To configure credentials:"
  echo "  1. Create a private copy at ~/secrets/setup_databricks_env.sh"
  echo "  2. Replace DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN"
  echo "  3. Source that private file before running workflows/mv_hoi/export.py"
  return 1 2>/dev/null || exit 1
fi

echo "Databricks environment configured."
