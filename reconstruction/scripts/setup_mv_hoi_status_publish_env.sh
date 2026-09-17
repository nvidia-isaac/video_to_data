#!/usr/bin/env bash
# Source this file to publish MV HOI status to Google Sheets.
#
# Usage:
#   source ~/secrets/setup_mv_hoi_status_publish_env.sh
#
# This repo copy is a template. Keep real credentials in a private copy outside
# git, such as ~/secrets/setup_mv_hoi_status_publish_env.sh.

export GOOGLE_APPLICATION_CREDENTIALS="$HOME/secrets/mv-hoi-status-publisher.json"
export MV_HOI_STATUS_SPREADSHEET_ID="REPLACE_ME"

if [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
  echo "ERROR: Google service account key not found: ${GOOGLE_APPLICATION_CREDENTIALS}"
  echo ""
  echo "To configure credentials:"
  echo "  1. Save the service account JSON at ~/secrets/mv-hoi-status-publisher.json"
  echo "  2. Share the target Google Sheet with the service account email as Editor"
  echo "  3. Create a private copy at ~/secrets/setup_mv_hoi_status_publish_env.sh"
  echo "  4. Set MV_HOI_STATUS_SPREADSHEET_ID to the spreadsheet ID"
  return 1 2>/dev/null || exit 1
fi

if [ "${MV_HOI_STATUS_SPREADSHEET_ID}" = "REPLACE_ME" ]; then
  echo "ERROR: MV_HOI_STATUS_SPREADSHEET_ID has not been configured."
  echo ""
  echo "Set it to the value from the Google Sheet URL:"
  echo "  https://docs.google.com/spreadsheets/d/<spreadsheet-id>/edit"
  return 1 2>/dev/null || exit 1
fi

echo "MV HOI status publish environment configured."
