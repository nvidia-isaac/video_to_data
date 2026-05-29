#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
# Source this file to set CSS (PDX) credentials for sync_css.py.
#
# Usage:
#   source reconstruction/scripts/setup_css_env.sh

export CSS_ENDPOINT_URL="https://pdx.s8k.io"
export CSS_ACCESS_KEY="v2p:AUTH_team-isaac"
export CSS_SECRET_KEY="REPLACE_ME"

if [ "${CSS_ACCESS_KEY}" = "REPLACE_ME" ] || [ "${CSS_SECRET_KEY}" = "REPLACE_ME" ]; then
  echo "ERROR: CSS credentials have not been configured in this script."
  echo ""
  echo "To obtain credentials:"
  echo "  1. Go to the CSS portal: https://pdx.s8k.io"
  echo "  2. Generate or retrieve your access key and secret key"
  echo "  3. Edit reconstruction/scripts/setup_css_env.sh and replace the REPLACE_ME values"
  return 1 2>/dev/null || exit 1
fi

echo "CSS environment configured."
