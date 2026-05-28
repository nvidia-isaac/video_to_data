#!/usr/bin/env bash
# Source this file to set AWS credentials for the HITL intake S3 bucket.
#
# Usage:
#   source ~/secrets/setup_hitl_aws_env.sh
#
# This repo copy is a template. Keep real credentials in a private copy outside
# git, such as ~/secrets/setup_hitl_aws_env.sh.

export AWS_ACCESS_KEY_ID="REPLACE_ME"
export AWS_SECRET_ACCESS_KEY="REPLACE_ME"
export AWS_DEFAULT_REGION="us-west-2"

if [ "${AWS_ACCESS_KEY_ID}" = "REPLACE_ME" ] || [ "${AWS_SECRET_ACCESS_KEY}" = "REPLACE_ME" ]; then
  echo "ERROR: HITL AWS credentials have not been configured in this script."
  echo ""
  echo "To configure credentials:"
  echo "  1. Create a private copy at ~/secrets/setup_hitl_aws_env.sh"
  echo "  2. Replace AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY with HITL S3 credentials"
  echo "  3. Source that private file from cron or mark_ready_cron.sh"
  return 1 2>/dev/null || exit 1
fi

echo "HITL AWS environment configured."
