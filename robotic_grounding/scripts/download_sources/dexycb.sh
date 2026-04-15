#!/bin/bash
# Download DexYCB dataset (NVLabs, CVPR 2021) and upload to CSS.
#
# DexYCB is distributed as a single ~119 GB archive on Google Drive
# (dex-ycb-20210415.tar.gz) that contains all 10 subjects + calibration +
# models + BOP annotations.  The URL is gated behind a click-through license
# at https://dex-ycb.github.io/, so there is no programmatic auth flow —
# supply the Google Drive file ID via the DEXYCB_GDRIVE_ID env var after
# you've accepted the license on the provider page in your browser.
#
# Run inside an OSMO dev_env container (2 TB /tmp storage recommended).
#
# Required env vars:
#   CSS_ENDPOINT_URL, CSS_ACCESS_KEY, CSS_SECRET_KEY
#   DEXYCB_GDRIVE_ID   — Google Drive file ID of dex-ycb-20210415.tar.gz.

set -ex

: "${CSS_ENDPOINT_URL:?CSS_ENDPOINT_URL must be set}"
: "${CSS_ACCESS_KEY:?CSS_ACCESS_KEY must be set}"
: "${CSS_SECRET_KEY:?CSS_SECRET_KEY must be set}"
: "${DEXYCB_GDRIVE_ID:?DEXYCB_GDRIVE_ID must be set (Google Drive file ID of dex-ycb-20210415.tar.gz)}"

DATASET=dexycb
STAGING=/tmp/${DATASET}
CSS_DEST=s3://datasets/v2d/human_motion_data/${DATASET}/dataset/
ARCHIVE=${STAGING}/dex-ycb-20210415.tar.gz

mkdir -p "${STAGING}/extracted"
cd "${STAGING}"

# 1. Tools
apt-get update -qq && apt-get install -y -qq curl tar
if ! command -v gdown &>/dev/null; then
  pip install -q gdown
fi
if ! command -v aws &>/dev/null; then
  if command -v pip &>/dev/null; then pip install -q awscli
  else python3 -m pip install -q awscli; fi
fi

# 2. Download the single 119 GB archive (gdown resolves the Drive confirm token).
if [ -s "${ARCHIVE}" ]; then
  echo "[dexycb] archive already present at ${ARCHIVE}"
else
  echo "[dexycb] downloading dex-ycb-20210415.tar.gz via gdown (this will take a while)"
  gdown --fuzzy --id "${DEXYCB_GDRIVE_ID}" -O "${ARCHIVE}"
fi

# 3. Extract into extracted/.
echo "[dexycb] extracting ${ARCHIVE}"
tar -xzf "${ARCHIVE}" -C extracted

echo "[dexycb] extracted layout:"
ls -la extracted/

# 4. Upload to CSS, excluding the RGB (*.jpg) + aligned depth (*.png) frames
# we don't use for retargeting.  Drop the excludes if you need them later.
aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.max_queue_size 10000
export AWS_ACCESS_KEY_ID=${CSS_ACCESS_KEY}
export AWS_SECRET_ACCESS_KEY=${CSS_SECRET_KEY}
aws s3 sync extracted/ "${CSS_DEST}" \
  --endpoint-url "${CSS_ENDPOINT_URL}" --region us-east-1 \
  --exclude "*.jpg" --exclude "*.png"

echo "[dexycb] upload complete."
echo "  verify from host: python scripts/list_css_sequences.py --dataset dexycb --stage raw"
