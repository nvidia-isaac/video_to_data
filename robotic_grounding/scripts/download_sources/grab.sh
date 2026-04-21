#!/bin/bash
# Download GRAB dataset (ECCV 2020) and upload to CSS.
#
# GRAB distributes as 10 per-subject ZIPs + an object_meshes ZIP, gated behind
# an account on https://grab.is.tue.mpg.de/.  Downloads are authenticated via a
# session cookie — the "URLs" on the downloads page only work while logged in,
# so a plain `curl <url>` from a new shell gets redirected to the login page.
#
# Flow:
#   1. POST email+password to https://grab.is.tue.mpg.de/login.php to mint a
#      session cookie.
#   2. curl each download.php URL with that cookie jar.
#
# This script then:
#   - Fetches the ZIPs into /tmp/grab/zips
#   - Extracts them into the conventional layout (s1/, s2/, ..., s10/, object_meshes/)
#   - Uploads the extracted data to CSS at
#     s3://datasets/v2d/human_motion_data/grab/dataset/
#
# Run inside an OSMO dev_env container with CSS + MPG credentials exported.
#
# Required env vars:
#   CSS_ENDPOINT_URL, CSS_ACCESS_KEY, CSS_SECRET_KEY
#   GRAB_USERNAME  — email you registered on grab.is.tue.mpg.de
#   GRAB_PASSWORD  — password for that account
#
# Optional env vars:
#   GRAB_SUBJECTS  — space-separated list of subjects to fetch (default: "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10")
#   SKIP_UPLOAD=1  — only download + extract; don't push to CSS

set -ex

: "${CSS_ENDPOINT_URL:?CSS_ENDPOINT_URL must be set}"
: "${CSS_ACCESS_KEY:?CSS_ACCESS_KEY must be set}"
: "${CSS_SECRET_KEY:?CSS_SECRET_KEY must be set}"
: "${GRAB_USERNAME:?GRAB_USERNAME must be set (registered email on grab.is.tue.mpg.de)}"
: "${GRAB_PASSWORD:?GRAB_PASSWORD must be set}"

DATASET=grab
STAGING=/tmp/${DATASET}
CSS_DEST=s3://datasets/v2d/human_motion_data/${DATASET}/dataset/
GRAB_SUBJECTS="${GRAB_SUBJECTS:-s1 s2 s3 s4 s5 s6 s7 s8 s9 s10}"

mkdir -p "${STAGING}/zips" "${STAGING}/extracted"
cd "${STAGING}"

# 1. Install tools
apt-get update -qq && apt-get install -y -qq unzip curl
if ! command -v aws &>/dev/null; then
  if command -v pip &>/dev/null; then pip install -q awscli
  else python3 -m pip install -q awscli; fi
fi

# 2. Login to grab.is.tue.mpg.de to mint a session cookie
COOKIES="${STAGING}/mpg_cookies.txt"
rm -f "${COOKIES}"
echo "[grab] logging in as ${GRAB_USERNAME}"
curl -s -c "${COOKIES}" -b "${COOKIES}" \
  -d "username=${GRAB_USERNAME}&password=${GRAB_PASSWORD}" \
  "https://grab.is.tue.mpg.de/login.php" >/dev/null

# Verify login succeeded — the session cookie is named PHPSESSID and should
# appear in the cookie jar.
if ! grep -q PHPSESSID "${COOKIES}"; then
  echo "[grab] ERROR: login did not return a session cookie. Check credentials." >&2
  exit 1
fi

# Build the list of ZIPs to fetch. The download endpoint is the same for all
# files on the downloads page.
ZIP_FILES=()
for s in ${GRAB_SUBJECTS}; do
  ZIP_FILES+=("grab__${s}.zip")
done
ZIP_FILES+=("object_meshes.zip")

# 3. Download each ZIP through the authenticated session
DOWNLOAD_BASE="https://download.is.tue.mpg.de/download.php?domain=grab&resume=1&sfile="
for fname in "${ZIP_FILES[@]}"; do
  out="zips/${fname}"
  if [ -s "${out}" ]; then
    echo "[grab] skipping ${fname} (already downloaded)"
    continue
  fi
  url="${DOWNLOAD_BASE}${fname}"
  echo "[grab] downloading ${fname}"
  curl -L --fail -b "${COOKIES}" -c "${COOKIES}" -o "${out}" "${url}"
done

# 4. Extract each ZIP into extracted/
cd extracted
for z in ../zips/*.zip; do
  echo "[grab] extracting $(basename "$z")"
  unzip -q -o "$z"
done

echo "[grab] extracted layout:"
ls -la

# 5. Upload to CSS.  GRAB is thousands of small .npz files — use high
# concurrency so per-file API overhead doesn't dominate.
if [ "${SKIP_UPLOAD:-0}" != "1" ]; then
  aws configure set default.s3.max_concurrent_requests 100
  aws configure set default.s3.max_queue_size 10000
  export AWS_ACCESS_KEY_ID=${CSS_ACCESS_KEY}
  export AWS_SECRET_ACCESS_KEY=${CSS_SECRET_KEY}
  aws s3 sync ./ "${CSS_DEST}" \
    --endpoint-url "${CSS_ENDPOINT_URL}" --region us-east-1

  echo "[grab] upload complete."
  echo "  verify from host: python scripts/list_css_sequences.py --dataset grab --stage raw"
else
  echo "[grab] SKIP_UPLOAD=1 set — skipping CSS upload. Data staged at ${STAGING}/extracted/"
fi
