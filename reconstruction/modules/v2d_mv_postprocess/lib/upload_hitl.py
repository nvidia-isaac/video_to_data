# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage HITL intake files (tiled overlay video + SuperAnnotate JSON).

Runs after ``check_accuracy`` passes — task dependencies in the workflow
ensure this program only runs when upstream QC succeeded.

Writes files into ``output_dir`` following the HITL batch layout::

    {output_dir}/
      dataset/{video_name}.mp4
      jsons/{video_name}.json

The OSMO workflow copies ``output_dir`` to the HITL intake S3 bucket (via a
second ``outputs`` entry of ``{hitl_s3_base}/{hitl_batch_name}/``), so this
program only prepares the local files.  The ``video_url`` written into the
SuperAnnotate JSON points at the public HTTPS URL the video will resolve to
after OSMO's upload completes.

``object_id`` and ``action_desc`` are read from the ``hoi_metadata.yaml``
produced by the preprocess module.

Usage (inside container):
    python -m v2d.mv.postprocess.lib.upload_hitl \
        --overlay_dir /data/render_hoi_overlay \
        --hoi_metadata_path /data/mv_preprocess/hoi_metadata.yaml \
        --output_dir  /data/upload_hitl \
        --hitl_s3_base s3://hitl-intake-testing/production-folder/... \
        --hitl_batch_name sc_office_4exo_1_v3 \
        --video_name v2d_mv_hoi_reconstruction_0-1-0_20260416_120000 \
        --s3_region us-west-2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import yaml


def _parse_s3_url(url: str) -> tuple[str, str]:
    stripped = url.rstrip("/").replace("s3://", "")
    parts = stripped.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _read_hoi_metadata(hoi_metadata_path: str) -> tuple[str, str]:
    """Return (object_id, action_desc) from hoi_metadata.yaml."""
    with open(hoi_metadata_path) as f:
        meta = yaml.safe_load(f) or {}
    object_id = (
        meta.get("object", {}).get("id")
        or meta.get("object_id")
        or meta.get("object_name")
        or ""
    )
    action_desc = meta.get("action_desc", "")
    return str(object_id), str(action_desc)


def upload_hitl(
    overlay_dir: str,
    hoi_metadata_path: str,
    output_dir: str,
    hitl_s3_base: str,
    hitl_batch_name: str,
    video_name: str,
    s3_region: str = "us-west-2",
) -> dict:
    object_id, action_desc = _read_hoi_metadata(hoi_metadata_path)

    video_src = os.path.join(overlay_dir, "tiled_hoi_overlay.mp4")
    if not os.path.isfile(video_src):
        candidates = sorted(glob.glob(os.path.join(overlay_dir, "*.mp4")))
        video_src = candidates[0] if candidates else None
    if not video_src:
        raise FileNotFoundError(f"no overlay video found in {overlay_dir}")

    dataset_dir = os.path.join(output_dir, "dataset")
    jsons_dir = os.path.join(output_dir, "jsons")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(jsons_dir, exist_ok=True)

    video_dst = os.path.join(dataset_dir, f"{video_name}.mp4")
    print(f"Staging video: {video_src} -> {video_dst}")
    shutil.copy2(video_src, video_dst)

    bucket, base_prefix = _parse_s3_url(hitl_s3_base)
    video_key = f"{base_prefix}/{hitl_batch_name}/dataset/{video_name}.mp4"
    video_url = f"https://{bucket}.s3.{s3_region}.amazonaws.com/{video_key}"

    sa_json = {
        "video_url": video_url,
        "object_id": object_id,
        "action_desc": action_desc,
    }
    json_dst = os.path.join(jsons_dir, f"{video_name}.json")
    print(f"Writing SuperAnnotate JSON: {json_dst}")
    with open(json_dst, "w") as f:
        json.dump(sa_json, f, indent=2)

    result = {
        "status": "STAGED",
        "hitl_batch_name": hitl_batch_name,
        "video_name": video_name,
        "object_id": object_id,
        "action_desc": action_desc,
        "video_url": video_url,
        "video_path": video_dst,
        "json_path": json_dst,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage tiled overlay + SuperAnnotate JSON for HITL intake"
    )
    parser.add_argument("--overlay_dir", type=str, required=True,
                        help="render_hoi_overlay output (contains tiled_hoi_overlay.mp4)")
    parser.add_argument("--hoi_metadata_path", type=str, required=True,
                        help="Path to hoi_metadata.yaml (from preprocess task) — "
                             "object_id and action_desc are read from here")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hitl_s3_base", type=str, required=True,
                        help="S3 base where the workflow will upload this task's outputs "
                             "(used to compute the public video_url written into the JSON)")
    parser.add_argument("--hitl_batch_name", type=str, required=True,
                        help="Batch folder name under hitl_s3_base (groups videos by batch)")
    parser.add_argument("--video_name", type=str, required=True,
                        help="Filename stem for the mp4 and json "
                             "(typically the OSMO workflow name)")
    parser.add_argument("--s3_region", type=str, default="us-west-2",
                        help="AWS region for the HITL S3 HTTPS URL")
    args = parser.parse_args()

    upload_hitl(
        overlay_dir=args.overlay_dir,
        hoi_metadata_path=args.hoi_metadata_path,
        output_dir=args.output_dir,
        hitl_s3_base=args.hitl_s3_base,
        hitl_batch_name=args.hitl_batch_name,
        video_name=args.video_name,
        s3_region=args.s3_region,
    )
