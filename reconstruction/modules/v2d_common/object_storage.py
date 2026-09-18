# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable S3 configuration shared by exports and host orchestration.

S3_* settings take precedence over explicit legacy CSS_* aliases. With no
endpoint override, S3 URLs use the SDK's standard endpoint and credential chain.
Swift URLs supply their own HTTPS endpoint unless an endpoint is configured.
"""
from __future__ import annotations

import os


def storage_env(name: str) -> str | None:
    return os.environ.get(f"S3_{name}") or os.environ.get(f"CSS_{name}") or None


def parse_storage_url(url: str) -> tuple[str | None, str, str]:
    """Return (endpoint, bucket, key prefix) for S3, Swift or bare bucket paths."""
    endpoint = storage_env("ENDPOINT_URL")
    value = url.strip()
    if value.startswith("swift://"):
        parts = value.removeprefix("swift://").rstrip("/").split("/", 3)
        if len(parts) < 3 or not all(parts[:3]):
            raise ValueError("Swift URL must be swift://host/account/bucket[/prefix]")
        host, _account, bucket = parts[:3]
        return endpoint or f"https://{host}", bucket, parts[3] if len(parts) > 3 else ""
    if value.startswith("s3://"):
        value = value.removeprefix("s3://").rstrip("/")
    elif "://" in value:
        raise ValueError("Remote path must use s3://, swift:// or bucket/prefix")
    else:
        value = value.strip("/")
    bucket, _, prefix = value.partition("/")
    if not bucket:
        raise ValueError("Remote path must include a bucket")
    return endpoint, bucket, prefix


def s3_client_kwargs(endpoint: str | None = None) -> dict:
    """Preserve the SDK credential chain when explicit credentials are absent."""
    access = storage_env("ACCESS_KEY")
    secret = storage_env("SECRET_KEY")
    if bool(access) != bool(secret):
        raise ValueError("Set both S3_ACCESS_KEY and S3_SECRET_KEY (legacy CSS_* aliases supported)")
    kwargs = {"endpoint_url": storage_env("ENDPOINT_URL") or endpoint}
    if access:
        kwargs.update(aws_access_key_id=access, aws_secret_access_key=secret)
    region = storage_env("REGION")
    if region:
        kwargs["region_name"] = region
    return kwargs
