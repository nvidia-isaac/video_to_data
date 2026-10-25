# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2D Challenge submission compliance, quota, and anti-reward-hacking gates."""

from tools.v2d_compliance.compliance_gate import (
    ComplianceError,
    ComplianceReport,
    QuotaStatus,
    audit_submission_artifact,
    check_bundle_atomicity,
    check_track_scope,
    check_upload_authorization,
    check_weekly_quota,
    detect_agent_environment,
)

__all__ = [
    "ComplianceError",
    "ComplianceReport",
    "QuotaStatus",
    "audit_submission_artifact",
    "check_bundle_atomicity",
    "check_track_scope",
    "check_upload_authorization",
    "check_weekly_quota",
    "detect_agent_environment",
]
