#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the scoped V2D rigid-object profile through simready-validate."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROFILE_ID = "V2D-HOI-Rigid-Object"
PROFILE_VERSION = "1.0.0"
MODULE_DIR = Path(__file__).resolve().parent


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def issue_to_dict(issue) -> dict:
    requirement = getattr(issue, "requirement", None)
    rule = getattr(issue, "rule", None)
    severity = getattr(issue, "severity", None)
    return {
        "severity": getattr(severity, "name", str(severity)),
        "requirement": getattr(requirement, "code", None),
        "rule": getattr(rule, "__name__", str(rule)) if rule else None,
        "message": str(getattr(issue, "message", "")),
        "at": str(getattr(issue, "at", "")),
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def register_validation_profile(foundation_root: Path) -> None:
    from omni.asset_validator import (
        Capability,
        CapabilityRegistry,
        Profile,
        ProfileRegistry,
        RequirementsRegistry,
    )

    sys.path.insert(0, str(foundation_root))
    importlib.import_module("capabilities")

    for feature_path in sorted((MODULE_DIR / "specs" / "features").glob("*.json")):
        feature_data = json.loads(feature_path.read_text(encoding="utf-8"))
        requirements = []
        missing_requirements = []
        for code in feature_data["requirements"]:
            requirement = RequirementsRegistry().find_requirement(code)
            if requirement is None:
                missing_requirements.append(code)
            else:
                requirements.append(requirement)
        if missing_requirements:
            raise RuntimeError(
                f"Feature {feature_data['id']} has unavailable requirements: "
                + ", ".join(missing_requirements)
            )
        custom_data = {
            key: value
            for key, value in feature_data.items()
            if key not in {"id", "version", "path", "requirements"}
        }
        feature_class = type(
            feature_data["id"],
            (Capability,),
            {
                "id": feature_data["id"],
                "version": feature_data["version"],
                "path": feature_data.get("path", ""),
                "requirements": requirements,
                "custom_data": custom_data,
            },
        )
        CapabilityRegistry().add(feature_class())

    profiles_path = MODULE_DIR / "specs" / "profiles" / "profiles.toml"
    profiles_data = tomllib.loads(profiles_path.read_text(encoding="utf-8"))
    feature_refs = profiles_data[PROFILE_ID][PROFILE_VERSION]["features"]
    features = []
    for feature_ref in feature_refs:
        feature_id, config = next(iter(feature_ref.items()))
        feature = CapabilityRegistry().find(feature_id, config["version"])
        if feature is None:
            raise RuntimeError(
                f"Profile {PROFILE_ID} references unavailable feature {feature_id}"
            )
        features.append(feature)
    profile_class = type(
        PROFILE_ID,
        (Profile,),
        {
            "id": PROFILE_ID,
            "version": PROFILE_VERSION,
            "path": profiles_path,
            "capabilities": features,
        },
    )
    ProfileRegistry().add(profile_class())


def run(asset: Path, report_path: Path) -> int:
    import simready.validate as simready_validate

    foundation_root = Path(os.environ["SIMREADY_FOUNDATION_SPEC_ROOT"])
    validator_metadata = {
        "simready_validate_version": package_version("simready-validate"),
        "asset_validator_version": package_version("omniverse-asset-validator"),
        "usd_profiles_version": package_version("omniverse-usd-profiles"),
        "usd_core_version": package_version("usd-core"),
        "foundation_core_version": package_version("simready-foundation-core"),
        "foundation_commit": os.environ.get("SIMREADY_FOUNDATION_COMMIT", "unknown"),
    }

    try:
        simready_validate.destroy()
        register_validation_profile(foundation_root)
        result = simready_validate.validate_asset(
            simready_validate.AssetValidationConfig(
                asset_path=str(asset),
                profile_id=PROFILE_ID,
                profile_version=PROFILE_VERSION,
                write_metadata=False,
            )
        )
        if result is None:
            raise RuntimeError("simready-validate returned no result")

        feature_results = result.features_summary
        passed = bool(feature_results) and all(
            feature.get("passed", False) for feature in feature_results.values()
        )
        issues = [issue_to_dict(issue) for issue in result.issues]
        failed_requirements = sorted(
            {
                issue["requirement"]
                for issue in issues
                if issue.get("requirement") is not None
            }
        )
        report = {
            "schema_version": 1,
            "status": "passed" if passed else "failed",
            "passed": passed,
            "asset": str(asset),
            "profile": {
                "id": result.profile_id,
                "version": result.profile_version,
                "official_simready_certification": False,
            },
            "features": feature_results,
            "failed_requirements": failed_requirements,
            "issues": issues,
            "validator": validator_metadata,
        }
        write_report(report_path, report)
        print(json.dumps(report, indent=2))
        return 0 if passed else 1
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "error",
            "passed": False,
            "asset": str(asset),
            "profile": {
                "id": PROFILE_ID,
                "version": PROFILE_VERSION,
                "official_simready_certification": False,
            },
            "error": f"{type(error).__name__}: {error}",
            "validator": validator_metadata,
        }
        write_report(report_path, report)
        print(json.dumps(report, indent=2))
        return 2
    finally:
        simready_validate.destroy()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger().setLevel(logging.WARNING)
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.asset.resolve(), args.report.resolve()))


if __name__ == "__main__":
    main()
