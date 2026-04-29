# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Unified motion parquet schema (`motion_v1`).

Shared format used by all whole-body retargeting scripts, the planner, and the
training loader. See `motion_schema.md` for design motivation.
"""

from .reader import load_motion_data_parquet
from .schema import (
    ALL_FIELDS,
    COMMON_REQUIRED_FIELDS,
    CONTACT_FIELDS,
    DIAGNOSTICS_FIELDS,
    DUAL_HAND,
    DUAL_HAND_PER_SIDE_FIELDS,
    DUAL_HAND_REQUIRED_FIELDS,
    EE_FIELDS,
    HAND_FIELDS,
    KNOWN_MOTION_KINDS,
    METADATA_FIELDS,
    OBJECT_FIELDS,
    REQUIRED_TRAINING_FIELDS,
    ROBOT_FIELDS,
    SCHEMA_VERSION,
    SINGLE_ROBOT,
    SINGLE_ROBOT_REQUIRED_FIELDS,
    SOURCE_FIELDS,
    MissingRequiredField,
    MotionData,
    SchemaVersionMismatch,
    build_schema,
    required_fields_for,
    resolve_motion_kind,
)
from .writer import DEFAULT_PARTITION_COLS, save_motion_parquet

__all__ = [
    "ALL_FIELDS",
    "COMMON_REQUIRED_FIELDS",
    "CONTACT_FIELDS",
    "DEFAULT_PARTITION_COLS",
    "DIAGNOSTICS_FIELDS",
    "DUAL_HAND",
    "DUAL_HAND_PER_SIDE_FIELDS",
    "DUAL_HAND_REQUIRED_FIELDS",
    "EE_FIELDS",
    "HAND_FIELDS",
    "KNOWN_MOTION_KINDS",
    "METADATA_FIELDS",
    "MissingRequiredField",
    "MotionData",
    "OBJECT_FIELDS",
    "REQUIRED_TRAINING_FIELDS",
    "ROBOT_FIELDS",
    "SCHEMA_VERSION",
    "SINGLE_ROBOT",
    "SINGLE_ROBOT_REQUIRED_FIELDS",
    "SOURCE_FIELDS",
    "SchemaVersionMismatch",
    "build_schema",
    "load_motion_data_parquet",
    "required_fields_for",
    "resolve_motion_kind",
    "save_motion_parquet",
]
