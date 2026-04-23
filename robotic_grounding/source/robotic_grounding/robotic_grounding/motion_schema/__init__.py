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
    CONTACT_FIELDS,
    DIAGNOSTICS_FIELDS,
    EE_FIELDS,
    HAND_FIELDS,
    METADATA_FIELDS,
    OBJECT_FIELDS,
    REQUIRED_TRAINING_FIELDS,
    ROBOT_FIELDS,
    SCHEMA_VERSION,
    SOURCE_FIELDS,
    MissingRequiredField,
    MotionData,
    SchemaVersionMismatch,
    build_schema,
)
from .writer import DEFAULT_PARTITION_COLS, save_motion_parquet

__all__ = [
    "ALL_FIELDS",
    "CONTACT_FIELDS",
    "DEFAULT_PARTITION_COLS",
    "DIAGNOSTICS_FIELDS",
    "EE_FIELDS",
    "HAND_FIELDS",
    "METADATA_FIELDS",
    "MissingRequiredField",
    "MotionData",
    "OBJECT_FIELDS",
    "REQUIRED_TRAINING_FIELDS",
    "ROBOT_FIELDS",
    "SCHEMA_VERSION",
    "SOURCE_FIELDS",
    "SchemaVersionMismatch",
    "build_schema",
    "load_motion_data_parquet",
    "save_motion_parquet",
]
