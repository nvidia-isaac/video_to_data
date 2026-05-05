# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Backward-compatible alias.

`NvhumanG1Data` is retired in favor of the unified `MotionData` schema. The
real integration test now lives in
`tests/test_motion_schema_parquet_integration.py`. Keep this file as an alias
so any CI or IDE task referencing it still resolves.
"""

from tests.test_motion_schema_parquet_integration import (  # noqa: F401
    _run_as_script,
)

if __name__ == "__main__":
    raise SystemExit(_run_as_script())
