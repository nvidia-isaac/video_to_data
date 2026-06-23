# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import glob
from pathlib import Path


def resolve_glob(pattern: str) -> list[str]:
    """Expand a glob pattern to sorted paths, or wrap a literal path in a list."""
    if '*' in pattern or '?' in pattern:
        results = sorted(glob.glob(pattern))
        if not results:
            raise FileNotFoundError(f"Glob pattern matched no files: {pattern!r}")
        return results
    return [pattern]


def apply_output_pattern(pattern: str, *stems: str) -> str:
    """Replace each * in pattern with the corresponding stem, left to right."""
    result = pattern
    for stem in stems:
        result = result.replace('*', stem, 1)
    return result


def resolve_output(pattern: str, paths_with_sources: list[tuple[str, list[str]]]) -> str:
    """
    Resolve an output path pattern by substituting * with stems.
    Stems from varying sources (len > 1) are substituted before scalar sources,
    so a single * in the output always takes the varying input's stem.
    """
    varying = [Path(p).stem for p, src in paths_with_sources if len(src) > 1]
    scalar  = [Path(p).stem for p, src in paths_with_sources if len(src) == 1]
    return apply_output_pattern(pattern, *(varying + scalar))


def broadcast_pairs(
    a: list[str],
    b: list[str],
) -> list[tuple[str, str]]:
    """
    Broadcast two path lists.
      1:1  → single pair
      1:N  → replicate a across b
      N:1  → replicate b across a
      N:N  → zip
      N:M  → raises ValueError
    """
    if len(a) == 1:
        return list(zip(a * len(b), b))
    if len(b) == 1:
        return list(zip(a, b * len(a)))
    if len(a) != len(b):
        raise ValueError(
            f"broadcast_pairs requires equal lengths, got {len(a)} and {len(b)}"
        )
    return list(zip(a, b))


def broadcast_zip(*path_lists: list[str]) -> list[tuple[str, ...]]:
    """
    Zip-broadcast any number of path lists.
    Each list must be length 1 or the common length N.
    """
    lengths = {len(p) for p in path_lists}
    non_one = lengths - {1}
    if len(non_one) > 1:
        raise ValueError(
            f"broadcast_zip: incompatible lengths {sorted(lengths)}; all must be 1 or a common N"
        )
    n = next(iter(non_one)) if non_one else 1
    expanded = [p * n if len(p) == 1 else p for p in path_lists]
    return list(zip(*expanded))
