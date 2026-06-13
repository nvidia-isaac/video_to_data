#!/usr/bin/env python3
"""Export BundleSDF textured OBJ output as a self-contained GLB."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


def export_textured_obj_to_glb(input_obj: str | Path, output_glb: str | Path) -> None:
    """Convert a BundleSDF OBJ/MTL/texture set into one GLB file.

    BundleSDF writes `textured_mesh.obj` plus `material.mtl` and a texture atlas
    next to it. Trimesh resolves those relative references and embeds the texture
    in the exported GLB without changing mesh geometry.
    """
    input_obj = Path(input_obj)
    output_glb = Path(output_glb)

    if not input_obj.exists():
        raise FileNotFoundError(f"Input OBJ not found: {input_obj}")
    if input_obj.suffix.lower() != ".obj":
        raise ValueError(f"Expected OBJ input, got: {input_obj}")
    if output_glb.suffix.lower() != ".glb":
        raise ValueError(f"Expected GLB output, got: {output_glb}")

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(str(input_obj), force="scene", process=False)
    scene.export(str(output_glb))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input textured OBJ path")
    parser.add_argument("--output", required=True, help="Output GLB path")
    args = parser.parse_args()

    export_textured_obj_to_glb(args.input, args.output)
    print(f"Exported GLB: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
