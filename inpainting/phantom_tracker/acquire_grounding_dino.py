"""Download and freeze Grounding DINO in a dedicated network-enabled step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from . import GROUNDING_DINO_MODEL, GROUNDING_DINO_REVISION
from .provenance import sha256_tree


def acquire(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        GROUNDING_DINO_MODEL, revision=GROUNDING_DINO_REVISION
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL, revision=GROUNDING_DINO_REVISION
    )
    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tree_hash, entries = sha256_tree(output_dir)
    record = {
        "model_id": GROUNDING_DINO_MODEL,
        "revision": GROUNDING_DINO_REVISION,
        "tree_sha256": tree_hash,
        "files": entries,
    }
    temporary = output_dir / "acquisition.partial.json"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "acquisition.json")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
