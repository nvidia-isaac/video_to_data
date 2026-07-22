from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from inpainting.phantom_tracker import assets


class PinnedAssetTests(unittest.TestCase):
    def test_exact_hash_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "nested" / "weights.bin"
            path.parent.mkdir()
            path.write_bytes(b"official weights")
            expected = hashlib.sha256(b"official weights").hexdigest()
            assets.verify_pinned_files(
                root, {"nested/weights.bin": expected}, asset_name="test model"
            )

            path.write_bytes(b"tampered weights")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                assets.verify_pinned_files(
                    root, {"nested/weights.bin": expected}, asset_name="test model"
                )

    def test_missing_pinned_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing pinned"):
                assets.verify_pinned_files(
                    directory,
                    {"missing.bin": "0" * 64},
                    asset_name="test model",
                )

    def test_combined_guard_checks_every_dino_and_hamer_pin(self) -> None:
        with patch.object(assets, "verify_pinned_files") as verify:
            assets.verify_pinned_inference_assets("/dino", "/hamer")
        self.assertEqual(
            verify.call_args_list,
            [
                call(
                    "/dino",
                    assets.GROUNDING_DINO_REQUIRED_SHA256,
                    asset_name="Grounding DINO",
                ),
                call(
                    "/hamer",
                    assets.HAMER_REQUIRED_SHA256,
                    asset_name="HaMeR",
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
