from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inpainting.phantom_tracker.provenance import sha256_file, sha256_tree


class ProvenanceTests(unittest.TestCase):
    def test_tree_hash_is_path_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "a" / "x").write_bytes(b"one")
            first, entries = sha256_tree(root)
            self.assertEqual(entries[0]["sha256"], sha256_file(root / "a" / "x"))
            (root / "a" / "x").write_bytes(b"two")
            second, _ = sha256_tree(root)
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
