from __future__ import annotations

import unittest
from types import SimpleNamespace

from inpainting.phantom_tracker.checkpoint import load_state_dict_strict


class _Model:
    def __init__(self, result: object | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.strict: bool | None = None

    def load_state_dict(self, state_dict: object, *, strict: bool) -> object:
        self.strict = strict
        if self.error is not None:
            raise self.error
        return self.result


class StrictCheckpointTests(unittest.TestCase):
    def test_load_uses_strict_true(self) -> None:
        model = _Model(SimpleNamespace(missing_keys=[], unexpected_keys=[]))
        report = load_state_dict_strict(model, {"weight": object()})
        self.assertIs(model.strict, True)
        self.assertEqual(report, {"missing_keys": [], "unexpected_keys": []})

    def test_any_reported_incompatible_key_is_rejected(self) -> None:
        for missing, unexpected in ((["missing"], []), ([], ["extra"])):
            with self.subTest(missing=missing, unexpected=unexpected):
                model = _Model(
                    SimpleNamespace(
                        missing_keys=missing, unexpected_keys=unexpected
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "incompatible keys"):
                    load_state_dict_strict(model, {"weight": object()})
                self.assertIs(model.strict, True)

    def test_backend_mismatch_is_rejected_with_context(self) -> None:
        model = _Model(error=RuntimeError("size mismatch"))
        with self.assertRaisesRegex(RuntimeError, "incompatible with the pinned model"):
            load_state_dict_strict(model, {"weight": object()})
        self.assertIs(model.strict, True)

    def test_state_dict_must_be_mapping(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a mapping"):
            load_state_dict_strict(_Model(), [("weight", object())])


if __name__ == "__main__":
    unittest.main()
