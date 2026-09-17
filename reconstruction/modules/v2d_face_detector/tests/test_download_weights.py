from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
sys.path.insert(0, str(LIB_DIR))

import download_weights


def test_download_is_atomic_and_checksum_verified(monkeypatch, tmp_path):
    payload = b"verified-yunet"
    monkeypatch.setattr(
        download_weights, "MODEL_SHA256", hashlib.sha256(payload).hexdigest()
    )

    def fake_urlretrieve(_url, destination):
        Path(destination).write_bytes(payload)

    monkeypatch.setattr(download_weights.urllib.request, "urlretrieve", fake_urlretrieve)
    destination = download_weights.download_weights(str(tmp_path))
    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_download_rejects_bad_checksum(monkeypatch, tmp_path):
    monkeypatch.setattr(download_weights, "MODEL_SHA256", "0" * 64)
    monkeypatch.setattr(
        download_weights.urllib.request,
        "urlretrieve",
        lambda _url, destination: Path(destination).write_bytes(b"wrong"),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_weights.download_weights(str(tmp_path))
    assert not (tmp_path / download_weights.MODEL_FILENAME).exists()
