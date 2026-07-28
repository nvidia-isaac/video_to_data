"""Immutable model-file pins and pre-inference integrity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .provenance import sha256_file


GROUNDING_DINO_REQUIRED_SHA256 = {
    "config.json": "491ed3c75ef0ccfb4706ff0592f4f7312cd3cf19952e7b545e9abed68a7357db",
    "model.safetensors": "4f29e24728239929bfc149fb5fe238b3707fddd6048d87d8eaabb38181c79a8b",
    "preprocessor_config.json": "c17fd68afb1f124bfb87a494d409925eec4201b3487c8a694dde64d9ce7109a3",
    "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}

HAMER_REQUIRED_SHA256 = {
    "_DATA/data/mano_mean_params.npz": "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988",
    "_DATA/hamer_ckpts/model_config.yaml": "0e5eeb82752e47dfd01db8e13ccc4c5eba9bf83f53da8285523b8d3e87247aa3",
    "_DATA/hamer_ckpts/dataset_config.yaml": "392d10aece296152769328f14bb1c0137d6ea2f50f7e69f716ee00e9ad6ac343",
    "_DATA/hamer_ckpts/checkpoints/hamer.ckpt": "e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53",
}


def verify_pinned_files(
    root: str | Path,
    expected_sha256: Mapping[str, str],
    *,
    asset_name: str,
) -> None:
    """Require every pinned regular file to match its exact SHA-256."""

    directory = Path(root)
    for relative, expected in expected_sha256.items():
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned {asset_name} file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Pinned {asset_name} SHA-256 mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )


def verify_pinned_inference_assets(
    grounding_dino_dir: str | Path, hamer_dir: str | Path
) -> None:
    """Verify all immutable model inputs before either network is executed."""

    verify_pinned_files(
        grounding_dino_dir,
        GROUNDING_DINO_REQUIRED_SHA256,
        asset_name="Grounding DINO",
    )
    verify_pinned_files(
        hamer_dir,
        HAMER_REQUIRED_SHA256,
        asset_name="HaMeR",
    )
