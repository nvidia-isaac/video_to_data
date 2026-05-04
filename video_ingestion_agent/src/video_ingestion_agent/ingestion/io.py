# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
I/O utilities for reading and writing JSONL files, with Pydantic model support.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def read_jsonl(file_path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Read JSONL file line by line.

    Args:
        file_path: Path to the JSONL file

    Yields:
        Dictionary for each line in the file

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num}: {e.msg}", e.doc, e.pos
                ) from e


def write_jsonl(
    data: list[dict[str, Any]] | list[BaseModel],
    file_path: str | Path,
    append: bool = False,
) -> None:
    """
    Write data to a JSONL file.

    Args:
        data: List of dictionaries or Pydantic models to write
        file_path: Path to the output JSONL file
        append: If True, append to existing file; otherwise overwrite
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"

    with open(file_path, mode, encoding="utf-8") as f:
        for item in data:
            if isinstance(item, BaseModel):
                json_str = item.model_dump_json()
            else:
                json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + "\n")


def read_jsonl_as_models(
    file_path: str | Path, model_class: type[BaseModel]
) -> Iterator[BaseModel]:
    """
    Read JSONL file and parse each line as a Pydantic model.

    Args:
        file_path: Path to the JSONL file
        model_class: Pydantic model class to parse into

    Yields:
        Pydantic model instance for each line
    """
    for data in read_jsonl(file_path):
        yield model_class(**data)


def write_models_jsonl(
    models: list[BaseModel], file_path: str | Path, append: bool = False
) -> None:
    """
    Write Pydantic models to a JSONL file.

    Args:
        models: List of Pydantic model instances
        file_path: Path to the output JSONL file
        append: If True, append to existing file; otherwise overwrite
    """
    write_jsonl(models, file_path, append=append)
