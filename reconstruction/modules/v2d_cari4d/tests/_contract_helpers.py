# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _parameter_defaults(function: ast.FunctionDef) -> dict[str, object]:
    positional = function.args.args
    defaults = [None] * (len(positional) - len(function.args.defaults)) + function.args.defaults
    values = {argument.arg: ast.literal_eval(default) for argument, default in zip(positional, defaults) if default is not None}
    values.update({argument.arg: ast.literal_eval(default) for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults) if default is not None})
    return values
