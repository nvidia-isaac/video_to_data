# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

from sphinx.application import Sphinx


def git_clone_code_block(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces :git_clone_code_block: with a git clone code block.

    The output URL depends on whether we are in release or internal mode.
    """

    def replacer(_: Any) -> str:
        release_state = app.config.video_ingestion_agent_docs_config["released"]
        internal_git_url = app.config.video_ingestion_agent_docs_config["internal_git_url"]
        external_git_url = app.config.video_ingestion_agent_docs_config["external_git_url"]
        if release_state:
            git_clone_target = external_git_url
        else:
            git_clone_target = internal_git_url
        return f"""
.. code-block:: bash

    git clone {git_clone_target}

"""

    source[0] = re.sub(r":git_clone_code_block:", replacer, source[0])


def code_link(app: Sphinx, _: Any, source: list[str]) -> None:
    """Replaces :code_link:`<relative/path>` with a link to the code."""

    def replacer(match: re.Match) -> str:
        relative_path = match.group("relative_path")
        release_state = app.config.video_ingestion_agent_docs_config["released"]
        internal_url = app.config.video_ingestion_agent_docs_config["internal_code_link_base_url"]
        external_url = app.config.video_ingestion_agent_docs_config["external_code_link_base_url"]
        file_name = relative_path.split("/")[-1]
        if release_state:
            base_url = external_url
        else:
            base_url = internal_url
        return f"`{file_name} <{base_url}/blob/main/{relative_path}>`__"

    source[0] = re.sub(r":code_link:`<(?P<relative_path>.*)>`", replacer, source[0])


def setup(app: Sphinx) -> None:
    app.connect("source-read", git_clone_code_block)
    app.connect("source-read", code_link)
    app.add_config_value("video_ingestion_agent_docs_config", {}, "env")
