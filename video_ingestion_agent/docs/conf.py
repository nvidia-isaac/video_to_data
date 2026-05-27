# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys

# Read version from pyproject.toml so we have a single source of truth.
_pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
with open(_pyproject_path) as f:
    _match = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    VIDEO_INGESTION_AGENT_VERSION_NUMBER = _match.group(1) if _match else "0.0.0"

# Make helpers and custom extensions importable
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("_ext"))

# -- Project information -----------------------------------------------------

project = "Video Ingestion Agent"
copyright = "2025-2026, NVIDIA"
author = "NVIDIA"
version = VIDEO_INGESTION_AGENT_VERSION_NUMBER
release = VIDEO_INGESTION_AGENT_VERSION_NUMBER
released = True

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_multiversion",
    "sphinxcontrib.mermaid",
    "video_ingestion_agent_doc_tools",
]

todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "_templates",
    "_redirect",
    "Thumbs.db",
    ".DS_Store",
    "venv_docs",
]

# Be picky about missing references
nitpicky = True
nitpick_ignore: list[tuple[str, str]] = []

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_title = f"Video Ingestion Agent {VIDEO_INGESTION_AGENT_VERSION_NUMBER}"
html_show_sphinx = False
html_theme_options = {
    "copyright_override": {"start": 2025},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "github_url": "https://github.com/nvidia-isaac/video_to_data",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["svg_zoom.js"]

# Sidebar: include version dropdown
html_sidebars = {"**": ["versioning.html", "sidebar-nav-bs"]}

# -- Mermaid configuration ---------------------------------------------------

mermaid_version = "11"

# -- Multi-version configuration ---------------------------------------------

smv_remote_whitelist = r"^.*$"
smv_branch_whitelist = r"^(main|release/.*)$"
smv_tag_whitelist = r"^v.*$"

# -- Custom docs config (used by video_ingestion_agent_doc_tools extension) --------------------

video_ingestion_agent_docs_config = {
    "released": released,
    "internal_git_url": "https://github.com/nvidia-isaac/video_to_data.git",
    "external_git_url": "https://github.com/nvidia-isaac/video_to_data.git",
    "internal_code_link_base_url": "https://github.com/nvidia-isaac/video_to_data/blob/main/video_ingestion_agent",
    "external_code_link_base_url": "https://github.com/nvidia-isaac/video_to_data/blob/main/video_ingestion_agent",
}
