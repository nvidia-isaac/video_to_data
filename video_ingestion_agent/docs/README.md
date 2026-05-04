# Building the Documentation

This directory contains the Sphinx documentation for Video Ingestion Agent.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Quick Start

```bash
# From the repository root:

# 1. Install docs dependencies
uv pip install -r docs/requirements.txt

# 2. Build the HTML docs
cd docs
make html

# 3. Open in browser
open _build/current/html/index.html   # macOS
xdg-open _build/current/html/index.html   # Linux
```

## Available Make Targets

| Command | Description |
|---------|-------------|
| `make html` | Build HTML documentation into `_build/current/html/` |
| `make livehtml` | Start a live-reload dev server (auto-rebuilds on file changes) |
| `make linkcheck` | Check all external links for validity |
| `make multi-docs` | Build multi-version docs (for release publishing) |
| `make clean` | Remove all build artifacts |

## Live Preview (Recommended for Development)

```bash
cd docs
make livehtml
```

This starts `sphinx-autobuild` at `http://127.0.0.1:8000`. It watches for file
changes and auto-reloads the browser.

## Virtual Environment Note

If `sphinx-build` is installed in a virtual environment and not on your system
PATH, set the `SPHINXBUILD` variable:

```bash
SPHINXBUILD=/path/to/.venv/bin/sphinx-build make html
```

## Project Structure

```
docs/
├── conf.py                  # Sphinx configuration
├── index.rst                # Landing page
├── Makefile                 # Build commands
├── requirements.txt         # Python dependencies for docs
├── _ext/
│   └── video_ingestion_agent_doc_tools.py     # Custom Sphinx directives
├── _redirect/
│   └── index.html           # Redirect for multi-version builds
├── _templates/
│   └── versioning.html      # Sidebar version selector
└── pages/
    ├── getting_started.rst
    ├── deployment.rst
    ├── architecture.rst
    ├── ingestion_pipeline.rst
    ├── retrieval_agent.rst
    ├── database_design.rst
    ├── model_backends.rst
    ├── configuration.rst
    ├── prompts.rst
    ├── webapp.rst
    ├── benchmark.rst
    └── development.rst
```

## Adding a New Page

1. Create `docs/pages/your_page.rst`
2. Add it to a `toctree` in `docs/index.rst`
3. Optionally add a card in the landing page grid
4. Run `make html` to verify

## Custom Directives

The `_ext/video_ingestion_agent_doc_tools.py` extension provides two custom directives:

- `:git_clone_code_block:` — Renders a git clone command that switches
  between internal and external URLs based on the `released` flag in `conf.py`.
- `:code_link:\`<path>\`` — Renders a clickable link to a source file on
  GitHub, automatically using the filename as display text.
