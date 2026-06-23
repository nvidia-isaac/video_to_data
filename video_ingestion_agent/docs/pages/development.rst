Development Guide
=================

This guide covers setting up a development environment, running tests, and contributing to
Video Ingestion Agent.

Setup
-----

:git_clone_code_block:

.. code-block:: bash

   cd video_ingestion_agent

   # Create virtual environment
   uv venv .venv
   source .venv/bin/activate

   # Install all dependencies including dev tools
   uv pip install -e ".[all]"

   # Install pre-commit hooks
   pre-commit install

Testing
-------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=video_ingestion_agent --cov-report=term-missing

   # Run a specific test file
   pytest tests/test_ingestion.py

   # Run a specific test
   pytest tests/test_ingestion.py::test_config_loading -v

Test Suite
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Test File
     - What it covers
   * - ``tests/test_ingestion.py``
     - Config loading, type conversions, graph creation, pipeline imports
   * - ``tests/test_common_utils.py``
     - JSON parsing, timestamp parsing, video utils, model tests
   * - ``tests/test_adapter.py``
     - EPIC-KITCHENS verb/noun adapter mapping
   * - ``tests/test_dedup.py``
     - Clip deduplication (heuristic and LLM-based strategies)
   * - ``tests/test_visual_extractor.py``
     - Visual feature extraction (requires GPU)
   * - ``tests/test_retrieval_config.py``
     - Retrieval agent config loading and validation
   * - ``tests/test_retrieval_parallel.py``
     - Parallel sub-task execution via LangGraph Send API

Code Quality
------------

Linting
^^^^^^^

.. code-block:: bash

   # Check for lint errors
   ruff check .

   # Auto-fix lint errors
   ruff check . --fix

   # Format code
   ruff format .

Type Checking
^^^^^^^^^^^^^

.. code-block:: bash

   mypy src/video_ingestion_agent

Pre-commit Hooks
^^^^^^^^^^^^^^^^^

Pre-commit hooks run automatically on ``git commit``. To run manually:

.. code-block:: bash

   pre-commit run --all-files

Project Layout
--------------

.. code-block:: text

   src/video_ingestion_agent/
   ├── ingestion/              # Ingestion pipeline
   │   ├── ingestion_graph.py  # LangGraph pipeline graph
   │   ├── config.py           # Pydantic config models
   │   ├── state.py            # PipelineState TypedDict
   │   ├── segmentation/       # Segmenter, critic, refiner
   │   └── entity_graph/       # Extractors, linker, DB writer
   ├── retrieval/              # Retrieval agent
   │   ├── retrieval_graph.py  # Agent graph
   │   ├── nodes/              # Agent decision nodes
   │   └── tools/              # Search tools
   ├── models/                 # Model backend abstraction
   ├── benchmark/              # EPIC-KITCHENS evaluation
   ├── webapp/                 # Gradio web interface
   └── utils/                  # Shared utilities

Key Patterns
^^^^^^^^^^^^

**LangGraph State Graphs**: Both the ingestion pipeline and retrieval agent use LangGraph.
Each node is a pure function that takes the current state and returns updates:

.. code-block:: python

   def my_node(state: PipelineState) -> dict:
       # Read from state
       clips = state["clips"]
       # Process
       result = process(clips)
       # Return state updates (merged into state)
       return {"processed_clips": result}

**Pydantic Configuration**: All config is validated via Pydantic models defined in
``ingestion/config.py``:

.. code-block:: python

   class PipelineConfig(BaseModel):
       models: ModelConfig
       segmentation: SegmentationConfig
       verification: VerificationConfig
       ...

**Model Backend Abstraction**: The ``ModelManager`` in ``models/model_manager.py`` provides a
unified interface. All backends implement ``generate_text()`` and ``generate_from_video()``.

Building Documentation
-----------------------

.. code-block:: bash

   # Install docs dependencies
   uv pip install -e ".[docs]"

   # Build HTML docs
   cd docs
   make html

   # View the docs
   # Open docs/_build/current/html/index.html in a browser

Continuous Integration
-----------------------

The project uses:

- **pytest** for testing with coverage reporting
- **ruff** for linting and formatting (replaces flake8, isort, black)
- **mypy** for type checking
- **pre-commit** for automated code quality checks

Ruff Configuration
^^^^^^^^^^^^^^^^^^^

Ruff is configured in ``pyproject.toml``:

.. code-block:: toml

   [tool.ruff]
   line-length = 100
   target-version = "py310"

   [tool.ruff.lint]
   select = ["E", "W", "F", "I", "B", "C4", "UP"]

Selected rule sets:

- ``E/W`` — pycodestyle errors/warnings
- ``F`` — pyflakes
- ``I`` — import sorting (isort)
- ``B`` — flake8-bugbear
- ``C4`` — flake8-comprehensions
- ``UP`` — pyupgrade

Contributing
------------

We welcome contributions. Here's how to get started:

1. **Open an issue** on `GitHub <https://github.com/nvidia-isaac/video_to_data/issues>`_ to
   discuss the change before writing code.
2. **Fork and branch** — create a feature branch from ``main``:

   .. code-block:: bash

      git checkout -b feature/my-change

3. **Write tests** — add or update tests in ``tests/`` for any new functionality.
4. **Run checks** before committing:

   .. code-block:: bash

      ruff check . && ruff format --check . && pytest

5. **Open a pull request** against ``main`` with a clear description of the change.

Branch Naming
^^^^^^^^^^^^^

- ``feature/<description>`` — new functionality
- ``fix/<description>`` — bug fixes
- ``docs/<description>`` — documentation changes
- ``refactor/<description>`` — code restructuring

See Also
--------

- :doc:`/pages/architecture` — Understand the system design before contributing
- :doc:`/pages/troubleshooting` — Common issues and solutions
- `GitHub Issues <https://github.com/nvidia-isaac/video_to_data/issues>`_ — Report bugs or
  request features
