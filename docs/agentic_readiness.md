# Agentic Readiness Guide

This repo is usable with coding agents, but it spans three very different
runtime surfaces: a Python/LangGraph agent, Dockerized reconstruction modules,
and Isaac Lab robotic grounding. The best results come from giving agents a
bounded task, a package target, and an explicit validation budget.

## Readiness Snapshot

Strengths:

- Clear package split between ingestion, reconstruction, and robotic grounding.
- Existing READMEs explain the main user workflows.
- `video_ingestion_agent/` has a conventional Python project layout, tests,
  docs, and CI.
- Reconstruction modules follow a repeatable host-wrapper/container-lib pattern.
- Claude-specific onboarding and diagnostic skills already capture practical
  first-run knowledge for the ingestion agent.

Gaps this guide helps close:

- Agent instructions were scattered across package docs and Claude-specific
  files.
- Some first-run commands were easy for agents to mis-copy.
- Heavy GPU, Docker, private registry, and model-weight steps need clearer
  human approval boundaries.

## Good Tasks for Agents

- Explain package architecture and trace a workflow.
- Add or improve docs, examples, and troubleshooting notes.
- Make small Python changes in `video_ingestion_agent/` with unit tests.
- Update reconstruction wrapper arguments when the matching lib signature
  changes.
- Add narrow tests or synthetic fixtures.
- Review PRs for missed validation, path drift, and unsafe credential handling.

## Tasks That Need Human Confirmation

Ask the user before asking an agent to:

- Download large model weights or datasets.
- Build all reconstruction containers.
- Start multi-GPU vLLM servers.
- Run Isaac Lab training, evaluation, or long simulations.
- Access NGC, CSS/PDX, HuggingFace gated models, or other private services.
- Modify tracked data assets, large binary files, or generated outputs.

## Prompt Template

```text
Please work in <package>.

Goal:
<one or two sentences>

Constraints:
- Keep the change scoped to <files or subsystem>.
- Do not run GPU/Docker/model-download steps unless needed.
- Preserve existing package conventions from AGENTS.md.

Validation budget:
Run <lightweight checks>. If full validation needs GPU/Docker/private access,
explain what should be run later.

Deliverable:
Open a PR with a summary, validation, and remaining risks.
```

## Package Validation Cheatsheet

| Package | Lightweight validation | Full validation caveat |
| --- | --- | --- |
| Root docs | Check links and referenced paths | None |
| `video_ingestion_agent/` | `pre-commit`, `mypy`, `pytest tests/ -o addopts=""`, Sphinx docs | Full test env installs heavier extras and needs `ffmpeg` |
| `reconstruction/` | Verify script/module paths and targeted wrapper imports | Docker builds, model downloads, and smoke tests need GPU |
| `robotic_grounding/` | `pre-commit` scoped to tracked package files | Full CI needs Docker, GPU, Isaac Lab, and registry access |

For detailed instructions used by coding agents, see the repo-level
[`AGENTS.md`](../AGENTS.md).
