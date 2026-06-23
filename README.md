# Video to Data

Monorepo for **Video to Data (V2D)** — an end-to-end pipeline that converts human demonstration videos into simulation-ready assets and physics-grounded robot training data.

## End-to-End Workflow

```
 ┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ Human demo    │ → │ 1. Video Ingestion   │ → │ 2. Reconstruction    │
 │ video / rosbag│   │    Agent             │   │ depth · masks ·      │
 │               │   │ action segments ·    │   │ meshes · 6D pose ·   │
 │               │   │ entity graph ·       │   │ SMPL body            │
 │               │   │ visual embeddings    │   │                      │
 └───────────────┘   └──────────────────────┘   └──────────────────────┘
                       video_ingestion_agent/      reconstruction/
```

1. **Video Ingestion Agent** — a LangGraph-driven agentic workflow that segments demonstration videos into temporally-bounded action clips, extracts an entity-relation scene graph, and stores per-frame SigLIP-2 embeddings. The result is a queryable action database (`graph.db` + `vector.db`) that lets downstream stages select which clips to process via natural-language retrieval, instead of brute-forcing the full video.
2. **Reconstruction** — containerized vision modules turn the selected RGB (or stereo) clips into per-frame depth, object masks, textured meshes, 6-DoF object poses, and SMPL human body parameters. Multi-view pipelines (`run_mv_hoi_reconstruction`, `run_mv_calibration`) orchestrate the full reconstruction from a rosbag.

## Packages

| Package | Role | Runtime |
|---|---|---|
| [`video_ingestion_agent/`](video_ingestion_agent/) | Video → action segments + entity scene graph + frame embeddings. LangGraph pipeline (segment → verify/refine → entity graph → embeddings) plus an EGAgent-style natural-language retrieval agent and an optional Gradio UI. | Python venv + vLLM server |
| [`reconstruction/`](reconstruction/) | Video → depth, masks, meshes, 6D poses, human body. 18 containerized modules + multi-view pipelines. | Docker (per-module images) |

## Prerequisites

- Docker with GPU support ([install](https://docs.docker.com/engine/install/ubuntu/))
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Python 3.10+

## Quickstart

### Video Ingestion Agent (video → queryable action database)

```bash
cd video_ingestion_agent

uv venv .venv && source .venv/bin/activate
uv pip install -e ".[all]"     # vLLM, webapp, benchmark, dev tools

# 1. Start the vLLM server (loads the VLM, ~1 minute)
python scripts/serve.py -c configs/ingestion.yaml

# 2. Ingest a video — segmentation → entity graph → report
python scripts/run_ingestion.py path/to/video.mp4 \
  -c configs/ingestion.yaml --no-verify -o runs/my_run

# 3. Retrieve clips with natural language
python scripts/run_retrieval.py "Find clips where someone picks up a mug" \
  -d outputs/ -c configs/retrieval.yaml

# 4. Or browse interactively in the web UI
python scripts/run_webapp.py
```

See [video_ingestion_agent/README.md](video_ingestion_agent/README.md) for hardware requirements, the full extras list, the verify/refine loop, and batch-ingestion across multiple GPUs. Pre-publication TODOs are tracked in [video_ingestion_agent/docs/release_readiness.md](video_ingestion_agent/docs/release_readiness.md).

### Reconstruction

We include a variety of algorithms and pipelines that are useful for human-object reconstruction. These packages are contained in the [reconstruction](reconstruction) subfolder.  

For the initial release, we provide an example pipeline for ego-centric hand-object reconstruction.

To get started, please follow the following the instructions [here](reconstruction/docs/ego_e2e_setup.md).

> Please note:  The reconstruction subfolder contains a wide variety of packages, many of which are partially tested or in development.  You may find these packages useful, but please note they are subject to change.  The above ego-centric pipeline has been tested, and is officially included as part of the initial video to data release.  If there is a package you would like to see supported, or you have any feedback, please let us know by opening an issue on GitHub!

## Design Philosophy

- **Host orchestration, containerized inference.** The host runs thin Python wrappers that `docker run` each module; all ML dependencies live inside their respective images. No CUDA or PyTorch is ever installed on the host.
- **Typed contracts between packages.** Modules communicate through strongly-typed dataclasses in [`v2d_common`](reconstruction/modules/v2d_common/) (`DepthImage`, `CameraIntrinsics`, `Transform3d`, `BoundingBox`, `Mask`) — never raw arrays across package boundaries.
- **File-based dataflow.** Modules write intermediate artifacts to disk (depth PNGs, pose JSONs, mask PNGs, etc.), enabling independent execution, caching, and pipeline composition via [`v2d_pipelines`](reconstruction/modules/v2d_pipelines/).

## Contributing

We welcome contributions. All contributions require a Developer Certificate of Origin (DCO) sign-off: commit with `git commit -s` so your commit carries a `Signed-off-by` trailer. Contributions without a sign-off cannot be accepted. See the per-package contributing guides for the full DCO text and workflow:

- Reconstruction: [reconstruction/CONTRIBUTING.md](reconstruction/CONTRIBUTING.md) (and module conventions in [reconstruction/README.md](reconstruction/README.md#contributing))
- Video Ingestion Agent: [video_ingestion_agent/CONTRIBUTING.md](video_ingestion_agent/CONTRIBUTING.md)

Each new reconstruction module must expose a Docker image, a `run_download_weights` entry point (if weights are required), a `run_shell` entry point, and a typed API surface consistent with `v2d_common`.

## License

This project is dual-licensed: source code under **Apache-2.0** and documentation/skill files under **CC-BY-4.0**, per the top-level [`LICENSE`](LICENSE). NVIDIA-authored code files carry an `SPDX-License-Identifier: Apache-2.0` header; mixed documentation/skill content is covered by the CC-BY-4.0 AND Apache-2.0 dual terms from the top-level `LICENSE` and does not carry a per-file header.

Some bundled third-party components are covered by their own licenses, **not** this project's terms, and retain their own `LICENSE` files in-tree:

- `reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/` — **NVIDIA Source Code License** (non-commercial: research/evaluation only).
- `reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/` — Meta **SAM License** (commercial use permitted, with acceptable-use restrictions).

Third-party dependency attributions are listed in [`reconstruction/NOTICE`](reconstruction/NOTICE) and [`video_ingestion_agent/NOTICE`](video_ingestion_agent/NOTICE).
