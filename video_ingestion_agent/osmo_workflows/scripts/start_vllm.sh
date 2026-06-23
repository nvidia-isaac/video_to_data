#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared vLLM server launcher for osmo workflows.
#
# Environment variables:
#   VLLM_CONFIG  - Pipeline config YAML for serve.py (default: configs/ingestion.yaml)
#   HF_TOKEN     - Hugging Face token (must be set by caller)
#   NIM_API_KEY  - NVIDIA NIM API key (optional, used by some backends)
#
# serve.py checks vlm_backend in the config and exits cleanly
# when the backend is not "vllm" (e.g. "api").

set -ex

cd /workspace/video_ingestion_agent

VLLM_CONFIG="${VLLM_CONFIG:-configs/ingestion.yaml}"

python3 scripts/serve.py -c "${VLLM_CONFIG}" &
SERVE_PID=$!

# Wait for server to be ready (model download + load can take 5-10 min)
echo "Waiting for vLLM server to become healthy..."
VLLM_READY=false
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        VLLM_READY=true
        break
    fi
    # If serve.py already exited, stop waiting
    if ! kill -0 $SERVE_PID 2>/dev/null; then
        break
    fi
    sleep 5
done

if [ "$VLLM_READY" = false ]; then
    # Check if serve.py skipped (backend != vllm) vs actual failure
    if wait $SERVE_PID; then
        echo "vLLM server not needed (backend is not 'vllm') -- continuing without it."
    else
        echo "ERROR: vLLM server failed to start"
        exit 1
    fi
fi
