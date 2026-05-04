# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application configuration for the Gradio webapp."""

import logging
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import yaml

logger = logging.getLogger(__name__)

NVIDIA_GREEN = gr.themes.Color(
    c50="#f0fbe0",
    c100="#daf5b1",
    c200="#c2ed7e",
    c300="#a8e449",
    c400="#93db1d",
    c500="#76B900",
    c600="#6aa800",
    c700="#5a9200",
    c800="#4a7c00",
    c900="#3a6200",
    c950="#2a4800",
)


@dataclass
class AppConfig:
    """Configuration for the Gradio webapp."""

    # Paths
    data_dir: str = "outputs/webapp"
    default_output_dir: str = "outputs/"
    default_db_dir: str = "/mnt/amlfs-02/shared/liuw/v2p/database"
    default_videos_dir: str = ""
    default_clips_dir: str = "outputs/clips"
    config_dir: str = "configs"

    # Default configs
    default_ingestion_config: str = "configs/ingestion.yaml"
    default_retrieval_config: str = "configs/retrieval.yaml"

    # UI settings
    max_clips_display: int = 20
    video_thumbnail_size: tuple = (320, 240)

    # Processing
    enable_gpu: bool = True
    default_device: str = "cuda"
    max_concurrent_ingestions: int = 1

    # Model settings (from retrieval.yaml)
    llm_model: str = "openai/openai/gpt-5.2"
    llm_backend: str = "api"
    embedding_model: str = "google/siglip2-base-patch16-256"
    api_key: str | None = None

    # vLLM backend settings (only used when llm_backend == "vllm")
    vllm_url: str = "http://localhost:8000/v1"
    vllm_local_media: bool = True
    vllm_tp_size: int = 1
    vllm_gpu_memory_utilization: float = 0.8

    # Custom CSS for styling
    custom_css: str = """
        /* ── Monospace log output ── */
        .log-output textarea {
            font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace !important;
            font-size: 13px !important;
            line-height: 1.45 !important;
        }

        /* ── App header ── */
        .app-header {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 16px 24px 12px;
            border-bottom: 2px solid #76B900;
        }
        .app-header .nvidia-wordmark {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #76B900;
        }
        .app-header .app-title {
            font-size: 20px;
            font-weight: 600;
        }

        /* ── Retrieve tab layout ── */
        .rt-shell { padding: 0 8px 16px; }
        .rt-spacer { flex: 1; }

        /* ── Search bar (Gemini-style) ── */
        .rt-searchbar {
            border: 1px solid var(--border-color-primary);
            border-radius: 16px;
            padding: 14px 16px 8px;
            margin-bottom: 14px;
            gap: 0 !important;
        }
        .rt-query-input textarea {
            font-size: 15px !important;
            border: none !important;
            box-shadow: none !important;
            padding: 4px 0 !important;
        }
        .rt-toolbar {
            align-items: center;
            gap: 6px;
            border-top: 1px solid var(--border-color-primary);
            padding-top: 8px;
            margin-top: 4px;
        }
        .rt-db-label {
            cursor: default !important;
            opacity: 0.8;
            pointer-events: none;
            font-weight: 600 !important;
            font-size: 13px !important;
        }
        .rt-db-dropdown { max-width: 320px; }
        .rt-refresh-btn { min-width: 36px !important; }
        .rt-search-btn {
            min-height: 38px !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }

        /* ── Pipeline horizontal bar ── */
        .rt-pipeline-wrap {
            border: 1px solid var(--border-color-primary);
            border-radius: 12px;
            padding: 18px 24px 14px;
            margin-bottom: 16px;
        }
        .pipeline-bar {
            display: flex;
            align-items: flex-start;
            justify-content: center;
            width: 100%;
        }
        .pb-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 100px;
            flex-shrink: 0;
        }
        .pb-dot {
            width: 18px; height: 18px;
            border-radius: 50%;
            background: var(--border-color-primary);
            border: 2px solid var(--border-color-primary);
            margin-bottom: 8px;
            transition: all .3s;
        }
        .pb-node.complete .pb-dot {
            background: #76B900;
            border-color: #76B900;
            box-shadow: 0 0 8px rgba(118,185,0,0.4);
        }
        .pb-node.running .pb-dot {
            background: #76B900;
            border-color: #a8e449;
            animation: pb-pulse 1.2s ease-in-out infinite;
        }
        .pb-node.error .pb-dot {
            background: #e53935;
            border-color: #e53935;
        }
        .pb-label {
            font-size: 12px;
            color: var(--body-text-color-subdued);
            text-align: center;
            white-space: nowrap;
        }
        .pb-node.complete .pb-label,
        .pb-node.running .pb-label {
            color: var(--body-text-color);
            font-weight: 600;
        }
        .pb-line {
            flex: 1;
            height: 3px;
            background: var(--border-color-primary);
            margin-top: 9px;
            min-width: 40px;
            border-radius: 2px;
            transition: background .3s;
        }
        .pb-line.complete { background: #76B900; }
        .pb-message {
            text-align: center;
            font-size: 13px;
            color: var(--body-text-color-subdued);
            margin-top: 6px;
        }
        @keyframes pb-pulse {
            0%, 100% { box-shadow: 0 0 6px rgba(118,185,0,0.4); }
            50%      { box-shadow: 0 0 16px rgba(118,185,0,0.7); }
        }

        /* ── Results ── */
        .rt-results-heading h3 { margin: 0 0 6px !important; }
        .rt-answer { margin-bottom: 10px; font-size: 14px; }

        /* ── Clip card grid ── */
        .clips-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        @media (max-width: 900px) {
            .clips-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .clip-card {
            background: var(--background-fill-secondary);
            border: 1px solid var(--border-color-primary);
            border-radius: 12px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            transition: border-color .2s, box-shadow .2s;
        }
        .clip-card:hover {
            border-color: #76B900;
            box-shadow: 0 0 10px rgba(118,185,0,0.15);
        }
        .clip-thumb {
            width: 100%;
            aspect-ratio: 16/10;
            background: var(--background-fill-primary);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .clip-thumb svg { opacity: 0.4; }
        .clip-video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 8px;
        }
        .clip-info { display: flex; flex-direction: column; gap: 3px; }
        .clip-id { font-weight: 700; font-size: 14px; }
        .clip-meta { font-size: 12px; color: #5a9200; font-weight: 600; }
        .clip-desc {
            font-size: 12px;
            color: var(--body-text-color-subdued);
            line-height: 1.4;
        }
        .clips-empty {
            text-align: center;
            padding: 40px 20px;
            color: var(--body-text-color-subdued);
            font-size: 14px;
            font-style: italic;
        }

        /* ── Working Memory accordion ── */
        .rt-wm-accordion {
            margin-top: 12px;
            border-radius: 10px !important;
        }
    """

    dark_mode_head: str = """
    <script>
    (() => {
        // Set URL parameter so Gradio picks up dark theme
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
        // Pre-apply dark class before Gradio initializes
        document.documentElement.classList.add('dark');
    })();
    </script>
    """

    @staticmethod
    def build_theme() -> gr.themes.Base:
        """Build a Gradio theme with NVIDIA brand colors."""
        return gr.themes.Soft(
            primary_hue=NVIDIA_GREEN,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        )

    @property
    def history_db(self) -> str:
        """Path to query history database."""
        return str(Path(self.data_dir) / "history.db")

    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        """Load config from YAML file.

        First loads defaults from project config files (retrieval.yaml,
        ingestion.yaml) so that model settings are always picked up,
        then overlays any explicit overrides from *path*.
        """
        config = cls.from_project_configs()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def from_project_configs(cls, config_dir: str = "configs") -> "AppConfig":
        """Load config with defaults from project config files.

        Reads retrieval.yaml and ingestion.yaml to extract
        default paths for database directory, clips directory, etc.
        """
        config = cls(config_dir=config_dir)

        # Try to load agent config for database/output settings
        retrieval_config_path = Path(config_dir) / "retrieval.yaml"
        if retrieval_config_path.exists():
            try:
                with open(retrieval_config_path) as f:
                    retrieval_cfg = yaml.safe_load(f) or {}

                # Extract database directory
                if "database" in retrieval_cfg and "directory" in retrieval_cfg["database"]:
                    config.default_db_dir = retrieval_cfg["database"]["directory"]
                    logger.info(f"Using database directory from config: {config.default_db_dir}")

                # Extract clips directory
                if "output" in retrieval_cfg and "clips_dir" in retrieval_cfg["output"]:
                    config.default_clips_dir = retrieval_cfg["output"]["clips_dir"]

                # Extract model settings
                models_cfg = retrieval_cfg.get("models", {})
                if "device" in models_cfg:
                    config.default_device = models_cfg["device"]
                    config.enable_gpu = config.default_device == "cuda"
                if "llm_model" in models_cfg:
                    config.llm_model = models_cfg["llm_model"]
                    logger.info(f"Using LLM model from config: {config.llm_model}")
                if "llm_backend" in models_cfg:
                    config.llm_backend = models_cfg["llm_backend"]
                    logger.info(f"Using LLM backend from config: {config.llm_backend}")
                if "embedding_model" in models_cfg:
                    config.embedding_model = models_cfg["embedding_model"]
                if "api_key" in models_cfg and models_cfg["api_key"]:
                    config.api_key = models_cfg["api_key"]

                # vLLM-specific settings
                if "vllm_url" in models_cfg:
                    config.vllm_url = models_cfg["vllm_url"]
                if "vllm_local_media" in models_cfg:
                    config.vllm_local_media = models_cfg["vllm_local_media"]
                if "vllm_tp_size" in models_cfg:
                    config.vllm_tp_size = models_cfg["vllm_tp_size"]
                if "vllm_gpu_memory_utilization" in models_cfg:
                    config.vllm_gpu_memory_utilization = models_cfg["vllm_gpu_memory_utilization"]

            except Exception as e:
                logger.warning(f"Failed to load agent config: {e}")

        # Try to load ingestion config for database/output settings
        ingestion_config_path = Path(config_dir) / "ingestion.yaml"
        if ingestion_config_path.exists():
            try:
                with open(ingestion_config_path) as f:
                    ingestion_cfg = yaml.safe_load(f) or {}

                # Extract output directory for ingestion
                if "database" in ingestion_cfg and "directory" in ingestion_cfg["database"]:
                    config.default_output_dir = ingestion_cfg["database"]["directory"]

            except Exception as e:
                logger.warning(f"Failed to load ingestion config: {e}")

        return config

    # Directories to scan for databases (relative or absolute)
    db_scan_dirs: tuple = ("/mnt/amlfs-02/shared/liuw/v2p/database",)

    def ensure_dirs(self):
        """Create necessary directories."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.default_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.default_clips_dir).mkdir(parents=True, exist_ok=True)

    def get_config_files(self) -> list[str]:
        """List available config files."""
        config_path = Path(self.config_dir)
        if not config_path.exists():
            return []
        files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
        return [str(f) for f in sorted(files)]

    def discover_databases(self) -> list[str]:
        """Scan configured directories for subdirectories containing graph.db.

        Returns a sorted list of directory paths (as strings) that contain
        a ``graph.db`` file.  Both top-level directories and one level of
        subdirectories are checked.
        """
        found: set[str] = set()

        for scan_root in self.db_scan_dirs:
            root = Path(scan_root)
            if not root.is_dir():
                continue

            # Check root itself
            if (root / "graph.db").exists():
                found.add(str(root) + "/")

            # Check immediate subdirectories (one level deep)
            try:
                for child in root.iterdir():
                    if child.is_dir() and (child / "graph.db").exists():
                        found.add(str(child) + "/")
            except PermissionError:
                continue

        return sorted(found)
