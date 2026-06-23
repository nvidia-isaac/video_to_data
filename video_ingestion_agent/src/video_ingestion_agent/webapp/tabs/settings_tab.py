# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Settings tab for configuring pipeline and model parameters."""

import logging
from typing import Any

import gradio as gr

from video_ingestion_agent.webapp.config import AppConfig

logger = logging.getLogger(__name__)


def create_settings_tab(services: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Create the settings tab.

    Args:
        services: Dict of service instances.
        config: Application configuration.

    Returns:
        Dict of component references for external use.
    """
    components = {}

    with gr.Column():
        gr.Markdown("## Pipeline Settings")
        gr.Markdown("Configure model and pipeline parameters for video analysis.")

        # Model Settings - Two columns for VLM and LLM
        gr.Markdown("### Model Configuration")

        with gr.Row():
            # VLM Settings
            with gr.Column(scale=1):
                gr.Markdown("#### Vision-Language Model (VLM)")
                gr.Markdown("*Used for: Frame analysis, entity extraction, visual QA*")

                vlm_backend = gr.Radio(
                    label="VLM Backend",
                    choices=["local", "api"],
                    value="local",
                    info="Use local model or API endpoint",
                )

                vlm_model = gr.Dropdown(
                    label="VLM Model",
                    choices=[
                        "nvidia/Cosmos-Reason2-8B",
                        "Qwen/Qwen2-VL-7B-Instruct",
                        "Qwen/Qwen2-VL-72B-Instruct",
                        "llava-hf/llava-v1.6-mistral-7b-hf",
                        "llava-hf/llava-v1.6-34b-hf",
                        "gpt-4o",
                        "gpt-4o-mini",
                        "claude-3-5-sonnet-20241022",
                    ],
                    value="nvidia/Cosmos-Reason2-8B",
                    allow_custom_value=True,
                    info="Model for visual understanding",
                )

                vlm_device = gr.Radio(
                    label="VLM Device",
                    choices=["cuda", "cuda:0", "cuda:1", "cpu", "auto"],
                    value="cuda" if config.enable_gpu else "cpu",
                    info="Compute device for local VLM",
                )

                with gr.Accordion("VLM Generation Parameters", open=False):
                    vlm_temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.3,
                        step=0.1,
                        info="Lower for more precise visual descriptions",
                    )

                    vlm_max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=256,
                        maximum=4096,
                        value=512,
                        step=128,
                        info="Max tokens for visual analysis response",
                    )

                vlm_fps = gr.Slider(
                    label="Video FPS for Analysis",
                    minimum=1,
                    maximum=30,
                    value=4,
                    step=1,
                    info="Frames per second to sample from video",
                )

            # LLM Settings
            with gr.Column(scale=1):
                gr.Markdown("#### Large Language Model (LLM)")
                gr.Markdown("*Used for: Task decomposition, reasoning, synthesis*")

                llm_backend = gr.Radio(
                    label="LLM Backend",
                    choices=["local", "api"],
                    value="local",
                    info="Use local model or API endpoint",
                )

                llm_model = gr.Dropdown(
                    label="LLM Model",
                    choices=[
                        "nvidia/Cosmos-Reason2-8B",
                        "meta-llama/Llama-3.1-8B-Instruct",
                        "meta-llama/Llama-3.1-70B-Instruct",
                        "Qwen/Qwen2.5-7B-Instruct",
                        "Qwen/Qwen2.5-72B-Instruct",
                        "mistralai/Mistral-7B-Instruct-v0.3",
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-opus-20240229",
                    ],
                    value="nvidia/Cosmos-Reason2-8B",
                    allow_custom_value=True,
                    info="Model for text reasoning",
                )

                llm_device = gr.Radio(
                    label="LLM Device",
                    choices=["cuda", "cuda:0", "cuda:1", "cpu", "auto"],
                    value="cuda" if config.enable_gpu else "cpu",
                    info="Compute device for local LLM",
                )

                with gr.Accordion("LLM Generation Parameters", open=False):
                    llm_temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        info="Higher for more creative reasoning",
                    )

                    llm_max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=256,
                        maximum=4096,
                        value=1024,
                        step=128,
                        info="Max tokens for reasoning response",
                    )

                use_same_model = gr.Checkbox(
                    label="Use same model for VLM and LLM",
                    value=True,
                    info="When checked, VLM settings are used for both",
                )

        # API Settings
        with gr.Accordion("API Configuration", open=False):
            gr.Markdown("*Configure API endpoints when using API backend*")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**VLM API**")
                    vlm_api_base = gr.Textbox(
                        label="VLM API Base URL",
                        value="",
                        placeholder="https://api.openai.com/v1",
                    )
                    vlm_api_key = gr.Textbox(
                        label="VLM API Key",
                        value="",
                        type="password",
                        placeholder="sk-...",
                    )

                with gr.Column():
                    gr.Markdown("**LLM API**")
                    llm_api_base = gr.Textbox(
                        label="LLM API Base URL",
                        value="",
                        placeholder="https://api.openai.com/v1",
                    )
                    llm_api_key = gr.Textbox(
                        label="LLM API Key",
                        value="",
                        type="password",
                        placeholder="sk-...",
                    )

        gr.Markdown("---")

        # Pipeline Settings
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Pipeline Configuration")

                max_sub_tasks = gr.Slider(
                    label="Max Sub-Tasks",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    info="Maximum sub-tasks for query decomposition",
                )

                max_relaxation = gr.Slider(
                    label="Max Relaxation Level",
                    minimum=0,
                    maximum=5,
                    value=3,
                    step=1,
                    info="Max search relaxation (0=strict, higher=permissive)",
                )

                max_iterations = gr.Slider(
                    label="Max Executor Iterations",
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    info="Maximum iterations in executor loop",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Search & Extraction")

                search_top_k = gr.Slider(
                    label="Search Top K",
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    info="Results per search query",
                )

                clips_dir = gr.Textbox(
                    label="Clips Output Directory",
                    value=config.default_clips_dir,
                    placeholder="outputs/clips",
                )

                clip_padding = gr.Slider(
                    label="Clip Padding (seconds)",
                    minimum=0.0,
                    maximum=5.0,
                    value=0.5,
                    step=0.5,
                    info="Extra time before/after clip",
                )

                with gr.Row():
                    use_vector_search = gr.Checkbox(
                        label="Vector Search",
                        value=True,
                    )
                    use_frame_search = gr.Checkbox(
                        label="Frame Search",
                        value=True,
                    )

        # Actions
        gr.Markdown("---")

        with gr.Row():
            save_btn = gr.Button("Apply Settings", variant="primary")
            reset_btn = gr.Button("Reset to Defaults")

        status_display = gr.Markdown("")

        # Config preview
        with gr.Accordion("Current Configuration (JSON)", open=False):
            config_preview = gr.JSON(
                label="Configuration",
                value=_get_default_config(config),
            )

    # State
    settings_state = gr.State(value=_get_default_config(config))

    # Event handlers
    def sync_models(use_same: bool, vlm_back, vlm_mod, vlm_dev, vlm_temp, vlm_max):
        """Sync LLM settings with VLM when checkbox is checked."""
        if use_same:
            return (
                vlm_back,  # llm_backend
                vlm_mod,  # llm_model
                vlm_dev,  # llm_device
                vlm_temp,  # llm_temperature
                vlm_max,  # llm_max_tokens
            )
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    def apply_settings(
        # VLM settings
        vlm_back,
        vlm_mod,
        vlm_dev,
        vlm_temp,
        vlm_max,
        vlm_fps_val,
        # LLM settings
        llm_back,
        llm_mod,
        llm_dev,
        llm_temp,
        llm_max,
        use_same,
        # API settings
        vlm_api_b,
        vlm_api_k,
        llm_api_b,
        llm_api_k,
        # Pipeline settings
        max_tasks,
        max_relax,
        max_iter,
        # Search settings
        top_k,
        clips_path,
        padding,
        use_vec,
        use_frame,
    ):
        """Apply settings and return updated config."""

        # If using same model, copy VLM settings to LLM
        if use_same:
            llm_back = vlm_back
            llm_mod = vlm_mod
            llm_dev = vlm_dev
            llm_temp = vlm_temp
            llm_max = vlm_max
            llm_api_b = vlm_api_b
            llm_api_k = vlm_api_k

        new_config = {
            "vlm": {
                "backend": vlm_back,
                "model": vlm_mod,
                "device": vlm_dev,
                "temperature": vlm_temp,
                "max_tokens": int(vlm_max),
                "fps": int(vlm_fps_val),
                "api_base": vlm_api_b if vlm_back == "api" else None,
                "has_api_key": bool(vlm_api_k) if vlm_back == "api" else False,
            },
            "llm": {
                "backend": llm_back,
                "model": llm_mod,
                "device": llm_dev,
                "temperature": llm_temp,
                "max_tokens": int(llm_max),
                "api_base": llm_api_b if llm_back == "api" else None,
                "has_api_key": bool(llm_api_k) if llm_back == "api" else False,
            },
            "use_same_model": use_same,
            "pipeline": {
                "max_sub_tasks": int(max_tasks),
                "max_relaxation_level": int(max_relax),
                "max_iterations": int(max_iter),
            },
            "search": {
                "top_k": int(top_k),
                "use_vector_search": use_vec,
                "use_frame_search": use_frame,
            },
            "clips": {
                "output_dir": clips_path,
                "padding_seconds": padding,
            },
        }

        # Store in services for other tabs to access
        services["pipeline_config"] = new_config

        # Also store API keys in memory (not in config JSON for security)
        if vlm_api_k:
            services["vlm_api_key"] = vlm_api_k
        if llm_api_k:
            services["llm_api_key"] = llm_api_k

        logger.info(f"Applied settings: VLM={vlm_mod}, LLM={llm_mod}")

        return (
            f"Settings applied. VLM: {vlm_mod}, LLM: {llm_mod}",
            new_config,
            new_config,
        )

    def reset_settings():
        """Reset to default settings."""
        default = _get_default_config(config)
        return (
            # VLM
            "local",
            "nvidia/Cosmos-Reason2-8B",
            "cuda" if config.enable_gpu else "cpu",
            0.3,
            512,
            4,
            # LLM
            "local",
            "nvidia/Cosmos-Reason2-8B",
            "cuda" if config.enable_gpu else "cpu",
            0.7,
            1024,
            True,
            # API
            "",
            "",
            "",
            "",
            # Pipeline
            5,
            3,
            20,
            # Search
            20,
            config.default_clips_dir,
            0.5,
            True,
            True,
            # Status
            "Settings reset to defaults.",
            default,
            default,
        )

    # Wire up events
    use_same_model.change(
        fn=sync_models,
        inputs=[
            use_same_model,
            vlm_backend,
            vlm_model,
            vlm_device,
            vlm_temperature,
            vlm_max_tokens,
        ],
        outputs=[llm_backend, llm_model, llm_device, llm_temperature, llm_max_tokens],
    )

    save_btn.click(
        fn=apply_settings,
        inputs=[
            # VLM
            vlm_backend,
            vlm_model,
            vlm_device,
            vlm_temperature,
            vlm_max_tokens,
            vlm_fps,
            # LLM
            llm_backend,
            llm_model,
            llm_device,
            llm_temperature,
            llm_max_tokens,
            use_same_model,
            # API
            vlm_api_base,
            vlm_api_key,
            llm_api_base,
            llm_api_key,
            # Pipeline
            max_sub_tasks,
            max_relaxation,
            max_iterations,
            # Search
            search_top_k,
            clips_dir,
            clip_padding,
            use_vector_search,
            use_frame_search,
        ],
        outputs=[status_display, config_preview, settings_state],
    )

    reset_btn.click(
        fn=reset_settings,
        inputs=[],
        outputs=[
            # VLM
            vlm_backend,
            vlm_model,
            vlm_device,
            vlm_temperature,
            vlm_max_tokens,
            vlm_fps,
            # LLM
            llm_backend,
            llm_model,
            llm_device,
            llm_temperature,
            llm_max_tokens,
            use_same_model,
            # API
            vlm_api_base,
            vlm_api_key,
            llm_api_base,
            llm_api_key,
            # Pipeline
            max_sub_tasks,
            max_relaxation,
            max_iterations,
            # Search
            search_top_k,
            clips_dir,
            clip_padding,
            use_vector_search,
            use_frame_search,
            # Status
            status_display,
            config_preview,
            settings_state,
        ],
    )

    components["settings_state"] = settings_state
    components["vlm_model"] = vlm_model
    components["llm_model"] = llm_model

    return components


def _get_default_config(config: AppConfig) -> dict[str, Any]:
    """Get default configuration dict."""
    return {
        "vlm": {
            "backend": "local",
            "model": "nvidia/Cosmos-Reason2-8B",
            "device": "cuda" if config.enable_gpu else "cpu",
            "temperature": 0.3,
            "max_tokens": 512,
            "fps": 4,
        },
        "llm": {
            "backend": "local",
            "model": "nvidia/Cosmos-Reason2-8B",
            "device": "cuda" if config.enable_gpu else "cpu",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "use_same_model": True,
        "pipeline": {
            "max_sub_tasks": 5,
            "max_relaxation_level": 3,
            "max_iterations": 20,
        },
        "search": {
            "top_k": 20,
            "use_vector_search": True,
            "use_frame_search": True,
        },
        "clips": {
            "output_dir": config.default_clips_dir,
            "padding_seconds": 0.5,
        },
    }
