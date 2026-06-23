# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video query tab — dark card-based layout with horizontal pipeline."""

import logging
from pathlib import Path
from typing import Any

import gradio as gr

from video_ingestion_agent.retrieval.config import RetrievalConfig, RetrievalModelConfig
from video_ingestion_agent.webapp.components.pipeline_visualizer import PipelineVisualizer
from video_ingestion_agent.webapp.config import AppConfig

logger = logging.getLogger(__name__)


_SVG_PLACEHOLDER = (
    "<svg width='40' height='40' viewBox='0 0 24 24' fill='none'>"
    "  <rect x='3' y='3' width='18' height='18' rx='2' stroke='#555' stroke-width='1.5'/>"
    "  <path d='M8 6l2 2-2 2' stroke='#888' stroke-width='1.5'/>"
    "  <circle cx='16' cy='16' r='2' stroke='#888' stroke-width='1.5'/>"
    "</svg>"
)


def _render_clip_cards(clips: list[dict], extracted_paths: list[str] | None = None) -> str:
    """Render extracted clips as an HTML card grid with inline video players."""
    if not clips:
        return (
            '<div class="clips-empty">No clips extracted yet. Enter a query and click Search.</div>'
        )

    cards = []
    for i, clip in enumerate(clips):
        desc = clip.get("description", "")
        if len(desc) > 90:
            desc = desc[:87] + "..."
        action = clip.get("action", "-")
        obj = clip.get("object", "-")
        video_name = Path(clip.get("video_path", "")).name if clip.get("video_path") else "-"

        if extracted_paths and i < len(extracted_paths) and extracted_paths[i]:
            thumb = (
                f'<video class="clip-video" controls preload="metadata"'
                f' src="/gradio_api/file={extracted_paths[i]}"></video>'
            )
        else:
            thumb = _SVG_PLACEHOLDER

        cards.append(
            f'<div class="clip-card" data-idx="{i}">'
            f'  <div class="clip-thumb">{thumb}</div>'
            f'  <div class="clip-info">'
            f'    <div class="clip-id">Segment ID: {i + 1}</div>'
            f'    <div class="clip-meta">{action} ({obj}) &middot; {video_name}</div>'
            f'    <div class="clip-desc">{desc}</div>'
            f"  </div>"
            f"</div>"
        )

    return '<div class="clips-grid">' + "\n".join(cards) + "</div>"


def create_query_tab(services: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Create the video query tab with dark card-based layout."""
    components = {}

    parallel_mode = getattr(config, "parallel_tasks", True)
    visualizer = PipelineVisualizer(parallel=parallel_mode)

    with gr.Column(elem_classes=["rt-shell"]):
        gr.Markdown(
            "Ask natural-language questions about ingested videos. "
            "The agent decomposes your query into sub-tasks, searches the entity graph "
            "and frame embeddings, then synthesizes an answer with matching clips.\n\n"
            "**Quick start:** select a database below, type a query, and click **Search**."
        )

        # --- Search bar (Gemini-style: input on top, toolbar below) ---
        with gr.Column(elem_classes=["rt-searchbar"]):
            with gr.Row():
                query_input = gr.Textbox(
                    show_label=False,
                    placeholder='e.g., "Find all clips where someone picks up a mug"',
                    lines=1,
                    scale=5,
                    container=False,
                    elem_classes=["rt-query-input"],
                )
            with gr.Row(elem_classes=["rt-toolbar"]):
                discovered_dbs = config.discover_databases()
                gr.Button(
                    "📂 Database",
                    size="sm",
                    scale=0,
                    min_width=90,
                    interactive=False,
                    elem_classes=["rt-db-label"],
                )
                db_dropdown = gr.Dropdown(
                    label="Database",
                    show_label=False,
                    choices=discovered_dbs,
                    value=config.default_db_dir
                    if config.default_db_dir in discovered_dbs
                    else config.default_db_dir,
                    allow_custom_value=True,
                    interactive=True,
                    container=False,
                    scale=1,
                    min_width=240,
                    elem_classes=["rt-db-dropdown"],
                )
                refresh_db_btn = gr.Button(
                    "↻",
                    size="sm",
                    scale=0,
                    min_width=36,
                    elem_classes=["rt-refresh-btn"],
                )
                gr.HTML('<div class="rt-spacer"></div>')
                search_btn = gr.Button(
                    "Search",
                    variant="primary",
                    scale=0,
                    min_width=120,
                    elem_classes=["rt-search-btn"],
                )

        db_status = gr.Textbox(value="", interactive=False, show_label=False, visible=False)

        clear_btn = gr.Button("Clear", visible=False, elem_classes=["rt-clear-btn"])

        # --- Pipeline Status (horizontal bar) ---
        with gr.Row(elem_classes=["rt-pipeline-wrap"]):
            pipeline_display = gr.HTML(
                value=visualizer.to_html(),
                elem_classes=["rt-pipeline"],
            )

        # --- Results ---
        gr.Markdown("### Results", elem_classes=["rt-results-heading"])

        answer_display = gr.Markdown(
            value="*Enter a query and click Search to find clips.*",
            elem_classes=["rt-answer"],
        )

        clips_grid = gr.HTML(
            value=_render_clip_cards([]),
            elem_classes=["rt-clips-grid"],
        )

        # --- Reconstruct from a query result ---
        # Always visible so users can see the option exists; the button is
        # disabled until a query returns clips. Clicking jumps to the
        # Reconstruct tab with the picked clip pre-loaded.
        gr.Markdown(
            "After search, pick a result clip below and click **Reconstruct →** "
            "to run the per-segment 3D reconstruction chain on it.",
            elem_classes=["rt-reconstruct-hint"],
        )
        with gr.Row(elem_classes=["rt-reconstruct-row"]):
            reconstruct_picker = gr.Dropdown(
                label="Pick a clip to reconstruct",
                choices=[],
                value=None,
                interactive=True,
                scale=4,
            )
            reconstruct_btn = gr.Button(
                "Reconstruct →",
                variant="primary",
                scale=1,
                min_width=160,
                interactive=False,
            )

        # --- Working Memory (collapsed) ---
        with gr.Accordion("Working Memory", open=False, elem_classes=["rt-wm-accordion"]):
            working_memory_display = gr.Textbox(
                label="Agent Working Memory",
                lines=12,
                max_lines=30,
                interactive=False,
                autoscroll=True,
                elem_classes=["log-output"],
            )

    # --- State ---
    query_result_state = gr.State(value=None)

    # ─── helpers ────────────────────────────────────────────────────────

    def resolve_db_paths(db_dir: str) -> tuple[str | None, str | None, str]:
        db_path = Path(db_dir)
        if db_path.suffix == ".db" and db_path.exists():
            vector_path = db_path.parent / f"{db_path.stem}_vector.db"
            vector_db = str(vector_path) if vector_path.exists() else None
            status = "✓ Graph DB" + (", ✓ Vector DB" if vector_db else "")
            return str(db_path), vector_db, status
        if not db_path.exists():
            return None, None, f"✗ Not found: {db_dir}"
        if not db_path.is_dir():
            return None, None, f"✗ Not a directory: {db_dir}"
        graph_db = db_path / "graph.db"
        vector_db = db_path / "vector.db"
        if not graph_db.exists():
            graph_db = db_path / "entity_graph.db"
        if not graph_db.exists():
            return None, None, f"✗ No graph.db in {db_dir}"
        vector_path = str(vector_db) if vector_db.exists() else None
        status = f"✓ {graph_db.name}" + (f", ✓ {vector_db.name}" if vector_path else "")
        return str(graph_db), vector_path, status

    # ─── run_query (streaming) ──────────────────────────────────────────

    def _reconstruct_choices(clips: list[dict]) -> list[tuple[str, str]]:
        """Format query-result clips as (label, index) Dropdown choices."""
        out: list[tuple[str, str]] = []
        for i, c in enumerate(clips):
            action = c.get("action", "?")
            obj = c.get("object", "?")
            start = float(c.get("start_t", c.get("start_time", 0)))
            end = float(c.get("end_t", c.get("end_time", 0)))
            out.append((f"Clip {i + 1}: {action} ({obj}) — {start:.1f}–{end:.1f}s", str(i)))
        return out

    def _empty_yield(pipeline_html, answer, db_st, wm):
        return (
            pipeline_html,
            answer,
            db_st,
            wm,
            _render_clip_cards([]),
            None,
            gr.update(choices=[], value=None),  # reconstruct_picker
            gr.update(interactive=False),  # reconstruct_btn
        )

    def run_query(query: str, db_dir: str):
        if not query.strip():
            yield _empty_yield(
                visualizer.to_html(),
                "*Please enter a query.*",
                "",
                "",
            )
            return

        graph_db_path, vector_db_path, db_status_msg = resolve_db_paths(db_dir)
        if not graph_db_path:
            yield _empty_yield(
                visualizer.to_html(),
                f"*{db_status_msg}*",
                db_status_msg,
                "",
            )
            return

        if not vector_db_path:
            logger.warning(f"Vector database not found in {db_dir}")

        visualizer.reset()

        try:
            from ..services import QueryService
            from ..services.query_service import QueryResult

            retrieval_config = RetrievalConfig(
                models=RetrievalModelConfig(
                    llm_model=config.llm_model,
                    llm_backend=config.llm_backend,
                    embedding_model=config.embedding_model,
                    api_key=config.api_key,
                    device=config.default_device if config.enable_gpu else "cpu",
                    vllm_url=config.vllm_url,
                    vllm_local_media=config.vllm_local_media,
                    vllm_tp_size=config.vllm_tp_size,
                    vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                ),
            )

            service = QueryService(
                graph_db_path=graph_db_path,
                vector_db_path=vector_db_path,
                clips_dir=config.default_clips_dir,
                config=retrieval_config,
            )

            result = None
            accumulated_memory: list[str] = []

            for update in service.run_query_streaming(query=query):
                if isinstance(update, QueryResult):
                    result = update
                else:
                    node_name = update.node_name
                    visualizer.start_node(node_name, update.message)
                    visualizer.complete_node(node_name, message=update.message)

                    if update.output and "working_memory" in update.output:
                        accumulated_memory = update.output["working_memory"]

                    yield (
                        visualizer.to_html(),
                        f"*Searching… {update.message}*",
                        db_status_msg,
                        "\n\n---\n\n".join(accumulated_memory)
                        if accumulated_memory
                        else "Processing…",
                        _render_clip_cards([]),
                        None,
                        gr.update(choices=[], value=None),
                        gr.update(interactive=False),
                    )

            if result is None:
                result = QueryResult(
                    success=False,
                    query=query,
                    error_message="No result from agent",
                    elapsed_time=0,
                )

            if result.success:
                answer = result.final_answer or "Done."

                all_memory = accumulated_memory or result.working_memory
                working_memory = (
                    "\n\n---\n\n".join(all_memory) if all_memory else "No working memory."
                )

                extracted_paths: list[str] = []
                if result.clips_extracted:
                    for clip_path_str in result.clips_extracted:
                        clip_path = Path(clip_path_str)
                        if clip_path.exists():
                            extracted_paths.append(str(clip_path.resolve()))
                        else:
                            extracted_paths.append("")

                    if any(extracted_paths):
                        valid = [p for p in extracted_paths if p]
                        logger.info(
                            f"Found {len(valid)} extracted clips, "
                            f"serving via /file= (paths: {valid[:3]}{'...' if len(valid) > 3 else ''})"
                        )

                for node_id in visualizer.node_order:
                    visualizer.complete_node(node_id)

                choices = _reconstruct_choices(result.clips)
                yield (
                    visualizer.to_html(),
                    f"{answer}\n\n**{len(result.clips)} clips** extracted in {result.elapsed_time:.1f}s",
                    db_status_msg,
                    working_memory,
                    _render_clip_cards(result.clips, extracted_paths),
                    result,
                    gr.update(
                        choices=choices,
                        value=choices[0][1] if choices else None,
                    ),
                    gr.update(interactive=bool(choices)),
                )
            else:
                visualizer.error_node("task_decomposer", result.error_message or "Unknown error")
                yield _empty_yield(
                    visualizer.to_html(),
                    f"**Query Failed:** {result.error_message}",
                    db_status_msg,
                    "",
                )

        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            visualizer.error_node("task_decomposer", str(e))
            yield _empty_yield(
                visualizer.to_html(),
                f"**Error:** {e}",
                db_status_msg,
                "",
            )

    # ─── clear ──────────────────────────────────────────────────────────

    def clear_query():
        visualizer.reset()
        return (
            "",
            visualizer.to_html(),
            "*Enter a query and click Search to find clips.*",
            "",
            "",
            _render_clip_cards([]),
            None,
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
        )

    # ─── helpers ────────────────────────────────────────────────────────

    def on_db_dropdown_change(selection: str):
        if not selection:
            return "Select a database"
        _, _, status = resolve_db_paths(selection)
        return status

    def refresh_databases():
        return gr.update(choices=config.discover_databases())

    # ─── wire events ────────────────────────────────────────────────────

    _query_outputs = [
        pipeline_display,
        answer_display,
        db_status,
        working_memory_display,
        clips_grid,
        query_result_state,
        reconstruct_picker,
        reconstruct_btn,
    ]

    search_btn.click(
        fn=run_query,
        inputs=[query_input, db_dropdown],
        outputs=_query_outputs,
    )

    db_dropdown.change(
        fn=on_db_dropdown_change,
        inputs=[db_dropdown],
        outputs=[db_status],
    )

    refresh_db_btn.click(
        fn=refresh_databases,
        inputs=[],
        outputs=[db_dropdown],
    )

    clear_btn.click(
        fn=clear_query,
        inputs=[],
        outputs=[query_input] + _query_outputs,
    )

    components["query_input"] = query_input
    components["search_btn"] = search_btn
    components["db_dropdown"] = db_dropdown
    components["query_result_state"] = query_result_state
    components["reconstruct_picker"] = reconstruct_picker
    components["reconstruct_btn"] = reconstruct_btn

    return components
