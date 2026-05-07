# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Database browser tab for exploring entity graphs."""

import logging
from pathlib import Path
from typing import Any

import gradio as gr

from video_ingestion_agent.webapp.config import AppConfig

logger = logging.getLogger(__name__)


def create_database_tab(services: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Create the database browser tab.

    Args:
        services: Dict of service instances.
        config: Application configuration.

    Returns:
        Dict of component references for external use.
    """
    components = {}

    with gr.Column():
        gr.Markdown(
            "Browse the entity graph produced by the ingestion pipeline. "
            "Select a database, click **Load Database**, then use the filters "
            "to explore segments, entities, and relationships.\n\n"
            "**Tip:** use the **Embedding analysis** section at the bottom to search "
            "frame embeddings by text description."
        )

        # ── Database selector (compact top bar) ──
        discovered_dbs = config.discover_databases()
        with gr.Row():
            db_dropdown = gr.Dropdown(
                label="Select Database",
                show_label=False,
                choices=discovered_dbs,
                value=config.default_db_dir
                if config.default_db_dir in discovered_dbs
                else config.default_db_dir,
                allow_custom_value=True,
                interactive=True,
                container=False,
                scale=4,
            )
            refresh_db_btn = gr.Button("↻", size="sm", scale=0, min_width=40)
            load_db_btn = gr.Button("Load Database", variant="primary", scale=1)

        stats_display = gr.JSON(label="Database Statistics", visible=False)

        # ── Filters group ──
        with gr.Group():
            gr.Markdown("#### Filters")
            with gr.Row():
                view_selector = gr.Radio(
                    choices=["Segments", "Entities", "Relationships", "Videos"],
                    value="Segments",
                    label="View",
                    scale=1,
                )
                action_filter = gr.Dropdown(
                    label="Action Type",
                    choices=["(All)"],
                    value="(All)",
                    interactive=True,
                    scale=1,
                )
                object_filter = gr.Dropdown(
                    label="Object",
                    choices=["(All)"],
                    value="(All)",
                    interactive=True,
                    scale=1,
                )
            with gr.Row():
                time_start = gr.Slider(
                    label="Start Time",
                    minimum=0,
                    maximum=10000,
                    value=0,
                    step=0.5,
                )
                time_end = gr.Slider(
                    label="End Time",
                    minimum=0,
                    maximum=10000,
                    value=10000,
                    step=0.5,
                )

        # ── Results ──
        row_count_display = gr.Markdown("*No data loaded.*")
        data_table = gr.Dataframe(
            label="Results",
            interactive=False,
            wrap=True,
            elem_classes=["db-results-table"],
        )

        # Hidden: video map data (populated on load, accessible via Videos radio)
        video_map_table = gr.Dataframe(
            value=[],
            headers=["id", "video_path", "duration", "fps"],
            interactive=False,
            visible=False,
        )

        # ── Entity Graph Visualization ──
        with gr.Accordion("Manipulated Objects", open=False):
            graph_plot = gr.Plot(label="Manipulated Objects")
            refresh_graph_btn = gr.Button("Refresh")

        # ── Embedding analysis ──
        with gr.Accordion("Embedding analysis", open=False):
            gr.Markdown("Search frame embeddings by text. Returns top-K matching frames.")
            with gr.Row():
                embedding_query = gr.Textbox(
                    label="Query",
                    placeholder="e.g. 'person cutting vegetables'",
                    scale=3,
                )
                embedding_top_k = gr.Number(
                    label="Top K",
                    value=10,
                    minimum=1,
                    maximum=50,
                    precision=0,
                    scale=1,
                )
                embedding_search_btn = gr.Button("Search", scale=1)
            embedding_gallery = gr.Gallery(
                label="Top-K matching frames",
                show_label=True,
                columns=5,
                object_fit="contain",
                height="auto",
            )
            embedding_results_table = gr.Dataframe(
                label="Results",
                value=[],
                headers=["frame_id", "video_id", "timestamp", "similarity", "video_path"],
                interactive=False,
                wrap=True,
                elem_classes=["db-results-table"],
            )

    # State
    db_service_state = gr.State(value=None)

    def resolve_graph_db_path(db_dir: str) -> str | None:
        """Resolve graph.db path from directory.

        Args:
            db_dir: Directory path or direct .db file path

        Returns:
            Path to graph.db or None if not found
        """
        db_path = Path(db_dir)

        # If it's a direct .db file, use it
        if db_path.suffix == ".db" and db_path.exists():
            return str(db_path)

        # Directory-based lookup
        if not db_path.exists() or not db_path.is_dir():
            return None

        # Look for graph.db
        graph_db = db_path / "graph.db"
        if graph_db.exists():
            return str(graph_db)

        # Try entity_graph.db as fallback
        graph_db = db_path / "entity_graph.db"
        if graph_db.exists():
            return str(graph_db)

        return None

    _DEFAULT_LIMIT = 100

    # Event handlers
    def load_database(db_dir: str):
        """Load database and return statistics and filter options."""
        db_path = resolve_graph_db_path(db_dir)

        if not db_path:
            return (
                gr.update(visible=True, value={"error": f"No graph.db found in {db_dir}"}),
                gr.update(value=[], headers=["id", "video_path", "duration", "fps"]),
                gr.update(choices=["(All)"]),
                gr.update(choices=["(All)"]),
                "*No data loaded.*",
                gr.update(value=[]),
                None,
                None,
                gr.update(),
                gr.update(),
            )

        try:
            from ..services import DatabaseService

            service = DatabaseService(db_path)

            # Get statistics
            stats = service.get_statistics()
            stats["loaded_from"] = db_path  # Add source path to stats

            # Get filter options
            options = service.get_filter_options()

            # Load initial data (segments)
            total = service.count_segments()
            segments = service.get_segments(limit=_DEFAULT_LIMIT)
            headers, data = _format_rows(segments)
            row_info = _format_row_count(len(data), total)

            # Video path ↔ ID map
            video_map = service.get_video_path_map()
            vm_headers = ["id", "video_path", "duration", "fps"]
            vm_data = [[m.get(h, "") for h in vm_headers] for m in video_map]

            max_dur = 1000.0
            for vm in video_map:
                try:
                    d = float(vm.get("duration", 0))
                    if d > max_dur:
                        max_dur = d
                except (ValueError, TypeError):
                    pass
            max_dur = round(max_dur, 1)

            return (
                gr.update(visible=True, value=stats),
                gr.update(value=vm_data, headers=vm_headers),
                gr.update(choices=["(All)"] + options.get("actions", [])),
                gr.update(choices=["(All)"] + options.get("objects", [])),
                row_info,
                gr.update(value=data, headers=headers),
                service,
                None,
                gr.update(maximum=max_dur, value=0),
                gr.update(maximum=max_dur, value=max_dur),
            )

        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return (
                gr.update(visible=True, value={"error": str(e)}),
                gr.update(value=[], headers=["id", "video_path", "duration", "fps"]),
                gr.update(choices=["(All)"]),
                gr.update(choices=["(All)"]),
                "*No data loaded.*",
                gr.update(value=[]),
                None,
                None,
                gr.update(),
                gr.update(),
            )

    def _format_row_count(shown: int, total: int) -> str:
        """Format row count display string."""
        if total == 0:
            return "*No matching rows.*"
        if shown >= total:
            return f"**Showing all {total} rows.**"
        return f"**Showing {shown} of {total} rows** (limit: {_DEFAULT_LIMIT})."

    def _format_rows(rows: list[dict]) -> tuple[list[str], list[list]]:
        """Convert a list of row-dicts into (headers, data) for gr.Dataframe.

        All columns present in the database are included.  Values are
        stringified so Gradio renders them uniformly.
        """
        if not rows:
            return [], []

        headers = list(rows[0].keys())
        data = []
        for row in rows:
            data.append([_cell_value(row.get(h)) for h in headers])
        return headers, data

    def _cell_value(val: object) -> str:
        """Format a single cell value for display."""
        if val is None:
            return ""
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    def apply_filters(
        service,
        view: str,
        action: str,
        obj: str,
        start: float,
        end: float,
    ):
        """Apply filters and reload data."""
        if service is None:
            return "*No data loaded.*", gr.update(value=[])

        time_range = (start, end) if end > start else None
        action_filter = action if action != "(All)" else None
        object_filter = obj if obj != "(All)" else None

        try:
            if view == "Segments":
                total = service.count_segments(
                    action_type=action_filter,
                    object_name=object_filter,
                    time_range=time_range,
                )
                rows = service.get_segments(
                    action_type=action_filter,
                    object_name=object_filter,
                    time_range=time_range,
                    limit=_DEFAULT_LIMIT,
                )

            elif view == "Entities":
                total = service.count_entities(time_range=time_range)
                rows = service.get_entities(
                    time_range=time_range,
                    limit=_DEFAULT_LIMIT,
                )

            elif view == "Relationships":
                total = service.count_relationships(time_range=time_range)
                rows = service.get_relationships(
                    time_range=time_range,
                    limit=_DEFAULT_LIMIT,
                )

            elif view == "Videos":
                rows = service.get_video_path_map()
                total = len(rows)
                headers = ["id", "video_path", "duration", "fps"]
                data = [[r.get(h, "") for h in headers] for r in rows]
                return _format_row_count(len(data), total), gr.update(value=data, headers=headers)

            else:
                return "*No data loaded.*", gr.update(value=[])

            headers, data = _format_rows(rows)
            if headers:
                return _format_row_count(len(data), total), gr.update(value=data, headers=headers)
            return _format_row_count(0, total), gr.update(value=[])

        except Exception as e:
            logger.error(f"Filter error: {e}")
            return "*Error loading data.*", gr.update(value=[])

    def create_graph(service, start: float, end: float):
        """Create entity graph visualization."""
        if service is None:
            return None

        try:
            from ..components.graph_visualizer import create_entity_graph_figure

            time_range = (start, end) if end > start else None
            graph_data = service.get_graph_data(time_range=time_range)

            fig = create_entity_graph_figure(
                entities=graph_data["nodes"],
                relationships=graph_data["edges"],
                time_range=time_range,
            )
            return fig

        except ImportError as e:
            logger.warning(f"Graph visualization not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Graph creation error: {e}")
            return None

    def run_embedding_search(service, query: str, top_k: int):
        """Search embeddings by text; return gallery images and table."""
        if service is None:
            return [], []
        if not (query or "").strip():
            return [], []
        k = max(1, min(50, int(top_k or 10)))
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        results = service.search_embeddings_by_text(text=query.strip(), top_k=k, device=device)
        gallery_list = []
        for i, r in enumerate(results):
            img = r.get("image")
            if img is not None:
                caption = f"#{i + 1} {r['frame_id']} sim={r['similarity']:.3f}"
                gallery_list.append((img, caption))
        table_data = [
            [r["frame_id"], r["video_id"], r["timestamp"], r["similarity"], r["video_path"] or ""]
            for r in results
        ]
        return gallery_list, table_data

    def refresh_databases():
        """Rescan directories and update the dropdown choices."""
        new_choices = config.discover_databases()
        return gr.update(choices=new_choices)

    # Wire up events
    refresh_db_btn.click(
        fn=refresh_databases,
        inputs=[],
        outputs=[db_dropdown],
    )

    load_db_btn.click(
        fn=load_database,
        inputs=[db_dropdown],
        outputs=[
            stats_display,
            video_map_table,
            action_filter,
            object_filter,
            row_count_display,
            data_table,
            db_service_state,
            graph_plot,
            time_start,
            time_end,
        ],
    )

    embedding_search_btn.click(
        fn=run_embedding_search,
        inputs=[db_service_state, embedding_query, embedding_top_k],
        outputs=[embedding_gallery, embedding_results_table],
    )

    _filter_inputs = [
        db_service_state,
        view_selector,
        action_filter,
        object_filter,
        time_start,
        time_end,
    ]
    _filter_outputs = [row_count_display, data_table]

    view_selector.change(fn=apply_filters, inputs=_filter_inputs, outputs=_filter_outputs)
    action_filter.change(fn=apply_filters, inputs=_filter_inputs, outputs=_filter_outputs)
    object_filter.change(fn=apply_filters, inputs=_filter_inputs, outputs=_filter_outputs)
    time_start.release(fn=apply_filters, inputs=_filter_inputs, outputs=_filter_outputs)
    time_end.release(fn=apply_filters, inputs=_filter_inputs, outputs=_filter_outputs)

    refresh_graph_btn.click(
        fn=create_graph,
        inputs=[db_service_state, time_start, time_end],
        outputs=[graph_plot],
    )

    components["db_dropdown"] = db_dropdown
    components["data_table"] = data_table

    return components
