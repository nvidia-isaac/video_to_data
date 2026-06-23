# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity graph visualization using a word cloud style Plotly plot."""

import math
from typing import Any

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


ENTITY_COLORS = {
    "person": "#FF6B6B",
    "object": "#4ECDC4",
    "location": "#45B7D1",
    "unknown": "#888888",
}

_WORD_CLOUD_PALETTE = [
    "#E63946",  # red
    "#457B9D",  # steel blue
    "#2A9D8F",  # teal
    "#E9C46A",  # gold
    "#F4A261",  # sandy orange
    "#264653",  # dark teal
    "#6A4C93",  # purple
    "#1982C4",  # blue
    "#8AC926",  # lime
    "#FF595E",  # coral
    "#CA6702",  # amber
    "#0077B6",  # ocean blue
    "#023E8A",  # navy
    "#D62828",  # crimson
    "#118AB2",  # cerulean
]

_MIN_FONT = 14
_MAX_FONT = 50


def _compute_importance(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> dict[str, float]:
    """Score each entity by its relationship count and temporal span."""
    rel_count: dict[str, int] = {}
    for rel in relationships:
        src = rel.get("source_id", rel.get("source", ""))
        tgt = rel.get("target_id", rel.get("target", ""))
        rel_count[src] = rel_count.get(src, 0) + 1
        rel_count[tgt] = rel_count.get(tgt, 0) + 1

    importance: dict[str, float] = {}
    for entity in entities:
        eid = entity.get("entity_id", entity.get("id", "unknown"))
        duration = max(entity.get("last_seen", 0) - entity.get("first_seen", 0), 0)
        connections = rel_count.get(eid, 0)
        importance[eid] = 1.0 + connections * 2.0 + min(duration, 100) * 0.1
    return importance


def _spiral_layout(
    labels: list[str],
    font_sizes: list[float],
    padding: float = 1.3,
) -> list[tuple[float, float]]:
    """Place labels along a spiral, largest first, avoiding overlaps.

    Args:
        labels: Text labels to place.
        font_sizes: Corresponding font sizes.
        padding: Multiplier on bounding boxes to add breathing room (1.0 = tight).
    """
    positions: list[tuple[float, float]] = []
    occupied: list[tuple[float, float, float, float]] = []

    for label, size in zip(labels, font_sizes, strict=False):
        half_w = (len(label) * size * 0.42 + size * 0.5) * padding
        half_h = size * 0.85 * padding

        angle = 0.0
        radius = 0.0
        placed = False
        for _ in range(5000):
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            if not any(
                abs(x - ox) < (half_w + ohw) and abs(y - oy) < (half_h + ohh)
                for ox, oy, ohw, ohh in occupied
            ):
                positions.append((x, y))
                occupied.append((x, y, half_w, half_h))
                placed = True
                break

            angle += 0.25
            radius += 0.12

        if not placed:
            positions.append((radius * math.cos(angle), radius * math.sin(angle)))
            occupied.append((positions[-1][0], positions[-1][1], half_w, half_h))

    return positions


def create_entity_graph_figure(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    time_range: tuple[float, float] | None = None,
    title: str = "Manipulated Objects",
) -> "go.Figure":
    """Create a word cloud style Plotly figure for entity visualization.

    Entity label size is proportional to importance (relationship count +
    temporal span).  Colour encodes entity type.

    Args:
        entities: List of entity dicts with entity_id, entity_type, first_seen, last_seen.
        relationships: List of relationship dicts with source_id, target_id, rel_type.
        time_range: Optional (start, end) to filter by time.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for graph visualization. Install with: pip install plotly"
        )

    if time_range:
        entities = [
            e
            for e in entities
            if e.get("last_seen", float("inf")) >= time_range[0]
            and e.get("first_seen", 0) <= time_range[1]
        ]

    # Keep only manipulated objects (drop person / location entities)
    entities = [e for e in entities if e.get("entity_type", e.get("type", "unknown")) == "object"]

    # Deduplicate by entity_id, merging time ranges
    merged: dict[str, dict[str, Any]] = {}
    for e in entities:
        eid = e.get("entity_id", e.get("id", "unknown"))
        if eid in merged:
            merged[eid]["first_seen"] = min(merged[eid]["first_seen"], e.get("first_seen", 0))
            merged[eid]["last_seen"] = max(merged[eid]["last_seen"], e.get("last_seen", 0))
        else:
            merged[eid] = {
                "entity_id": eid,
                "entity_type": e.get("entity_type", e.get("type", "unknown")),
                "first_seen": e.get("first_seen", 0),
                "last_seen": e.get("last_seen", 0),
            }
    entities = list(merged.values())

    if not entities:
        fig = go.Figure()
        fig.add_annotation(
            text="No manipulated objects found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    importance = _compute_importance(entities, relationships)

    info = []
    for entity in entities:
        eid = entity.get("entity_id", entity.get("id", "unknown"))
        info.append(
            {
                "id": eid,
                "type": entity.get("entity_type", entity.get("type", "unknown")),
                "first_seen": entity.get("first_seen", 0),
                "last_seen": entity.get("last_seen", 0),
                "importance": importance.get(eid, 1.0),
            }
        )
    info.sort(key=lambda x: x["importance"], reverse=True)

    max_imp = max(e["importance"] for e in info)
    min_imp = min(e["importance"] for e in info)
    imp_range = max_imp - min_imp if max_imp > min_imp else 1.0

    font_sizes = [
        _MIN_FONT + math.sqrt((e["importance"] - min_imp) / imp_range) * (_MAX_FONT - _MIN_FONT)
        for e in info
    ]

    labels = [e["id"] for e in info]
    positions = _spiral_layout(labels, font_sizes)

    # Build per-entity relationship summary for hover text
    rel_lookup: dict[str, list[str]] = {}
    for rel in relationships:
        src = rel.get("source_id", rel.get("source", ""))
        tgt = rel.get("target_id", rel.get("target", ""))
        rtype = rel.get("rel_type", rel.get("type", "related"))
        rel_lookup.setdefault(src, []).append(f"{rtype} → {tgt}")
        rel_lookup.setdefault(tgt, []).append(f"{src} → {rtype}")

    xs, ys, texts, hovers, colors = [], [], [], [], []
    for i, e in enumerate(info):
        x, y = positions[i]
        rels = rel_lookup.get(e["id"], [])
        hover = (
            f"<b>{e['id']}</b><br>"
            f"Time: {e['first_seen']:.1f}s – {e['last_seen']:.1f}s<br>"
            f"Interactions: {len(rels)}"
        )
        if rels:
            hover += "<br><br>" + "<br>".join(rels[:10])
            if len(rels) > 10:
                hover += f"<br>… and {len(rels) - 10} more"

        xs.append(x)
        ys.append(y)
        texts.append(e["id"])
        hovers.append(hover)
        colors.append(_WORD_CLOUD_PALETTE[i % len(_WORD_CLOUD_PALETTE)])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="text",
            text=texts,
            hoverinfo="text",
            hovertext=hovers,
            textfont=dict(
                size=font_sizes,
                color=colors,
                family="Arial Black, sans-serif",
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
            scaleanchor="x",
        ),
        plot_bgcolor="white",
    )

    return fig


def create_timeline_figure(
    segments: list[dict[str, Any]],
    video_duration: float = None,
    title: str = "Action Timeline",
) -> "go.Figure":
    """Create timeline visualization of action segments.

    Args:
        segments: List of segment dicts with start_t, end_t, action_type.
        video_duration: Total video duration for x-axis.
        title: Figure title.

    Returns:
        Plotly Figure object.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required. Install with: pip install plotly")

    if not segments:
        fig = go.Figure()
        fig.add_annotation(
            text="No segments to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    action_types = sorted({s.get("action_type", "unknown") for s in segments})
    y_positions = {action: i for i, action in enumerate(action_types)}

    fig = go.Figure()

    for segment in segments:
        action = segment.get("action_type", "unknown")
        start = segment.get("start_t", 0)
        end = segment.get("end_t", 0)
        obj = segment.get("primary_object_id", "")
        y = y_positions[action]

        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start, start],
                y=[y - 0.3, y - 0.3, y + 0.3, y + 0.3, y - 0.3],
                fill="toself",
                mode="lines",
                line=dict(width=1),
                hoverinfo="text",
                hovertext=f"{action}<br>{obj}<br>{start:.1f}s - {end:.1f}s",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Time (seconds)",
            range=[0, video_duration] if video_duration else None,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(action_types))),
            ticktext=action_types,
        ),
        height=max(200, len(action_types) * 50),
        margin=dict(l=150),
    )

    return fig
