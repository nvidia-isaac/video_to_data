# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""LangGraph pipeline visualization component."""

from dataclasses import dataclass


@dataclass
class NodeState:
    """State of a node in the pipeline."""

    name: str
    display_name: str
    status: str = "pending"  # "pending", "running", "complete", "error"
    iterations: int = 0
    last_output: str | None = None
    message: str = ""


class PipelineVisualizer:
    """Visualize LangGraph execution flow."""

    PARALLEL_NODES = [
        ("task_decomposer", "Task Decomposer"),
        ("task_search", "Sub-task Search"),
        ("vqa_synthesizer", "VQA Synthesizer"),
    ]

    SEQUENTIAL_NODES = [
        ("task_decomposer", "Task Decomposer"),
        ("search_planner", "Search Planner"),
        ("executor", "Executor"),
        ("analyzer", "Analyzer"),
        ("vqa_synthesizer", "VQA Synthesizer"),
    ]

    DEFAULT_NODES = PARALLEL_NODES

    def __init__(self, nodes: list[tuple] = None, parallel: bool = True):
        """Initialize visualizer.

        Args:
            nodes: List of (node_id, display_name) tuples.
            parallel: Use parallel pipeline layout (default True).
        """
        if nodes is None:
            nodes = self.PARALLEL_NODES if parallel else self.SEQUENTIAL_NODES
        self.parallel = parallel
        self.node_states: dict[str, NodeState] = {
            node_id: NodeState(name=node_id, display_name=display_name)
            for node_id, display_name in nodes
        }
        self.node_order = [node_id for node_id, _ in nodes]
        self.current_node: str | None = None

    def reset(self):
        """Reset all nodes to pending state."""
        for state in self.node_states.values():
            state.status = "pending"
            state.iterations = 0
            state.last_output = None
            state.message = ""
        self.current_node = None

    def start_node(self, node_name: str, message: str = ""):
        """Mark a node as started."""
        if node_name in self.node_states:
            self.current_node = node_name
            self.node_states[node_name].status = "running"
            self.node_states[node_name].message = message
            if node_name in ("executor", "task_search"):
                self.node_states[node_name].iterations += 1

    def complete_node(self, node_name: str, output: str = None, message: str = ""):
        """Mark a node as complete."""
        if node_name in self.node_states:
            self.node_states[node_name].status = "complete"
            self.node_states[node_name].last_output = output
            self.node_states[node_name].message = message
            self.current_node = None

    def error_node(self, node_name: str, error: str):
        """Mark a node as errored."""
        if node_name in self.node_states:
            self.node_states[node_name].status = "error"
            self.node_states[node_name].message = error
            self.current_node = None

    @property
    def is_complete(self) -> bool:
        """Check if pipeline is complete."""
        return all(state.status == "complete" for state in self.node_states.values())

    def to_markdown(self) -> str:
        """Format pipeline as Markdown."""
        lines = ["### Pipeline Status", ""]

        for node_id in self.node_order:
            state = self.node_states[node_id]

            icon = {
                "pending": "⬜",
                "running": "🔄",
                "complete": "✅",
                "error": "❌",
            }.get(state.status, "⬜")

            label = state.display_name
            if node_id == "executor" and state.iterations > 0:
                label += f" (iteration {state.iterations})"
            elif node_id == "task_search" and state.iterations > 0:
                label += f" ({state.iterations} completed)"

            line = f"{icon} **{label}**"
            if state.message:
                line += f": {state.message}"

            lines.append(line)

            # Show arrow to next node (except last)
            if node_id != self.node_order[-1]:
                lines.append("  ↓")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Format pipeline as a horizontal progress bar with dots and lines."""
        html = ['<div class="pipeline-bar">']

        for i, node_id in enumerate(self.node_order):
            state = self.node_states[node_id]
            status = state.status
            label = state.display_name

            if node_id == "executor" and state.iterations > 0:
                label += f" ({state.iterations}x)"
            elif node_id == "task_search" and state.iterations > 0:
                label += f" ({state.iterations} done)"

            html.append(
                f'<div class="pb-node {status}">'
                f'<div class="pb-dot"></div>'
                f'<div class="pb-label">{label}</div>'
                f"</div>"
            )

            if i < len(self.node_order) - 1:
                line_status = "complete" if status == "complete" else "pending"
                html.append(f'<div class="pb-line {line_status}"></div>')

        html.append("</div>")

        for state in self.node_states.values():
            if state.status == "running" and state.message:
                html.append(f'<div class="pb-message">{state.message}</div>')
                break

        return "\n".join(html)

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram for visualization."""
        lines = ["graph TD"]

        # Define nodes with styling
        for node_id in self.node_order:
            state = self.node_states[node_id]
            label = state.display_name

            if node_id == "executor" and state.iterations > 0:
                label += f"<br/>(iter {state.iterations})"

            lines.append(f'    {node_id}["{label}"]')

            # Style based on status
            if state.status == "running":
                lines.append(f"    style {node_id} fill:#FFD700,stroke:#333")
            elif state.status == "complete":
                lines.append(f"    style {node_id} fill:#90EE90,stroke:#333")
            elif state.status == "error":
                lines.append(f"    style {node_id} fill:#FF6B6B,stroke:#333")
            else:
                lines.append(f"    style {node_id} fill:#E0E0E0,stroke:#333")

        # Define edges
        lines.append("")
        if self.parallel:
            lines.append("    task_decomposer -->|fan-out| task_search")
            lines.append("    task_search -->|merge| vqa_synthesizer")
        else:
            lines.append("    task_decomposer --> search_planner")
            lines.append("    search_planner --> executor")
            lines.append("    executor --> analyzer")
            lines.append("    analyzer -->|continue| search_planner")
            lines.append("    analyzer -->|done| vqa_synthesizer")

        return "\n".join(lines)

    def get_state_dict(self) -> dict[str, dict]:
        """Get state as dictionary for JSON serialization."""
        return {
            node_id: {
                "name": state.name,
                "display_name": state.display_name,
                "status": state.status,
                "iterations": state.iterations,
                "message": state.message,
            }
            for node_id, state in self.node_states.items()
        }


def create_default_visualizer(parallel: bool = True) -> PipelineVisualizer:
    """Create a visualizer for the default LangGraph agent pipeline.

    Args:
        parallel: Use parallel pipeline layout (default True).
    """
    return PipelineVisualizer(parallel=parallel)
