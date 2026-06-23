# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base tool interface for agentic workflows."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    data: Any
    error: str | None = None

    def to_string(self) -> str:
        """Convert result to string for LLM context."""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, list):
            if not self.data:
                return "No results found."
            return "\n".join(str(item) for item in self.data)
        return str(self.data)


class BaseTool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for LLM to reference."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, dict[str, Any]]:
        """
        Parameter schema for the tool.

        Format:
        {
            "param_name": {
                "type": "string|number|boolean|array",
                "description": "...",
                "required": True/False,
                "default": <value>  # optional
            }
        }
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema for LLM function calling."""
        required = [name for name, spec in self.parameters.items() if spec.get("required", False)]

        properties = {}
        for name, spec in self.parameters.items():
            properties[name] = {"type": spec["type"], "description": spec["description"]}
            if "enum" in spec:
                properties[name]["enum"] = spec["enum"]

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }
