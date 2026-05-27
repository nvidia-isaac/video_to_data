# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Base class for agent nodes.

Provides common functionality for LLM calls, JSON parsing, and tool access.
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from video_ingestion_agent.models.model_manager import BaseModel, get_model_manager
from video_ingestion_agent.retrieval.config import RetrievalConfig
from video_ingestion_agent.retrieval.state import AgentState
from video_ingestion_agent.retrieval.tools.base import BaseTool
from video_ingestion_agent.utils.parsing import parse_llm_json

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for all agent nodes.

    Provides:
    - LLM model management and calling
    - JSON parsing from LLM responses
    - Tool access
    - Debug logging for LLM inputs/outputs
    """

    def __init__(
        self,
        config: RetrievalConfig,
        tools: dict[str, BaseTool],
        debug: bool = False,
        debug_dir: str | None = None,
    ):
        """Initialize the node.

        Args:
            config: Retrieval agent configuration
            tools: Dict of tool_name -> BaseTool instances
            debug: Enable debug logging of LLM inputs/outputs
            debug_dir: Directory to save debug logs (default: ./debug_logs)
        """
        self.config = config
        self.tools = tools
        self.llm_model_name = config.models.llm_model
        self.device = config.models.device
        self.backend = config.models.llm_backend
        self.api_key = config.models.api_key
        self._model: BaseModel | None = None
        self.debug = debug or os.environ.get("VIDEO_INGESTION_AGENT_DEBUG", "").lower() in (
            "1",
            "true",
        )
        self._call_count = 0

        # Debug directory (can include session timestamp from caller)
        self.debug_dir = Path(debug_dir) if debug_dir else Path("./debug_logs")

    def _get_model(self) -> BaseModel:
        """Get LLM model (lazy initialization)."""
        if self._model is None:
            manager = get_model_manager()
            self._model = manager.get_model(
                model_name=self.llm_model_name,
                backend=self.backend,
                device=self.device,
                api_key=self.api_key,
            )
            logger.info(f"Loaded LLM: {self.llm_model_name}")
        return self._model

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """Call LLM with a prompt.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for LLM behavior/persona
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        model = self._get_model()
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": prompt})

        # Debug: log input
        if self.debug:
            self._log_llm_input(conversation)

        response = model.generate_text(
            conversation=conversation, max_new_tokens=max_tokens, temperature=0.0
        )

        # Debug: log output
        if self.debug:
            self._log_llm_output(response)

        return response

    def _log_llm_input(self, conversation: list[dict]) -> None:
        """Log LLM input to file.

        Args:
            conversation: The conversation messages sent to LLM
        """
        self._call_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # File logging only
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.debug_dir / f"{timestamp}_{self.name}_{self._call_count:03d}_input.txt"
            with open(log_file, "w") as f:
                f.write(f"Node: {self.name}\n")
                f.write(f"Model: {self.llm_model_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("=" * 80 + "\n\n")
                for msg in conversation:
                    f.write(f"[{msg['role'].upper()}]\n")
                    f.write(msg["content"])
                    f.write("\n\n" + "-" * 40 + "\n\n")

    def _log_llm_output(self, response: str) -> None:
        """Log LLM output to file.

        Args:
            response: The LLM response text
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # File logging only
        if self.debug_dir:
            log_file = self.debug_dir / f"{timestamp}_{self.name}_{self._call_count:03d}_output.txt"
            with open(log_file, "w") as f:
                f.write(f"Node: {self.name}\n")
                f.write(f"Model: {self.llm_model_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("=" * 80 + "\n\n")
                f.write("[RESPONSE]\n")
                f.write(response)

    def _parse_json(self, response: str) -> dict:
        """Extract JSON from LLM response.

        Handles both ```json``` code blocks and raw JSON.
        Delegates to ``utils.parsing.parse_llm_json``.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON as dict, or empty dict on failure
        """
        try:
            return parse_llm_json(response, expect_array=False)
        except ValueError as e:
            logger.warning(f"JSON parse error: {e}")
        return {}

    @abstractmethod
    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Execute the node.

        Args:
            state: Current agent state

        Returns:
            Dict of state updates
        """
        pass

    @property
    def name(self) -> str:
        """Return the node name for logging."""
        return self.__class__.__name__
