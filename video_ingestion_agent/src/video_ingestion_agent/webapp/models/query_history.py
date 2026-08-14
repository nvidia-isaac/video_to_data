# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Query history data models."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ClipResult:
    """Individual clip from a query result."""

    clip_id: str
    video_path: str
    start_time: float
    end_time: float
    description: str
    confidence: float
    action: str | None = None
    object_name: str | None = None
    extracted_path: str | None = None

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "clip_id": self.clip_id,
            "video_path": self.video_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "description": self.description,
            "confidence": self.confidence,
            "action": self.action,
            "object_name": self.object_name,
            "extracted_path": self.extracted_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClipResult":
        """Create from dictionary."""
        return cls(
            clip_id=data.get("clip_id", ""),
            video_path=data.get("video_path", ""),
            start_time=data.get("start_time", 0),
            end_time=data.get("end_time", 0),
            description=data.get("description", ""),
            confidence=data.get("confidence", 0),
            action=data.get("action"),
            object_name=data.get("object_name"),
            extracted_path=data.get("extracted_path"),
        )


@dataclass
class SubTaskResult:
    """Result from a single sub-task in the agent pipeline."""

    task_id: int
    description: str
    search_type: str
    target_action: str | None = None
    target_object: str | None = None
    clips_found: int = 0
    analysis: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "search_type": self.search_type,
            "target_action": self.target_action,
            "target_object": self.target_object,
            "clips_found": self.clips_found,
            "analysis": self.analysis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubTaskResult":
        """Create from dictionary."""
        return cls(
            task_id=data.get("task_id", 0),
            description=data.get("description", ""),
            search_type=data.get("search_type", ""),
            target_action=data.get("target_action"),
            target_object=data.get("target_object"),
            clips_found=data.get("clips_found", 0),
            analysis=data.get("analysis", ""),
        )


@dataclass
class QueryRecord:
    """Complete query session record."""

    id: int
    query: str
    project_id: int | None
    timestamp: datetime
    duration_seconds: float

    # Results
    clips: list[ClipResult] = field(default_factory=list)
    sub_tasks: list[SubTaskResult] = field(default_factory=list)
    final_answer: str = ""

    # Agent trace
    working_memory: list[str] = field(default_factory=list)

    # Metadata
    config_used: dict[str, Any] = field(default_factory=dict)

    @property
    def clip_count(self) -> int:
        """Number of clips found."""
        return len(self.clips)

    @property
    def task_count(self) -> int:
        """Number of sub-tasks executed."""
        return len(self.sub_tasks)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "query": self.query,
            "project_id": self.project_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "clips": [c.to_dict() for c in self.clips],
            "sub_tasks": [t.to_dict() for t in self.sub_tasks],
            "final_answer": self.final_answer,
            "working_memory": self.working_memory,
            "config_used": self.config_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryRecord":
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            query=data.get("query", ""),
            project_id=data.get("project_id"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
            duration_seconds=data.get("duration_seconds", 0),
            clips=[ClipResult.from_dict(c) for c in data.get("clips", [])],
            sub_tasks=[SubTaskResult.from_dict(t) for t in data.get("sub_tasks", [])],
            final_answer=data.get("final_answer", ""),
            working_memory=data.get("working_memory", []),
            config_used=data.get("config_used", {}),
        )

    @classmethod
    def from_db_row(cls, row: tuple) -> "QueryRecord":
        """Create from database row."""
        return cls(
            id=row[0],
            query=row[1],
            project_id=row[2],
            timestamp=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
            duration_seconds=row[4] or 0,
            final_answer=row[5] or "",
            working_memory=json.loads(row[6]) if row[6] else [],
            config_used=json.loads(row[7]) if row[7] else {},
            # clips and sub_tasks loaded separately
        )
