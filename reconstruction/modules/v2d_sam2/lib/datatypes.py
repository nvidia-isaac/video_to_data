from dataclasses import dataclass

from v2d.common.datatypes import BoundingBox, Point


@dataclass
class Sam2Prompt:
    frame_index: int
    object_id: int
    points: list[Point] = None
    point_labels: list[int] = None
    box: BoundingBox | None = None
    
    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "object_id": self.object_id,
            "points": [p.to_dict() for p in self.points] if self.points else None,
            "point_labels": self.point_labels if self.point_labels else None,
            "box": self.box.to_dict() if self.box else None
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'Sam2Prompt':
        return Sam2Prompt(
            frame_index=d["frame_index"], 
            object_id=d["object_id"], 
            points=[Point.from_dict(p) for p in d["points"]] if d.get("points") else None, 
            point_labels=d.get("point_labels") if d.get("point_labels") else None,
            box=BoundingBox.from_dict(d["box"]) if d.get("box") else None
        )


@dataclass
class Sam2Prompts:
    prompts: list[Sam2Prompt]
    
    def to_dict(self) -> dict:
        return {
            "prompts": [p.to_dict() for p in self.prompts]
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'Sam2Prompts':
        return Sam2Prompts(prompts=[Sam2Prompt.from_dict(p) for p in d["prompts"]])
