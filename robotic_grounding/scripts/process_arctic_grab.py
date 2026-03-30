from pathlib import Path

import numpy as np
import trimesh
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import ManoSharpaData, list_sequence_ids
from robotic_grounding.retarget.retarget_utils import (
    DEFAULT_PARTITION_COLS,
)

DATASET_DIRS: dict[str, str] = {
    "arctic": "arctic_processed",
}

if __name__ == "__main__":
    input_dir = HUMAN_MOTION_DATA_DIR / DATASET_DIRS["arctic"]
    available = list_sequence_ids(str(input_dir))
    grab_sequences = [sequence_id for sequence_id in available if "grab" in sequence_id]
    print(f"Found {len(grab_sequences)} sequences in {input_dir}")
    for sequence_id in grab_sequences:
        print(sequence_id)

        data = ManoSharpaData.from_parquet(
            str(input_dir),
            filters=[("sequence_id", "=", sequence_id)],
        )

        data.sequence_id = data.sequence_id.replace(
            data.object_name, f"rigid_{data.object_name}"
        )

        data.mano_right_object_contact_part_ids = (
            np.asarray(data.mano_right_object_contact_part_ids).clip(max=1).tolist()
        )
        data.mano_left_object_contact_part_ids = (
            np.asarray(data.mano_left_object_contact_part_ids).clip(max=1).tolist()
        )

        data.object_body_names = ["object"]

        object_mesh_path = Path(data.object_mesh_paths[0]).parent / "mesh_tex.obj"
        object_mesh = trimesh.load(str(object_mesh_path)).apply_scale(0.001)
        object_mesh.export(str(object_mesh_path.parent / "mesh_tex_tiny.obj"))
        data.object_mesh_paths = [
            str(Path(data.object_mesh_paths[0]).parent / "mesh_tex_tiny.obj")
        ]

        vertices = np.asarray(object_mesh.vertices)
        data.object_mesh_radius = [
            np.linalg.norm(vertices - vertices.mean(axis=0), axis=1).max()
        ]

        data.object_articulation = [0.0] * len(data.object_articulation)

        data.object_body_position = np.asarray(data.object_body_position)[
            :, :1
        ].tolist()

        data.object_body_wxyz = np.asarray(data.object_body_wxyz)[:, :1].tolist()

        data.save_to_parquet(
            str(input_dir),
            partition_cols=DEFAULT_PARTITION_COLS,
        )
