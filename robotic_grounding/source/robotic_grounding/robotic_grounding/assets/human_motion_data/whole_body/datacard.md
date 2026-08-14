```yaml
---

# Tentative
license: CC-By-4.0

task_categories: robotics

tags:
  - video
  - depth
  - human
  - object
  - reconstruction
  - locomotion
  - manipulation
  - demonstration

size_bin: 1<n<10

---
```

## Dataset Description: <br>
This sample dataset contains three post-processed examples from the HOI dataset, retargeted to the G1 robot.
These examples serve as quick-start demonstrations for third-person-view whole-body loco-manipulation training,
allowing users to test-run the code release. It contains the following elements:
- USD files of reconstructed supporting surfaces, generated using the reconstruct_support_surfaces.py script in this repo.
- Object asset files including .obj, .urdf, .png and .mtl, generated from the raw HOI dataset using the reconstruction pipeline in V2D.
- Parquet files containing the retargeted G1 motion data from the raw HOI dataset.

<br>

This dataset is ready for commercial or non-commercial uses.

## Dataset Owner(s): <br>
NVIDIA Corporation

## Dataset Creation Date: <br>
July 2026 <br>

## Version:
0.2.0 <br>

## License/Terms of Use: <br>
[Creative Commons Attribution 4.0 International (CC BY-4.0)](https://creativecommons.org/licenses/by/4.0/) <br>

## Intended Usage: <br>
This dataset is ideal for training HOI reconstruction models like [CARI4D](https://nvlabs.github.io/CARI4D/). It can also be used for grounding during robot policy training.

## Dataset Characterization <br>
** Data Collection Method<br>
* Manually-Collected - The HOI sequences are performed and recorded by human operators. <br>

** Labeling Method<br>
* Hybrid: Manually-Labeled, Automated - HOI sequences are ingested and processed by an automated pipeline to generate pose annotations. Human annotators label segments where pose annotations do not meet standards. <br>

## Dataset Format <br>
| Property | Value |
|----------|-------|
| Sequence layout | `soma/sequence_id=<sequence_id>/` contains one retargeted action sequence and its associated object assets |
| Retargeted robot motion | `robot_name=g1/data.parquet` stores one G1 action trajectory per sequence, including `fps`, `coord_frame`, `robot_joint_names`, `robot_root_position`, `robot_root_wxyz`, and `robot_joint_positions` |
| End-effector and hand poses | `data.parquet` columns `ee_link_names`, `ee_pose_w`, `hand_sides`, `hand_frame_names`, `hand_frames_w`, `hand_finger_joint_names`, and `hand_finger_joints` store per-frame Cartesian hand/end-effector targets and finger joints |
| Object metadata and poses | `data.parquet` columns `object_name`, `object_body_names`, `object_mesh_paths`, `object_urdf_paths`, `object_root_position`, `object_root_axis_angle`, `object_body_position`, and `object_body_wxyz` store object identity, asset references, and per-frame rigid-body poses |
| Contact annotations | `data.parquet` columns `hand_contact_link_names`, `hand_link_contact_positions`, `hand_link_contact_normals`, `hand_object_contact_positions`, `hand_object_contact_normals`, `hand_object_contact_part_ids`, and `hand_contact_active` store per-frame hand-object contact information |
| Source and IK diagnostics | `data.parquet` columns `source_dataset`, `raw_motion_file`, `source_kind`, `source_payload`, `source_joint_names`, `ik_error_per_frame`, `ik_num_iterations`, and `frame_task_errors` record provenance and retargeting quality metrics |
| Object assets | `object/textured_mesh.obj`, `object/textured_mesh.urdf`, `object/material.mtl`, and `object/material_0.png` provide sequence-local textured object geometry and URDF references when available |
| Reconstructed support surfaces | `reconstructed_stage/<sequence_id>_support.usda` stores reconstructed table, chair, or other support geometry for replay and simulation when available |

## Dataset Quantification <br>
3 action sequences, each containing retargeted G1 motion data and object or support assets where available. <br>
In current storage formats, the folder is approximately 90MB.

## Reference(s): <br>
All the reconstruciton and retargeting code is in the [video_to_data repo](https://github.com/nvidia-isaac/video_to_data/tree/main).

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).   