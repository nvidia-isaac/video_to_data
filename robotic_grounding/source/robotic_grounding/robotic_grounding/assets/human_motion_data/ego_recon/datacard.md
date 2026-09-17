---
license:
- cc-by-4.0
task_categories:
- robotics
tags:
- other
- 3d
- image
- tabular
- timeseries
- robotics
- manipulation
- reconstruction
- retargeting
- demonstration
- human
- synthetic
- reinforcement-learning
size_categories:
- n<1K
pretty_name: Ego Reconstruction Retargeting Sample
---

## Dataset Description: <br>

The Ego Reconstruction Retargeting Sample contains one tissue-box manipulation sequence reconstructed from an internally recorded NVIDIA egocentric video and retargeted to two robot embodiments: the floating-hand `sharpa_wave` embodiment and the fixed-base whole-body `vega_sharpa` embodiment. The package includes retargeted motion trajectories, a textured tissue-box model and rigid URDF, and embodiment-specific reconstructed support geometry for simulation replay.

All released files in this package are generated or derived by the reconstruction and robotic-grounding code in the V2D repository. The upstream MP4 is not included and is covered by a separate datacard. The two trajectory records represent the same source demonstration in different embodiment-specific schemas and sampling rates; they are not two independent human demonstrations.

This dataset is ready for commercial or non-commercial uses.

## Dataset Owner(s): <br>

NVIDIA Corporation

## Dataset Creation Date: <br>

July 30, 2026. This is the version-control date on which the complete two-embodiment package was first present.

## Versioning:

v0.3 <br>

Previous Version(s): No previous version <br>

## License/Terms of Use: <br>

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## Intended Usage: <br>

This sample is intended for robotics developers and researchers using the V2D and Robotic Grounding pipelines. It can be used to:

- inspect and replay a reconstructed human-object interaction;
- validate object reconstruction, support-surface reconstruction, and hand or whole-body retargeting;
- train and evaluate reinforcement-learning expert policies for the included floating-hand and Vega Sharpa embodiments.

The package is a small reproducibility and workflow-validation sample. It contains one source demonstration and is not intended to represent the diversity of human manipulation strategies, environments, objects, or robot tasks.

## Dataset Characterization <br>

**Data Collection Method** <br>

* Manually-Collected - The upstream source is an egocentric video of a human tissue-box manipulation performed and recorded internally at NVIDIA. The raw video is excluded from this package and will have a separate datacard. <br>

**Labeling Method** <br>

* Hybrid: Automated, Manually-Labeled - V2D performs automated hand and object reconstruction, support-surface recovery, contact processing, and robot retargeting. Human reviewers inspect the generated results and apply corrections when required, including scene-alignment and quality corrections. <br>

## Dataset Format <br>

The dataset has one source sequence, `tissue_box_simple`, with shared object assets and two embodiment-specific retargeted trajectories.

| Property | Value |
|---|---|
| Dataset layout | `processed/` contains the generated object assets and partitioned motion trajectories; `reconstructed_stage/` contains support geometry used during replay and simulation |
| Floating-hand trajectory | `processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave/*.parquet` is one Parquet row containing 390 frames at 30 Hz (13.00 seconds) and 62 fields |
| Floating-hand features | MANO hand shape and pose, wrist poses, 22 finger joints per hand, robot frame poses, per-frame IK diagnostics, hand-object contact geometry, and object poses |
| Vega trajectory | `processed/sequence_id=tissue_box_simple/robot_name=vega_sharpa/data.parquet` is one `motion_v1` Parquet row containing 649 frames at 50 Hz (12.98 seconds) and 42 fields |
| Vega features | A 58-joint whole-body trajectory, two end-effector poses, per-hand frames and finger joints, contact activity and geometry, IK diagnostics, and object poses |
| Vega coordinate frame | `robot_base_z_up`, as declared in the Vega Parquet record |
| Object model | `tissue_box_simple.obj`, `tissue_box_simple_material_0.png`, and `tissue_box_simple.mtl` provide textured visual geometry |
| Rigid simulation asset | `tissue_box_simple_visual.stl` and `tissue_box_simple_rigid.urdf` provide mesh geometry and a single-link rigid-body definition; the URDF uses the STL for both visual and collision geometry |
| Floating-hand support geometry | `reconstructed_stage/tissue_box_simple_support.usda` contains a metric, Z-up collision surface in the floating-hand placement world |
| Vega support geometry | `reconstructed_stage/tissue_box_simple_vega_sharpa_support.usda` contains the corresponding collision surface in the Vega placement world |
| Source media | The internally recorded raw MP4 and intermediate reconstruction bundle are not included |
| Data splits | No train, validation, or test split is defined |

The two Parquet files use different embodiment-specific schemas and should not be concatenated without explicit schema normalization. Their stored source and asset references were generated in absolute pipeline paths; consumers moving the files outside the V2D asset tree may need to re-root those references. Large generated assets are distributed through Git LFS and should be hydrated before use.

## Dataset Quantification <br>

| Measurement | Exact value |
|---|---:|
| Source human demonstrations | 1 |
| Retargeted trajectory records | 2 |
| Embodiment-specific frames | 1,039 total: 390 floating-hand frames and 649 Vega frames |
| Canonical tracked files | 9 |
| Floating-hand Parquet fields | 62 |
| Vega Parquet fields | 42 |
| Vega robot joints | 58 |
| Textured OBJ geometry | 23,714 vertices and 38,662 faces |
| Texture images | 1 RGBA PNG at 1024 × 1024 pixels |
| Reconstructed support assets | 2 USDA files |
| Total declared storage | 9,544,183 bytes (approximately 9.10 MiB) |

The frame total counts both embodiment-specific representations of the same source interaction and must not be interpreted as 1,039 independent observations. Ignored workspace caches such as `.cache/`, `.mypy_cache/`, and `.pytest_cache/` are not part of the canonical dataset and are excluded from these counts.

## Reference(s): <br>

- [NVIDIA V2D `video_to_data` repository](https://github.com/nvidia-isaac/video_to_data)

## Ethical Considerations: <br>

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

The raw egocentric video is excluded from this package. The released trajectories retain derived MANO hand shape and motion parameters; participant consent for releasing these derived values has been confirmed, and the package has been confirmed to contain no sensitive personal data. These data should nevertheless be treated as derived human-motion data.

This sample contains a single interaction, object, capture setting, and human performance. It cannot characterize population-level variation or the full range of valid manipulation strategies. Monocular reconstruction, automated contact processing, retargeting, and manual correction can introduce geometric, temporal, contact, and embodiment-specific errors. Users should visually and quantitatively validate trajectories before using them for policy evaluation, physical-robot deployment, or safety-critical decisions.
