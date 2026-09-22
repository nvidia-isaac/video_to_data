# FlashCHORD v0.4 integration boundary

This directory is a native shallow port of FlashCHORD. It keeps its own package namespace, lockfile, Docker image,
Hydra configuration, scripts, tests, and OSMO templates. It does not import `robotic_grounding`, and the existing
Isaac Lab manager does not import it. [SOURCE.json](SOURCE.json) records the exact source and patch provenance.

## Repository boundary

| Concern | FlashCHORD lane | Existing robotic-grounding lane |
| --- | --- | --- |
| Python | `flash_chord` from this directory's `src/` | `robotic_grounding` from `../source/` |
| Dependencies | this directory's `uv.lock` | `../pyproject.toml` and Isaac Lab image |
| Container | `docker/Dockerfile` | `../workflow/Dockerfile` |
| Commands | this directory's native scripts | existing `scripts/rsl_rl/*` commands |
| Config | this directory's Hydra groups | existing task/agent config |
| Outputs | native PPO/FlashSAC checkpoints and evaluation JSON | existing RSL-RL outputs |

The manager `.dockerignore` excludes this directory, so adding FlashCHORD does not change the Isaac Lab image.
The manager pre-commit configuration also excludes it. This directory retains its native pre-commit configuration,
with its paths scoped for the monorepo location, while the separate Flash CI gate builds the pinned native image and
runs the full test suite.

## Motion and artifact compatibility changes

The changes below are Flash-side adapters for artifacts produced by V2D main at integration commit
`19ae526a2c90841739abc76543a6329f5eb91ecb`. They do not change V2D's motion schema or producers.

Current main actively uses two training-data containers. Floating Sharpa retargeters write the unversioned
`ManoSharpaData` format. Dexmate/Vega and G1 writers use `motion_v1/single_robot`. The port consumes those two
current contracts by name; it does not introduce a separate legacy data path.

| Input contract | Port behavior | What is unchanged |
| --- | --- | --- |
| Unversioned `ManoSharpaData` from current floating-Sharpa retargeters | Dispatch to the native floating-hand reference adapter. | V2D's wrist, finger, object, contact, and named-frame fields remain authoritative. |
| `motion_v1/single_robot` `ee_link_names` + `(T,E,7)` `ee_pose_w` | Loaded first on Flash's named-frame axis. | Pose order is `[xyz,wxyz]`; no coordinate or quaternion conversion. |
| Optional `hand_sides`-indexed `hand_frame_names` + `(T,K_side,7)` `hand_frames_w` | Validate each side and append the frames after the EE frames, preserving declared side and frame order. | Robot joints remain authoritative; redundant per-side finger-joint copies are not consumed. |
| Vega/Dexmate files containing all ten `left/right_<digit>_DP` frames | Validate recorded DP poses against Flash Vega-v2 DP forward kinematics. | The v0.4 palm + true-fingertip validation remains preferred when that complete set exists. |
| Serialized Vega DP orientations | Native DP error tolerance is `1e-3` rad, just above the measured current-artifact maximum of `9.7657e-4` rad. | Position tolerance stays `5e-4` m; the v0.4 palm/fingertip orientation tolerance stays `1e-4` rad. |
| Per-side contacts `(T,K_side,3)` and part IDs `(T,K_side)` | Validate each side independently, allowing schema-valid `K_left != K_right`. | Time length, finite values, unit active normals, integral IDs, and object-body ID range checks remain strict. |
| Producer-absolute paths containing `/assets/` | Preserve an existing path; otherwise reroot the suffix against the physical Parquet's nearest `assets` ancestor, then use Flash's historical asset fallback. | Paths with no known marker are not rewritten. |
| Current articulated-object layout `object_assets/meshes/<dataset>/<object>/<body>` with no per-body URDF list | Resolve sibling `object_assets/urdfs/<dataset>/<object>.urdf` for current ARCTIC/Synthbox data, then bind the declared bodies and `rotation` joint. | Unknown datasets and malformed layouts still fail closed; the bundled ARCTIC URDF remains a fallback. |
| Support surfaces | Prefer colocated `<sequence>_<robot>_support.usda`, then `<sequence>_support.usda`, then the historical Flash asset tree. | Support remains optional unless the selected scene config requires it. |

`motion_v1/dual_hand` is intentionally not added. V2D declares that schema variant, but no current production
writer emits it and none of the five released Flash recipes uses it. Add such an adapter only with a real supported
producer artifact and an end-to-end cohort requirement.

V2D also permits objectless `motion_v1/single_robot` for robot-only tasks. FlashCHORD's five released recipes are
manipulation tasks whose scene, command, observations, and objectives require a tracked object, so the port does not
claim robot-only support. Every Dexmate/G1 cohort in the release verification has an object.

## Local data visibility

From this directory, `docker/run.sh start` automatically detects the sibling V2D assets tree and mounts it
read-only at `/v2d/assets`. Override the source with `V2D_ASSETS_DIR=/absolute/path/to/assets`; the mounted root
must contain `human_motion_data/` so asset paths can be rerooted without changing the Parquet.

Example inside the Flash container:

```bash
PARQUET=/v2d/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=vega_sharpa
python scripts/debug/replay.py "task.parquet='${PARQUET}'" \
  embodiment=dexmate_sharpa action=residual_joint_position \
  collision=manipulation collision.robot_support_collision=true
```

OSMO inputs must expose the same Parquet-plus-assets closure inside the submitted image or a mounted dataset. The
native workflow templates are unchanged and continue to accept an explicit `parquet` value.

## Verification at import

- Complete no-corpus suite: `613 passed, 55 skipped`; all skips require optional reference-sequence corpora.
- Every training-ready Parquet checked into current main loads and builds a matching Newton scene: processed
  Synthbox and ego-reconstruction floating Sharpa (`ManoSharpaData`), ego-reconstruction Dexmate/Vega
  (`motion_v1/single_robot`, including ten strict DP validations), and the blue-trash, snack-box, and corn-handover
  G1 motions (`motion_v1/single_robot`). The raw `synthbox_loaded` fixture is correctly rejected until retargeting
  fills its intentionally empty `robot_*` fields.
- All five native recipe scripts compose with real current-main Hive partition paths. Because those paths contain
  `=`, shell commands must quote the Hydra value as `"task.parquet='${PARQUET}'"`.
- Current main does not track a blue-trash support USDA. That checked-in representative scene passes with
  `scene.support=false`; snack and corn pass with their tracked support assets.

These are import/scene gates, not the final training-reproduction gate. The release cohort still requires bounded
train/save/resume/evaluate smokes and the signed-off full Sharpa, Dexmate/Sharpa, and G1 training runs.
