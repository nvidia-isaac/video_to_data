# Mesh to USD

This folder provides one top-level preparation workflow and two downstream
utilities for one reconstructed rigid object:

1. Prepare and validate a physics-enabled USD from a target mesh plus an
   exported HOI support sequence. Exact-sequence and explicitly standalone
   geometry modes remain available.
2. Validate an existing generated USD without Isaac Sim.
3. Run a gravity, contact, and settling drop test in Isaac Sim.

The workflows are deterministic and do not use a VLM, LLM, planner, or visual
auditor.

## Agentic workflow

Repository-local skills provide the same setup/run/doctor workflow for Claude
and Codex. Identical packages are stored under `.claude/skills/` and
`.codex/skills/`:

| Skill | Use it for |
|---|---|
| [`mesh-to-usd-setup`](../../../../.claude/skills/mesh-to-usd-setup/SKILL.md) | Docker/GPU checks, generator and validator images, input contracts, and optional FoundationPose preparation |
| [`mesh-to-usd-run`](../../../../.claude/skills/mesh-to-usd-run/SKILL.md) | Target-mesh plus support-sequence preparation, structural validation, drop testing, explicit standalone fallback, batch processing, resume, and final evidence review |
| [`mesh-to-usd-doctor`](../../../../.claude/skills/mesh-to-usd-doctor/SKILL.md) | Diagnosing the first support, generation, validation, packaging, simulation, or batch failure |

Example prompts:

```text
Use $mesh-to-usd-setup to prepare this checkout for mesh-to-USD generation.

Use $mesh-to-usd-run with target mesh /absolute/path/to/output.glb and support
sequence /absolute/path/to/data_export/sequence, validate the USD, run the
recorded-support drop test on GPU 0, and inspect the report and video.

Use $mesh-to-usd-run to process this exported HOI sequence with recorded
support when its sequence mesh is the intended target asset.

Use $mesh-to-usd-run in explicit standalone geometry-only mode for
/absolute/path/to/output.glb because no support sequence applies.

Use $mesh-to-usd-doctor to diagnose the failed job at /absolute/path/to/job and
give me the narrowest safe resume command.
```

The normal run requires user-specified target-mesh and support-sequence paths;
there is no default object or sequence. The skills use defaults for optional
tuning instead of interviewing the user. They never accept the NVIDIA Isaac
Sim EULA on the user's behalf, silently replace missing recorded support with
a geometry pose, or treat generation-only output as a passing drop test.

## Build

From `reconstruction/`:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/build.py
```

This builds `v2d_hoi_mesh_to_usd` from Isaac Sim 6.0.1 and the separate
`v2d_hoi_mesh_to_usd_validator` image from Python 3.12. Use `--target generator`
or `--target validator` to build only one image.

Cross-mesh recorded-support calibration also uses the FoundationPose image:

```bash
python modules/v2d_foundation_pose/docker/build.py
```

Download its weights once to the repository-standard location:

```bash
python modules/v2d_foundation_pose/docker/run_download_weights.py \
  --output_dir data/weights/foundationpose
```

Review the NVIDIA Isaac Sim EULA, then pass `--accept-eula` to generation and
drop-test commands or set `ACCEPT_EULA=Y`. The standalone validator is CPU-only
and does not use the Isaac Sim image or require EULA acceptance.

## Top-level preparation workflow

For the normal target-mesh plus support-sequence workflow:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py \
  --asset /absolute/path/to/mesh/<object_id>/<method>/output.glb \
  --support-sequence-dir /absolute/path/to/data_export/<sequence_id> \
  --output-dir /absolute/path/to/mesh_to_usd \
  --accept-eula
```

When the sequence mesh itself is the intended target asset:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py \
  --sequence-dir /absolute/path/to/exported_sequence \
  --output-dir /absolute/path/to/mesh_to_usd \
  --accept-eula
```

For an explicitly requested standalone geometry-only fallback:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py \
  --asset /absolute/path/to/merged_recon/output.glb \
  --output-dir /absolute/path/to/mesh_to_usd \
  --accept-eula
```

Standalone preparation does not infer a recorded support orientation and must
not be selected merely because support preparation failed.

The `--asset` and `--sequence-dir` input modes are mutually exclusive. Asset
mode writes the generated USD and reports. Adding `--support-sequence-dir` to
asset mode provides recorded ground and camera observations for support-pose
inference. If the target asset is the exact recorded mesh and the sequence has
`poses.npy`, those poses are used directly. Otherwise, the workflow does not
require existing object poses: it runs multi-view FoundationPose with the
target mesh over the initial search prefix, then finds the first stable window
directly in the new target-mesh poses. Those poses, rather than an assumed
shared mesh frame or a reference-mesh pose, determine the target's local-up
vector.

Cross-mesh calibration requires the complete exported-sequence and target-mesh
contracts described below, plus FoundationPose weights. The weights default to
`data/weights/foundationpose`; use `--foundation-pose-weights-dir` to override
that location. A summary-only export is insufficient.
FoundationPose runs only through the bounded initial search prefix (up to 90
frames by default), not through the full recording. For shorter recordings,
the automatic prefix is clamped to the available frame count; an explicit
out-of-range `--frame-end` remains an error. Target registration is attempted
on source frames 0, 5, 10, 15, 20, and 25 by default, bounded so that a full
stable-window check remains after initialization. A candidate is accepted only
after FoundationPose maintains valid mask overlap for the complete stability
window; a registration that immediately loses the target is retried rather
than accepted as a frozen last-pose stream. If every eligible attempt fails,
the workflow stops; it does not substitute the recording mesh pose. Missing
inputs, tracking failure, inconsistent target poses, or no stable target window
in that prefix also stop support-pose preparation. There is no dependency on
the reference `poses.npy`, ICP, identity-frame assumption, or geometry-pose
fallback. Retry is an HOI mesh-to-USD policy: each attempt invokes the general
FoundationPose interface once with a source slice
`[frame_start, frame_end_exclusive)`.

The calibration writes `foundation_pose_support/poses.npy` and
`foundation_pose_support/pose_tracking_metadata.json`, plus
`foundation_pose_support/foundation_pose_support_report.json`. The tracking
metadata records the successful source-frame range, so a pose suffix is never
misreported as starting at frame 0. It also records all configured camera
names, the cameras that contributed to registration, the highest-visibility
registration camera, registration visibility ratios, and per-camera
tracking-frame counts. The support report records the retry candidates and
outcome of every attempt. FoundationPose still fuses every camera that passes
the visibility threshold; the highest-visibility field does not imply
single-camera tracking. The support JSON preserves that camera provenance
together with hashes for the target mesh and target poses, target symmetry
sidecar, ground plane, and FoundationPose report. Legacy support JSONs that
declare `shared-output-aligned-frame` are rejected. The drop test continues to
verify the target mesh hash against the generated USD.

Sequence mode infers support directly from its recorded mesh and writes
`recorded_support_pose.json`. If no stable window exists in the initial search
prefix, the command exits before launching the USD conversion and never falls
back to a geometry-derived pose.

All preparation modes write `mesh_to_usd_workflow_report.json`, which records
the selected input mode and paths to the generated outputs. Recorded-support
modes verify that the exact target-mesh hash in `generation_report.json` matches
the target hash in `recorded_support_pose.json` before publishing the support
JSON.

### Sequence folder contract

The required sequence content depends on how the support pose is obtained.
Directory and file names are exact.

Recorded-pose mode (`--sequence-dir`) requires:

```text
exported_sequence/
├── object_mesh/
│   └── output_aligned.glb
├── poses.npy
└── ground_plane.json
```

The same contract is sufficient for `--asset` plus `--support-sequence-dir`
when the asset is byte-identical to `object_mesh/output_aligned.glb` and
`poses.npy` is present. `poses.npy` is authoritative and must contain an
`(N, 4, 4)` pose array. The initial search prefix must contain a stable window.

Cross-mesh FoundationPose mode requires:

```text
exported_sequence/
├── edex
├── images/
│   └── <camera_name>.h5
├── depth/
│   └── <camera_name>.h5
├── object_masks/
│   └── <camera_name>.h5
└── ground_plane.json
```

`edex` is a file. By default, each modality directory contains one H5 frame
source per camera. The mesh-to-USD adapter supplies
`foundation_pose_hoi_h5_paths.yaml` while calling FoundationPose's standard
`camera_params_path`, `rgb_dir`, `depth_dir`, and `mask_dir` interface. Use
`--foundation-pose-config-path` to provide different path templates without
changing the FoundationPose input interface.
The target asset is supplied separately and requires this sibling layout:

```text
target_mesh/
├── output_aligned.glb
└── output_symmetry.json
```

Cross-mesh mode does not require the sequence's `poses.npy` or reference
`object_mesh/`: FoundationPose generates target-mesh poses directly. An exact
mesh without usable recorded poses also follows this contract.

Batch discovery additionally requires this file at the sequence root:

```text
exported_sequence/
└── hoi_metadata.yaml
```

Its `object.id` links the sequence to the catalog mesh directory. It is not
required by the direct single-sequence command.

## Generate a rigid USD from a GLB directly

The lower-level GLB converter remains available when orchestration is handled
elsewhere:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd.py \
  --asset /absolute/path/to/merged_recon/output.glb \
  --output-dir /absolute/path/to/mesh_to_usd \
  --accept-eula
```

The command writes:

- `rigid_object.usd`: meter-scale, Z-up rigid object with explicit CoACD
  colliders, mass, center of mass, collider-derived inertia, and a physics
  material.
- `visual_asset.usd`: normalized visual geometry referenced by the rigid USD.
- `generation_report.json`: source, geometry, collider, mass, inertia, material,
  and provenance details.
- `simready_validation_report.json`: scoped profile, feature results, failed
  requirements, issue details, and exact validator/Foundation versions.

`rigid_object.usd` is the package entry point, not a standalone file. Keep it,
`visual_asset.usd`, and any generated texture directories together when copying
or publishing the asset. The entry point may be renamed to `output.usd` as long
as its adjacent dependencies are preserved.

Validation runs by default after generation and fails the host command when the
scoped profile does not pass. Use `--no-validate` only when validation is
intentionally deferred.

This workflow targets opaque reconstructed objects. Some scanners export glTF
materials with `KHR_materials_transmission=1` even when their color texture is
opaque, which makes the object appear black in the Isaac Sim evidence video.
The generator therefore sets positive transmission on imported glTF MDL
materials to zero while preserving their textures and other material inputs.
The applied changes are recorded in
`generation_report.json.visual_material_normalization`.

The input mesh must already have physical scale. USD `metersPerUnit` metadata is
honored; non-USD mesh units are interpreted as meters by the converter. This
workflow does not infer real-world dimensions. Check
`generation_report.json.geometry_metrics.extents_m` before treating the asset as
physically ready.

The default mass is `0.3 kg` to match the existing grounding URDF fallback. It
is recorded as `assumed_grounding_default`, not as measured object metadata.
Use `--mass-kg <value>` when a known mass is available; that value is recorded
as `command_line`. Inertia is always recomputed from the generated collider
geometry for the selected mass.

## Validate an existing rigid USD

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_validate_usd.py \
  --asset /absolute/path/to/mesh_to_usd/rigid_object.usd \
  --output-dir /absolute/path/to/mesh_to_usd
```

The custom `V2D-HOI-Rigid-Object` profile checks only the contract of this
workflow:

- supported USD file type, default prim, and composed imageable geometry;
- Z-up, `metersPerUnit = 1`, and authored kilogram units;
- a rigid body with mass and valid rigid-body hierarchy;
- mesh collider API placement, collider scale, and invisible guide purpose.

It intentionally does not require SDF collision approximation, grasp
identifiers, articulation, material conformance, or any LLM/VLM output. Passing
this profile is a scoped structural validation, not SimReady certification.
Runtime contact and settling remain the responsibility of the drop test.

The validator image pins `simready-validate`, Asset Validator, USD Profiles,
OpenUSD, and the SimReady Foundation source revision. It builds the Foundation
requirement package once in the image, then loads those precompiled rules at
runtime instead of reparsing all Foundation specification Markdown.

## Run the drop test

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_drop_test.py \
  --asset /absolute/path/to/mesh_to_usd/rigid_object.usd \
  --output-dir /absolute/path/to/drop_test \
  --accept-eula
```

The command preserves authored rigid-body physics and mass. For an input asset
without an authored mass, `--fallback-mass-kg` defaults to `0.3`.

The low-level `run_drop_test.py` default initial-orientation policy is
`--initial-pose principal-6`, and the final standing check is enforced by
default. This low-level default is only for explicitly standalone geometry
testing. The normal target-mesh plus support-sequence workflow passes
`--initial-pose recorded-support` and its generated support JSON explicitly.
Use `--initial-pose as-authored` and/or `--no-fail-if-not-standing` only when
that alternate policy is intentional.

`principal-6` computes three principal axes from the visible mesh vertices and
tests both signs of each axis as world +Z. This produces exactly six candidates.
For deterministic yaw, the widest remaining principal axis is aligned with
world +X. All six tests run sequentially in one Isaac Sim process with an
independent scene and report entry. The aggregate test passes when at least one
candidate satisfies the core rigid-body checks and, when enabled, the standing
requirement.

Use `--initial-pose principal-6-support` to refine each of those six coarse
poses using the authored collision mesh when one is present, so the fitted
surface matches the geometry that PhysX will contact. Assets without authored
colliders fall back to visible mesh faces. The refinement selects the largest
approximately coplanar, downward-facing patch in the lower part of that mesh,
fits a single plane to its triangles, and levels that plane before the drop.
Corrections are limited to 30 degrees, and a patch must cover at least 1% of the
squared maximum object extent. Smaller incidental faces and poses with no
suitable patch fall back to unmodified PCA. This mode does not require symmetry
annotations or another geometry dependency.

For the normal workflow, supply the target reconstruction and an exported HOI
sequence in which the object begins in a stable supported pose. The workflow
tracks that exact target mesh with FoundationPose when it differs from the
sequence mesh, combines the target-specific poses with the fitted ground
plane, and infers one recording-supported orientation instead of searching
geometry candidates. The top-level preparation command generates both
required outputs:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py \
  --asset /absolute/path/to/target_mesh.glb \
  --support-sequence-dir /absolute/path/to/exported_sequence \
  --output-dir /absolute/path/to/mesh_to_usd \
  --accept-eula

python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_drop_test.py \
  --asset /absolute/path/to/mesh_to_usd/rigid_object.usd \
  --output-dir /absolute/path/to/drop_test \
  --initial-pose recorded-support \
  --recorded-support /absolute/path/to/mesh_to_usd/recorded_support_pose.json \
  --accept-eula
```

When target and sequence meshes are byte-identical and sequence `poses.npy`
exists, those recorded poses are authoritative. Otherwise the complete
cross-mesh camera contract is required and the target-specific FoundationPose
poses are authoritative; mask visibility is not treated as pose confidence.
By default, inference scans 30-frame windows in chronological order within the
first 90 target-pose frames and uses the first window that passes every
structural and stability check. It never scans later parts of the recording.
This assumes the object begins the recording resting on the ground or another
horizontal support surface. If tracking fails or the initial prefix contains
no stable window, inference reports an error and exits without a fallback.

`--stable-window-frames` and `--initial-search-frames` configure the automatic
search. `--frame-start` and `--frame-end` may be supplied together as an
explicit inclusive/exclusive override; supplying only one is an error.
Inference also fails on missing inputs, malformed selected poses, or
inconsistent rotations. The output records the selection policy and searched
frame range.
The drop test verifies that the recording mesh's exact file hash matches the
source mesh recorded in the generated USD. It never falls back to a
principal-axis pose. Conversely, geometry modes reject a recorded-support
input instead of silently using it.

`infer_recorded_support.py` remains available as a lower-level diagnostic when
only the support JSON is needed. Normal sequence preparation should use
`run_mesh_to_usd_workflow.py` so inference, conversion, and cross-output hash
validation are one operation.

## Batch catalog processing

Use the batch wrapper when catalog meshes and complete exported sequences are
available as local directories or mounted inputs. Object directories are joined
only by the exact `object.id` in `hoi_metadata.yaml`. Each matched Einstein,
BundleSDF, and SAM3D `output_aligned.glb` is processed independently.

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_batch.py \
  --mesh-root /absolute/path/to/mesh \
  --sequence-root /absolute/path/to/data_export \
  --output-root /absolute/path/to/results \
  --gpu-device 0 \
  --accept-eula \
  --dev
```

The batch selects `output_aligned.glb` by default. To process the canonical
un-aligned catalog mesh explicitly, pass `--mesh-filename output.glb`. Its
adjacent `output_symmetry.json` must describe symmetries in the same raw mesh
frame; do not reuse aligned-frame symmetry transforms without converting them.

Support input is selected independently for each object and reconstruction
method. An exact recorded mesh may reuse its own `poses.npy`; candidates with
an invalid initial stable window are recorded and the next sequence is tried.
A cross-mesh target instead selects the newest sequence with complete
FoundationPose inputs without reading the sequence's existing object poses.
Its support window is selected later from target-mesh tracking. A tracking or
target-stability failure is reported as `FAILED` and does not silently switch
to a different recording.

If no candidate provides the inputs needed by that target, the USD is still
generated and structurally validated, but its job is explicitly reported as
`GENERATED-WITHOUT-DROP-TEST` and has no MP4. It is not a passing drop-test
result, and there is no automatic geometry-pose fallback. A cross-mesh job also
becomes generation-only when its complete FoundationPose inputs or weights are
missing. Missing alignment sidecars or target meshes are rejected during
discovery; FoundationPose, generation, validation, or executed drop-test errors
are reported as `FAILED`. The batch report records selection and rejected
candidates under `support_selection.<object_id>.<method>`.

The run is resumable by default and writes
`mesh_to_usd_batch_report.json`. Each `<object_id>/<method>/` folder contains
the preparation outputs, strict support JSON, drop-test report, MP4 evidence,
and `batch_job_report.json`. Successful workflow and job reports include
`support_provenance`, containing the selected sequence, support frame range,
pose-calibration method, and camera provenance. Exact recorded-pose support
explicitly marks camera use as not applicable; cross-mesh FoundationPose
support lists its multi-view registration and tracking cameras. Video,
structural validation, final standing, and the drop test are enabled and
required by default. A failed standing test keeps its video for diagnosis but
does not pass the batch job. Resume verifies
content hashes for the target mesh, selected support inputs, FoundationPose
weights/configuration, workflow options, configured generator/validator image
references, and previously written artifacts; changing any of them reruns the
job. A run with no matched mesh/sequence jobs
exits with an error instead of reporting a vacuous pass. Use `--dry-run` to
produce and validate the complete job manifest without launching containers.
The batch restricts mesh generation and drop testing to GPU 0 by default;
select another device for those steps with `--gpu-device`. FoundationPose
retains its module's existing GPU behavior. Direct single-asset commands retain
their existing all-GPU default unless this option is provided.

The mesh-to-USD generation step is shared by both approaches. Only the drop
test's initial scene orientation differs: one recorded candidate versus six
geometry-derived candidates.

Each orientation is applied once before lifting and preserved during the hold;
no correction is applied after release. The six candidates provide broad
coverage for manufactured daily objects, but the workflow does not infer which
pose is semantically upright. Review the corresponding segments of the combined
video when multiple candidates pass. Convex-hull stable-pose ranking is
intentionally out of scope for this version.

The report's `representative_pose_id` identifies the passing candidate with the
best diagnostic rank (standing, rigid-body pass, then lowest tilt). It is not a
semantic pose selection.

The standing diagnostic requires contact, settling, support geometry within
`--ground-tolerance` of the plane (default `0.02 m`), and final tilt within
`--standing-angle` (default `15` degrees).

Settling accepts either the existing consecutive low-velocity check or a
consecutive full-pose stability window of the same `--settle-seconds`
duration. The pose window's maximum pairwise translation is limited to 1% of
the object's largest extent (clamped to `0.001-0.01 m`), and its maximum
pairwise rotation is limited to `2` degrees.
Unlike velocity settling, pose stability does not end the free-drop phase
early; it is evaluated after the full drop and observation period. This
prevents solver-velocity jitter from producing a false failure while ensuring
the generated video ends with a visibly stable object. Configure these bounds
with `--pose-settle-position-ratio` and `--pose-settle-angle`.

Outputs:

- `drop_test_result.json`: aggregate counts and one `pose_results` entry per
  candidate, including load, contact, settling, final pose, and standing
  diagnostics. Each entry records its frame and time range in the combined
  video.
- `drop_test.mp4`: one sequential 30 FPS video containing all six pose tests,
  generated by default. Use `--video-fps` to change the evidence frame rate or
  pass `--no-video` to disable capture for a faster run. The video rate cannot
  exceed `--physics-hz`.

The default lift distance is `0.25` times the object's largest extent rather
than the current candidate's vertical height, so side-on candidates receive the
same controlled perturbation. Override it with `--lift-height-ratio`, or specify
an absolute distance with `--lift-height`. The kinematic lift updates both the
PhysX body pose and the USD root transform, so each video shows the same lift
distance reported in its `pose_results` entry.

Final tilt affects the default command result. Disable that requirement with
`--no-fail-if-not-standing`; this can be useful for rotationally symmetric
objects such as balls. Video remains enabled by default. If the standing
requirement fails, the host command prints a warning with the absolute path to
the combined video. If video was explicitly disabled, the warning instead
explains how to rerun with video enabled.

## Host tests

The command construction and pure geometry/physics helpers can be checked
without launching Isaac Sim:

```bash
pytest modules/v2d_hoi_object_reconstruction/mesh_to_usd/tests
```
