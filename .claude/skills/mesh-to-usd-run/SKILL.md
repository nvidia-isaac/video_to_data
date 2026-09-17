---
name: mesh-to-usd-run
description: Generate, validate, drop-test, resume, and batch-process rigid USD assets from reconstructed object meshes or exported HOI sequences in this repository. Use when a user asks to convert a GLB to USD, create a physics-enabled rigid object, infer a recorded support pose, run an Isaac Sim hold-and-drop test, process a catalog of Einstar, BundleSDF, or SAM3D meshes, resume a partial job, or verify the generated USD, reports, and MP4 evidence.
---

# Run Mesh-to-USD

Use the top-level host wrappers from `reconstruction/`. Unless the user asks for
commands only, run the workflow, monitor it, and verify both machine-readable
reports and visual evidence.

## Resolve the normal input contract

Do not define a repository-default object or sequence. The normal run requires
the user to supply both a target mesh and an exported HOI support sequence for
the same object:

| Value | Default |
|---|---|
| input mode | target `--asset` plus `--support-sequence-dir` |
| target asset | user-supplied reconstructed mesh |
| support sequence | user-supplied complete exported HOI sequence for the same object |
| GPU | `0` |
| output | fresh directory under `data/outputs/mesh_to_usd/` using the input stem and timestamp |
| validation | enabled |
| mass and material | workflow defaults; override mass only when measured metadata is supplied |
| drop video | enabled |
| normal initial pose | `recorded-support` |
| standalone initial pose | `principal-6`, only when explicitly requested |

Use paths explicitly supplied in the request or prior conversation. If either
is unavailable, ask one blocking question for the missing path or paths. Do
not discover and choose an arbitrary local sequence, substitute the included
toy-airplane smoke-test GLB, or silently switch to geometry-only pose
selection.

Create a fresh output directory unless the user explicitly supplied an
existing job to resume. Never delete or overwrite unrelated results.

Do not ask about optional physics tuning before the default run. Ask one
blocking question only when Isaac Sim must launch and neither
`ACCEPT_EULA=Y` nor explicit EULA acceptance is available. Never set EULA
acceptance implicitly.

## Preflight and continue

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/reconstruction"
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py --help
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_drop_test.py --help
```

Use `mesh-to-usd-setup` to repair a missing image, GPU, weight, or input
prerequisite, then return to this run automatically.

## Choose exactly one preparation mode

### Target mesh with support sequence (default)

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py \
  --asset <absolute-target-mesh> \
  --support-sequence-dir <absolute-sequence> \
  --output-dir <absolute-output> \
  --gpu-device 0 \
  --accept-eula
```

Run the generated recording-supported pose:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_drop_test.py \
  --asset <absolute-output>/rigid_object.usd \
  --output-dir <absolute-output>/drop_test \
  --initial-pose recorded-support \
  --recorded-support <absolute-output>/recorded_support_pose.json \
  --gpu-device 0 \
  --accept-eula
```

When the target bytes equal the sequence mesh and recorded poses exist, the
workflow uses those poses directly. Otherwise it tracks the exact target mesh
with FoundationPose over the bounded initial prefix, finds its first stable
window, and combines those poses with the recorded ground plane. If tracking,
input validation, or stable-window selection fails, preserve the failure and
stop. Never reuse reference-mesh poses, ICP, PCA, or a silent geometry
fallback. Describe the result as the recorded support orientation, not as a
semantic upright inference.

### Exact exported HOI sequence

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py \
  --sequence-dir <absolute-sequence> \
  --output-dir <absolute-output> \
  --gpu-device 0 \
  --accept-eula
```

Use its strict recorded support for the drop test:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_drop_test.py \
  --asset <absolute-output>/rigid_object.usd \
  --output-dir <absolute-output>/drop_test \
  --initial-pose recorded-support \
  --recorded-support <absolute-output>/recorded_support_pose.json \
  --gpu-device 0 \
  --accept-eula
```

Use exact-sequence mode only when its `object_mesh/output_aligned.glb` is the
target asset. It is an alternative recorded-support input contract, not a
reason to replace a separately supplied target mesh.

### Explicit standalone geometry fallback

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py \
  --asset <absolute-mesh-path> \
  --output-dir <absolute-output> \
  --gpu-device 0 \
  --accept-eula
```

Use this mode only when the user explicitly requests a standalone or
geometry-only test, or explicitly confirms that no support sequence applies.
It does not establish a recorded or semantic upright orientation. Run its
six-pose geometry test explicitly:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_drop_test.py \
  --asset <absolute-output>/rigid_object.usd \
  --output-dir <absolute-output>/drop_test \
  --initial-pose principal-6 \
  --gpu-device 0 \
  --accept-eula
```

Use `--initial-pose principal-6-support` only when collision-surface leveling
is explicitly requested or needed for standalone diagnosis. Do not present
either geometry mode as the default workflow.

Omit `--accept-eula` from examples above only when `ACCEPT_EULA=Y` is already
set. Add `--dev` only when the user is testing local implementation changes.

## Run a catalog batch deliberately

First validate discovery without launching containers:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_batch.py \
  --mesh-root <absolute-mesh-root> \
  --sequence-root <absolute-sequence-root> \
  --output-root <absolute-output-root> \
  --gpu-device 0 \
  --dry-run
```

Inspect the manifest, matched object IDs, selected mesh filename, methods, and
rejected support candidates. Then rerun without `--dry-run`, preserving the
same arguments and adding EULA acceptance. Resume is enabled by default and
must revalidate content hashes. Distinguish `PASSED`, `FAILED`, and
`GENERATED-WITHOUT-DROP-TEST`; generation-only is not a drop-test pass.

## Prove that the run started

Record the source revision, exact command, input contract, output path, GPU,
image references, and start time. Monitor the wrapper, active container, GPU,
disk, and report timestamps until a real stage starts. A quiet log alone is not
a stall.

Preserve partial output on failure and hand the job to `mesh-to-usd-doctor`.

## Verify completion

Preparation requires:

- nonempty `rigid_object.usd` and adjacent `visual_asset.usd`;
- every referenced texture or package dependency kept beside the entry point;
- `generation_report.json` with the exact source hash and plausible
  `geometry_metrics.extents_m`;
- passing `simready_validation_report.json`; and
- `mesh_to_usd_workflow_report.json` matching the requested input mode.

Recorded-support modes additionally require strict mesh-hash agreement and
support provenance, including the selected frame range and camera source when
FoundationPose was used.

Drop-test completion requires:

- valid `drop_test_result.json` with load, contact, settling, and standing
  diagnostics for every attempted pose;
- nonempty `drop_test.mp4`; and
- visual inspection of lift, release, collision, and final stability.

Do not call the scoped structural profile SimReady certification. Do not infer
physical scale, mass, semantic uprightness, or real-world stability from a
generated file alone. Report a standing failure separately from core
rigid-body/contact behavior; change `--fail-if-not-standing` only when the user
explicitly defines standing as not applicable.
