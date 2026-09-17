---
name: mesh-to-usd-doctor
description: Diagnose and repair failures in this repository's mesh-to-USD generation, scoped USD validation, recorded-support inference, FoundationPose calibration, catalog batching, or Isaac Sim drop tests. Use when Docker, GPU, EULA, weights, input contracts, CoACD collision generation, USD references, validation, scale, mesh hashes, contact, settling, standing, videos, reports, resume, or batch status are missing, inconsistent, stalled, or failing.
---

# Mesh-to-USD Doctor

Inspect evidence first and ask later. Work from `reconstruction/`, preserve the
job directory and videos, and diagnose the first bad stage rather than
rebuilding or rerunning everything.

## Discover context automatically

1. Read the exact command and error from the conversation or available log.
2. If no output directory is named, inspect the newest relevant directory
   under `data/outputs/mesh_to_usd/` and paths referenced by recent commands.
3. Infer the input mode from `mesh_to_usd_workflow_report.json` and command:
   standalone asset, exact sequence, cross-mesh support, or batch.
4. Infer the first incomplete stage from reports and timestamps: support,
   generation, validation, or drop test.
5. Ask one focused question only when neither input nor output can be found.

## Run the fast diagnostic set

```bash
git rev-parse --short HEAD
python --version
docker version
nvidia-smi
df -h .
docker ps -a --no-trunc
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py --help
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_drop_test.py --help
```

Inspect, when present:

```text
mesh_to_usd_workflow_report.json
foundation_pose_support/foundation_pose_support_report.json
foundation_pose_support/pose_tracking_metadata.json
recorded_support_pose.json
generation_report.json
simready_validation_report.json
drop_test/drop_test_result.json
mesh_to_usd_batch_report.json
*/batch_job_report.json
```

Check nonempty files, hashes, timestamps, referenced paths, active processes,
containers, GPU activity, and disk growth. Separate observed evidence, root
cause, repair, and resume command.

## Classify the first failure

| Evidence | Classification | Narrow repair |
|---|---|---|
| Docker unavailable or GPU invisible in generator/drop container | setup/runtime | Repair Docker or NVIDIA Container Toolkit; rerun the scoped GPU probe |
| EULA rejection | authorization | Ask for explicit acceptance or `ACCEPT_EULA=Y`; never set it implicitly |
| missing generator/validator image | setup | Use `mesh-to-usd-setup` and build only the missing target |
| malformed standalone or sequence contract | input | Fix the exact missing asset, pose, ground-plane, H5, EDEX, symmetry, or metadata file |
| no stable initial pose window | recorded support | Inspect the initial prefix and pose consistency; do not search later frames or fall back automatically |
| FoundationPose registration/tracking fails | cross-mesh support | Inspect target scale, masks, camera inputs, registration attempts, and support report; retry only with evidence |
| target/support mesh hashes differ | provenance | Regenerate support and USD from the same target bytes; never edit the report to match |
| implausible `extents_m` | source scale | Correct the source mesh scale or use an explicitly approved scale transform, then regenerate |
| CoACD/collider failure | geometry | Inspect source topology, decomposition report, and tiny-hull filtering before changing decomposition limits |
| missing `visual_asset.usd` or texture dependency | package | Restore/regenerate the complete adjacent USD package; do not publish the entry point alone |
| scoped validator fails | USD structure | Repair the named default-prim, units, rigid-body, mass, hierarchy, collider, scale, or purpose requirement |
| object appears black or transparent | material | Inspect source alpha/transmission and `visual_material_normalization`; preserve textures while correcting invalid material semantics |
| drop asset does not load or falls through | USD/collision | Inspect composition, authored collider APIs, ground contact, scale, and physics scene |
| contact succeeds but never settles | simulation/threshold | Inspect pose and velocity histories and video before changing time or settle thresholds |
| core physics passes but standing fails | orientation/geometry | Inspect the exact pose segment and support surface; do not relabel it PASS or disable standing without user policy |
| batch says `GENERATED-WITHOUT-DROP-TEST` | missing support | Report generation-only explicitly; supply a complete support sequence rather than claiming a pass |

## Detect a stall correctly

Check the host wrapper PID, active container, GPU/CPU use, report changes,
artifact sizes, and timestamps. A quiet Isaac Sim or FoundationPose log alone
is not a stall. Report which signal stopped changing and for how long.

## Repair and re-verify

Prefer the smallest reversible repair. Do not delete results, weaken validation,
change scale or physics parameters without recording it, bypass mesh hashes,
or switch support policies silently. After repair:

1. Rerun the first failed probe or stage with the original input and output.
2. Confirm its expected report is complete and internally consistent.
3. Resume downstream work with unchanged provenance.
4. Return to `mesh-to-usd-run` for final report and video inspection.

If the evidence shows a product defect, preserve the minimal reproducer and
report the revision, exact command, input mode, first bad artifact, container
images, hardware, logs, expected behavior, and observed behavior.
