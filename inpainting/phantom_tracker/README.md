# Phantom hand-tracking condition

This stage runs Phantom's non-EPIC hand stack as a reproducible TACO
condition: `IDEA-Research/grounding-dino-base` at immutable revision
`12bdfa3120f3e7ec7b434d90674b3396eccf88eb` proposes hand boxes and the
official Phantom HaMeR fork at `7f4a703e49278f2002f831d93aed867ad6977281`
regresses MANO pose. That is the HaMeR submodule pin of parent Phantom commit
`a8bb81c1bbe6ade129a1f6f0906482f510354a5e`; they are intentionally different
repositories and commit IDs. The image uses PyTorch 2.7.1/CUDA 12.8 for RTX PRO
6000 Blackwell support.

## Repository boundary

Phantom lives outside `video_to_data_internal`, but no host checkout path is
part of this implementation or any committed artifact. The external checkout
was inspected read-only to identify the intended algorithms and pinned
revisions. Reproduction is instead anchored by immutable upstream identities:

| Component | Revision | How it is used |
|---|---|---|
| Parent `MarionLepert/phantom` | `a8bb81c1bbe6ade129a1f6f0906482f510354a5e` | Provenance anchor for the reviewed pipeline and its HaMeR submodule pin; parent source is not imported at runtime. |
| `MarionLepert/phantom-hamer` | `7f4a703e49278f2002f831d93aed867ad6977281` | Shallow-fetched and detached inside the tracker image at `/opt/phantom-hamer`; this is the executed MANO regressor source. |
| `MarionLepert/phantom-E2FGVI` | `5b45ffe400288006facb350e00d319bfc6c5cbd3` | Shallow-fetched and detached inside the separate E2FGVI image. |
| Grounding DINO model | `12bdfa3120f3e7ec7b434d90674b3396eccf88eb` | Acquired into the ignored model directory and SHA-256 verified before inference. |

There is deliberately no Phantom git submodule, vendored checkout, or symlink
in Video2Data. The locally versioned `inpainting.phantom_tracker` package owns
the TACO-specific bimanual association, calibrated metric conversion,
container orchestration, validation, and the adapter into the common
`tracking.npz` contract. The Phantom HaMeR source remains unmodified inside the
image. Licensed MANO files and downloaded checkpoints remain outside git and
are mounted read-only.

This separation also makes the network boundary explicit. `build` fetches the
pinned source and `acquire` fetches the pinned weights; `infer` resolves the
immutable image ID, revalidates all hashes, and runs with `--network none`.
Downstream Sharpa or parallel-jaw retargeting consumes only the common
Video2Data artifact and has no dependency on the external Phantom checkout.

TACO is bimanual while Phantom's non-EPIC processor selects only one top DINO
box. This stage makes the extension explicit: in calibrated TACO imagery,
image-left is anatomical left; minimum-displacement association then
preserves identities. Uncertain assignments are invalid, reported as
`identity_ambiguous`, and subject to a declared quality gate. Ground truth is
never read by the stage. An explicit 0.1%–12% full-frame area gate rejects a
known DINO failure mode where the union of both arms is labeled as one hand;
the exact bounds are recorded and configurable.

Network access is allowed during `build`, to fetch the pinned HaMeR source, and
during `acquire`, to fetch the pinned weights. Inference uses exactly physical
GPU 0, `--network none`, a read-only container root, read-only video/model/MANO
mounts, and a single writable output mount.
Every pinned Grounding DINO and HaMeR file is SHA-256 verified on the host
immediately before container execution and again in the container immediately
before its network is loaded. HaMeR uses an exact `strict=True` state-dict load;
missing or unexpected keys abort the run. A dry run only resolves and prints the
plan and does not create its output directory.

```bash
python -m inpainting.phantom_tracker.runner build

python -m inpainting.phantom_tracker.runner acquire \
  --download-dir inpainting/artifacts/phantom_tracker/downloads \
  --models-dir inpainting/artifacts/phantom_tracker/models

python -m inpainting.phantom_tracker.runner infer \
  --video /absolute/path/to/color.mp4 \
  --intrinsics /absolute/path/to/egocentric_intrinsic.txt \
  --models-dir inpainting/artifacts/phantom_tracker/models \
  --mano-dir /absolute/path/to/mano_v1_2/models \
  --output-dir /absolute/path/to/run/phantom/tracking \
  --sequence-id taco_dust__brush__cup_20231005_253 \
  --gpu 0
```

Add `--dry-run` to the last command for an OSMO-style resolved execution plan.
The committed bundle contains `phantom_raw_predictions.npz`, common-contract
`tracking.npz`, `hand_overlay.mp4`, and `run_metadata.json`. The sidecar
fingerprints source commits, the immutable Docker image ID, all model/MANO
inputs, the TACO camera matrix, runtime versions, parameters, and outputs.
HaMeR's weak-perspective camera anchor is remapped to calibrated metric camera
coordinates; the metric MANO-local hand geometry is then added unchanged.
`tracking.npz` includes 21 camera-frame global MANO rotations per hand as
normalized WXYZ quaternions for the shared Sharpa adapter. These use pinned
native-side manotorch `AxisLayerFK` anatomy-aligned rest axes, matching the
shared Video2Data MANO-to-Sharpa boundary. Robot retargeting is
intentionally not performed here: the common Video2Data Sharpa retargeter must
consume this tracker output, rather than introducing Phantom-specific robot
semantics.
