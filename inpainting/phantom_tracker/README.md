# Phantom hand-tracking condition

This stage runs Phantom's non-EPIC hand stack as a reproducible TACO
condition: `IDEA-Research/grounding-dino-base` at immutable revision
`12bdfa3120f3e7ec7b434d90674b3396eccf88eb` proposes hand boxes and the
official Phantom HaMeR fork at `7f4a703e49278f2002f831d93aed867ad6977281`
regresses MANO pose. That is the HaMeR submodule pin of parent Phantom commit
`a8bb81c1bbe6ade129a1f6f0906482f510354a5e`; they are intentionally different
repositories and commit IDs. The image uses PyTorch 2.7.1/CUDA 12.8 for RTX PRO
6000 Blackwell support.

TACO is bimanual while Phantom's non-EPIC processor selects only one top DINO
box. This stage makes the extension explicit: in calibrated TACO imagery,
image-left is anatomical left; minimum-displacement association then
preserves identities. Uncertain assignments are invalid, reported as
`identity_ambiguous`, and subject to a declared quality gate. Ground truth is
never read by the stage. An explicit 0.1%–12% full-frame area gate rejects a
known DINO failure mode where the union of both arms is labeled as one hand;
the exact bounds are recorded and configurable.

Network access is allowed only for `acquire`. Inference uses exactly physical
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
  --mano-dir /home/mverghese/visual_inpainting/mano_v1_2/models \
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
