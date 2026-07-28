# Video2Data condition adapter

`inpainting.adapters.video2data` emits the common `tracking.npz` and
`robot_trajectory.npz` contracts from either of these model-independent input
boundaries:

1. a packaged reconstruction `result/` containing `result.npz` and
   `manifest.json` from WiLoR or HaMeR; or
2. raw `v2d.wilor.lib.video_to_hands` files `000000.json` through
   `<N-1>.json`, plus the official TACO intrinsic and frame-aligned
   world-to-camera files.

The raw WiLoR route is the shortest path for the initial TACO condition because
it does not require object reconstruction, monocular depth, or SLAM. The
adapter keys detections by handedness, never JSON-list position. A frame with
multiple detections claiming one side is rejected as ambiguous. Missing hands
become invalid NaN rows; missing frame JSON is a hard frame-alignment error.

## Safe preflight

Preflight is the default and never writes or runs a learned model:

```bash
python3 -m inpainting.adapters.video2data \
  --wilor-json-dir <condition-dir>/wilor_raw \
  --source-video <taco-color.mp4> \
  --taco-intrinsic <egocentric_intrinsic.txt> \
  --taco-extrinsic <egocentric_frame_extrinsic.npy> \
  --output-dir <condition-dir>/tracking
```

For a full reconstruction bundle, replace the raw/calibration arguments with:

```bash
--result-dir <reconstruction-output>/result
```

`--execute` is required to run licensed MANO FK and existing Robotic Grounding
Sharpa IK. Existing outputs also require `--overwrite`.

The adapter's default licensed MANO root is read directly, never copied or
modified:

```text
/home/mverghese/visual_inpainting/mano_v1_2/models/MANO_LEFT.pkl
/home/mverghese/visual_inpainting/mano_v1_2/models/MANO_RIGHT.pkl
```

## Preferred split-container execution

Raw WiLoR JSON is complete and frame-aligned for all three selected TACO
conditions. Adapting those files does not rerun WiLoR or require its learned
checkpoints. The executable path is split at `tracking.npz` so each stage runs
in the environment that already contains its dependencies:

1. Run MANO FK in `v2d_wilor:latest`, using the licensed MANO directory as a
   read-only mount. The stage is CPU-capable and writes `tracking.npz` followed
   by the `tracking.json` commit/provenance sidecar.
2. Run Sharpa retargeting in `robotic-grounding:latest`, pointing
   `--robot-assets-dir` at the hydrated Robotic Grounding assets already baked
   into that image. It reads the exact tracking archive and sidecar and writes
   `robot_trajectory.npz` plus `robot_trajectory.json`.

When raw inference is regenerated, resolve the immutable image ID on the host
and make the runner execute that ID rather than the mutable tag:

```bash
V2D_WILOR_IMAGE_ID="$(docker image inspect v2d_wilor:latest --format '{{.Id}}')"

cd reconstruction
python3 -m v2d.wilor.docker.run_video_to_hands \
  --video_path <taco-color.mp4> \
  --output_dir <condition-dir>/wilor_raw \
  --weights_dir <original-wilor-weights-dir> \
  --image_id "$V2D_WILOR_IMAGE_ID" \
  --gpu 0
```

Inference never skips individual existing frame files. It computes a complete
generation in a private sibling directory, rehashes the video, optional bbox
inputs, all four consumed weight files, and implementation sources, then
publishes the directory with one rename. `run_generation.json` commits the
immutable image ID, pinned upstream revisions, parameters, exact expected
frame set, per-frame hashes, and ordered aggregate hash. A subsequent command
is a no-model strict resume only when that complete generation and every
current input match exactly. A legacy, partial, tampered, or differently
parameterized output directory is refused; use a new destination (or explicitly
archive the old destination) instead of mixing generations.

Inside `v2d_wilor:latest`, preflight and then execute the MANO stage with:

```bash
# Resolve this on the host and pass the exact value into the container.
V2D_WILOR_IMAGE_ID="$(docker image inspect v2d_wilor:latest --format '{{.Id}}')"

python3 -m inpainting.adapters.video2data_tracking \
  --wilor-json-dir <condition-dir>/wilor_raw \
  --source-video <taco-color.mp4> \
  --taco-intrinsic <egocentric_intrinsic.txt> \
  --taco-extrinsic <egocentric_frame_extrinsic.npy> \
  --output-dir <condition-dir> \
  --mano-model-dir <mounted-mano-v1.2-root> \
  --public-weights-dir <original-wilor-weights-dir> \
  --wilor-image-id "$V2D_WILOR_IMAGE_ID" \
  --device cpu \
  --sequence-id <sequence-id>

python3 -m inpainting.adapters.video2data_tracking \
  --wilor-json-dir <condition-dir>/wilor_raw \
  --source-video <taco-color.mp4> \
  --taco-intrinsic <egocentric_intrinsic.txt> \
  --taco-extrinsic <egocentric_frame_extrinsic.npy> \
  --output-dir <condition-dir> \
  --mano-model-dir <mounted-mano-v1.2-root> \
  --device cpu \
  --sequence-id <sequence-id> \
  --wilor-image-id "$V2D_WILOR_IMAGE_ID" \
  --public-weights-dir <original-wilor-weights-dir> \
  --execute
```

`--public-weights-dir` is provenance-only for this adaptation: the sidecar
fingerprints and verifies its public WiLoR files.
It does not make the MANO stage run WiLoR. `--wilor-image-id` records the image
identity used for the raw-observation pipeline and must be an immutable
`sha256:<64 hex>` ID, never a mutable tag.

Then, inside `robotic-grounding:latest`, preflight and execute the Sharpa stage:

```bash
SHARPA_IMAGE_ID="$(docker image inspect robotic-grounding:latest --format '{{.Id}}')"

python3 -m inpainting.adapters.sharpa_from_tracking \
  --tracking <condition-dir>/tracking.npz \
  --tracking-metadata <condition-dir>/tracking.json \
  --output-dir <condition-dir> \
  --robot-assets-dir <hydrated-robotic-grounding-assets> \
  --sharpa-image-id "$SHARPA_IMAGE_ID" \
  --mano-to-robot-scale 1.2 \
  --device cpu

python3 -m inpainting.adapters.sharpa_from_tracking \
  --tracking <condition-dir>/tracking.npz \
  --tracking-metadata <condition-dir>/tracking.json \
  --output-dir <condition-dir> \
  --robot-assets-dir <hydrated-robotic-grounding-assets> \
  --sharpa-image-id "$SHARPA_IMAGE_ID" \
  --mano-to-robot-scale 1.2 \
  --device cpu \
  --max-frame-task-error-m 0.07 \
  --execute
```

Both CLIs default to read-only preflight. `--execute` is an explicit commit
request, and replacing an existing stage result additionally requires
`--overwrite`. The tracking sidecar records source video geometry and FPS,
raw/result input files, hand-pose source, MANO implementation, shape and scale
policy, validity counts, optional WiLoR provenance, and the output hash. The
trajectory sidecar binds the consumed tracking hash and metadata to the Sharpa
implementation, robot assets, retargeting scale/error policy, validity counts,
and output hash. `--sharpa-image-id` is mandatory and accepts only an immutable
`sha256:<64 hex>` Docker ID. The trajectory sidecar records that ID, SHA-256
for the adapter and every Python source in the imported local retarget package,
and SHA-256 for both consumed XMLs and every mesh they reference. Tracking
inputs, implementation sources, and the complete XML/mesh inventory are
rehash-verified immediately before the atomic commit. The NPZ contracts are
re-opened and validated before either sidecar is committed.

The investigation uses an explicit `0.07 m` maximum Sharpa frame-task
positional-residual guard. This is a catastrophic-solution rejection gate, not
a convergence claim. A `0.05 m` trial rejected 25 of 74 otherwise finite
observed frames per side on sequence `20231005_253`; `0.07 m` retained all
observations. The sidecar records median, p95, and maximum residuals plus every
rejection, and resets the temporal seed after a rejected solve.

## Corrected semantics

- Source frame count, geometry, and FPS come from probing the original video;
  no 30 FPS constant is accepted.
- Packaged hand validity is intersected with camera-pose validity. Raw WiLoR
  validity is the presence of exactly one unambiguous observation for a side.
- Invalid rows remain NaN through tracking and robot retargeting. Sharpa's
  temporal seed is reset after every invalid frame.
- Packaged `hand_scale` is applied around the posed MANO vertex centroid before
  camera translation. Raw WiLoR has no aligned scale and therefore declares
  `hand_scale=1.0` rather than inventing a depth correction.
- Raw WiLoR translation is moved from its virtual centered pinhole to the real
  TACO K by preserving the predicted centroid pixel and rescaling metric depth
  by `fx_real / scaled_focal_length`.
- The official TACO world-to-camera matrix is inverted per frame, then applied
  to MANO positions and joint orientations.
- Stored left-hand WiLoR/HaMeR axis angles are in right-MANO parameter space.
  Y/Z components are negated before native-left MANO FK.
- Per-frame raw WiLoR betas are reduced to a per-side median shape over valid
  detections, avoiding frame-to-frame hand-shape jitter.

## WiLoR checkpoint audit (not needed for existing raw JSON)

The Blackwell-compatible `v2d_wilor:latest` image is present. These three
public cache files are verified and available under
`inpainting/artifacts/weights/wilor/pretrained_models/`:

```text
mano_mean_params.npz
wilor_final.ckpt
detector.pt
```

They are only needed to regenerate raw observations. Before using the download
wrapper, the separately licensed model must already exist at:

```text
<weights_dir>/pretrained_models/MANO_RIGHT.pkl
```

The wrapper never downloads or modifies that licensed file. It accepts
`--weights_dir` and optional `--dev`; there is no `--gpu` argument:

```bash
cd reconstruction
python3 -m v2d.wilor.docker.run_download_weights \
  --weights_dir data/weights/wilor
```

The wrapper runs a lightweight downloader in the WiLoR image. It does not
instantiate the model and does not request a GPU. The downloader fetches only
the following three public artifacts from `warmshao/WiLoR-mini` at pinned
revision `b00adea9a6843bbb4c9042109c5eb29ab2a59dea`, then verifies each against
its repository-pinned SHA-256 digest:

```text
https://huggingface.co/warmshao/WiLoR-mini/resolve/b00adea9a6843bbb4c9042109c5eb29ab2a59dea/pretrained_models/mano_mean_params.npz
https://huggingface.co/warmshao/WiLoR-mini/resolve/b00adea9a6843bbb4c9042109c5eb29ab2a59dea/pretrained_models/wilor_final.ckpt
https://huggingface.co/warmshao/WiLoR-mini/resolve/b00adea9a6843bbb4c9042109c5eb29ab2a59dea/pretrained_models/detector.pt
```

The provided licensed MANO model should be staged explicitly at the required
path instead of treating it as a public checkpoint.

The optional HaMeR checkpoint is also absent. Its repository downloader uses
this public archive:

```bash
cd reconstruction
python3 -m v2d.hamer.docker.run_download_weights \
  --weights_dir data/weights/hamer
```

```text
https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
```

For the raw WiLoR condition, HaMeR and all object/depth/SLAM checkpoints are
unnecessary.

## Current execution status

The earlier execution blockers no longer apply to the split route:

- every expected raw WiLoR frame JSON exists for the three selected TACO
  conditions;
- the licensed MANO pair and official TACO cameras are present;
- `v2d_wilor:latest` supplies `torch` and `manotorch` for the CPU MANO stage;
- `robotic-grounding:latest` supplies the Sharpa dependencies, XMLs, and
  hydrated meshes required by the retargeter; and
- no single environment needs to contain both dependency stacks.

The host checkout's Git LFS mesh pointers do not block this containerized
route, because the Sharpa stage is explicitly directed to the hydrated assets
inside `robotic-grounding:latest`. The public WiLoR weights are not executed
when consuming the already-generated raw JSON, but their verified fingerprints
remain mandatory provenance in each tracking sidecar.

Each stage's preflight reports any input-, dependency-, asset-, provenance-, or
existing-output issue as a machine-readable blocker and exits with status 2
while blocked.

All three selected clips completed the earlier split stages with the following
valid counts. Raw directories created before `run_generation.json` was added
are intentionally *not* resumable under the current strict contract; each must
be regenerated into a fresh destination and then passed through tracking v2
before claiming current generation-level validation. The observation counts
themselves are unchanged:

| Sequence | Frames | Left valid | Right valid | Sharpa rejects at 0.07 m |
|---|---:|---:|---:|---:|
| `taco_empty__kettle__plate_20231031_060` | 155 | 155 | 146 | 0 |
| `taco_cut__knife__plate_20231013_105` | 152 | 152 | 152 | 0 |
| `taco_dust__brush__cup_20231005_253` | 74 | 74 | 74 | 0 |

Sequence `253` has now completed that fresh-generation upgrade. Its committed
raw WiLoR JSON, tracking v2 sidecar, and Sharpa v2 sidecar all pass strict
resume validation; the regenerated observations and both downstream NPZ
archives are numerically identical to the preserved legacy generation. The
`060` and `105` rows remain useful earlier-generation results, but their raw
directories and sidecars should not be described as current v2 provenance
until they are regenerated through the same path.

The nine invalid right-hand rows in `060` are the original missing WiLoR
observations and remain NaN through Sharpa; they were not interpolated. Camera
projection overlays for the first, middle, and final frames of every clip were
visually inspected and align with the visible hands.
