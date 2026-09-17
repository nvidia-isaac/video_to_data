# Local multi-view calibration and HOI reconstruction

This guide calibrates a multi-view camera rig and then runs
`v2d.pipelines.run_mv_hoi_reconstruction` on one sequence from a Linux
workstation. Lightweight host Python processes launch each stage in its own
Docker container. Calibration produces the EDEX camera parameters consumed by
reconstruction. The supported local object-detection path uses the text prompt
in `hoi_metadata.yaml` with Grounding DINO.

A matching, validated calibration can be reused for multiple reconstruction
sequences captured with the same unchanged rig. Run calibration again when the
rig geometry changes or no suitable EDEX is available.

The reconstruction runner performs, in order:

1. rosbag extraction and multi-view preprocessing;
2. face detection and anonymized-video generation;
3. Foundation Stereo depth estimation;
4. Grounding DINO object detection, SAM2 object masks, and FoundationPose;
5. Detectron2 person tracking, SAM2 human masks, and SAM3D Body optimization;
6. SOMA-X export, ground-plane estimation, and fused-point-cloud export;
7. human/object Chamfer and silhouette evaluation; and
8. HOI overlay and Wis3D generation.

The reconstruction pipeline is a reconstruction and diagnostic runner. It does
**not** run production `check_accuracy`, upload to HITL, query human QC, trim a
sequence, or produce a committed dataset export.

## Data expectations

These are capture recommendations, not automated input gates. Recordings that
do not meet them may still run, but person selection, masks, depth, and pose
tracking are more likely to fail or produce unusable results.

- Make the intended person unambiguous. They should generally be centered,
  closest to the rig and therefore one of the largest people in the views,
  visible for most of the recording, and responsible for the most meaningful
  movement. Avoid similarly prominent bystanders, people crossing tracks, and
  prolonged occlusion of the target person's body or hands.
- Keep the manipulated object large enough to resolve in multiple camera views.
  Avoid prolonged occlusion by hands, the body, other people, or the environment.
- Use controlled, continuous manipulation. Do not toss or throw the object, and
  avoid rapid movement that creates motion blur or large frame-to-frame
  displacement.

## Capture equipment

The configured capture rig uses an NVIDIA Nova Orin recording platform with
four Leopard Imaging Hawk stereo cameras arranged at the front, back, left, and
right of the capture space. The pipeline expects eight camera streams—left and
right from each stereo camera—and the default Nova Hawk extractor writes them
at 1920×1200 resolution.

Calibration is tied to the physical camera geometry. A validated calibration
can be reused while the rig remains unchanged; recalibrate after moving or
remounting a camera.

## Host and compute requirements

- Linux with Python 3.10 or newer
- Docker with permission to run containers
- NVIDIA driver and NVIDIA Container Toolkit for reconstruction
- An NVIDIA GPU with compute capability 8.0 or newer from the currently
  supported compiled set: `sm_80`, `sm_86`, `sm_89`, or `sm_90`; calibration
  itself is CPU-only
- Enough local storage for the images, weights, extracted frames, depth, and
  diagnostic videos
- Network access for initial image builds and model downloads

With model weights, caches, and compatible TensorRT engines already prepared,
local reconstruction of a 30-second sequence on an RTX 4090 takes approximately
one hour. This is a warm-start reference rather than a performance guarantee:
frame count and sequence content affect runtime, and first-run downloads or
TensorRT engine generation can make the run longer. Calibration time is not
included in this estimate.

The complete local pipeline has been validated on an NVIDIA GeForce RTX 4090
(`sm_89`). The OSMO pipeline has been validated on NVIDIA H100 (`sm_90`), L40
(`sm_89`), and L40S (`sm_89`) GPUs. The current native CUDA extensions also
compile for `sm_80` and `sm_86`, but those configurations are not claimed as
validated. Blackwell `sm_120` is not supported by the current compiled image
set. TensorRT engines are GPU-architecture- and runtime-specific, so the
selected engines must match the execution environment.

Start in the `reconstruction` directory. All paths passed to either runner
should be absolute.

```bash
cd /absolute/path/to/video_to_data/reconstruction

python3 -m venv .venv
source .venv/bin/activate
./scripts/install_packages.sh
```

Confirm that Docker works. Before reconstruction, also confirm that Docker can
see the GPU:

```bash
nvidia-smi
docker info >/dev/null
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker run --rm --gpus all \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  python -c 'import torch; cc=torch.cuda.get_device_capability(); supported={(8, 0), (8, 6), (8, 9), (9, 0)}; assert cc in supported, f"unsupported compute capability: {cc}"; result=(torch.ones(1024, device="cuda") + 1).sum().item(); assert result == 2048; print(f"CUDA kernel OK: {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}")'
```

The checks pull the public CUDA and PyTorch images if they are not already
present. The final command checks the GPU architecture and executes a compiled
CUDA tensor kernel; `nvidia-smi` alone verifies only driver/container access.

## Build the local images

Build the calibration and MV-HOI image set from the current checkout:

```bash
./workflows/mv_hoi/build_images.sh
```

Calibration uses `v2d_rosbag` and `v2d_mv_calibration`; reconstruction uses
`v2d_rosbag` and the remaining images:

```text
v2d_rosbag
v2d_mv_calibration
v2d_mv_preprocess
v2d_face_detector
v2d_foundation_stereo
v2d_grounding_dino
v2d_sam2
v2d_foundation_pose
v2d_detectron2
v2d_sam3d_body
v2d_mv_postprocess
```

Verify that they exist:

```bash
for image in \
  v2d_rosbag v2d_mv_calibration v2d_mv_preprocess v2d_face_detector \
  v2d_foundation_stereo v2d_grounding_dino v2d_sam2 \
  v2d_foundation_pose v2d_detectron2 v2d_sam3d_body \
  v2d_mv_postprocess; do
  docker image inspect "$image" >/dev/null
done
```

## Calibrate the camera rig

Calibration requires a directory containing at least one nonempty `.mcap`
recording of the calibration board. It does not require `hoi_metadata.yaml`, a
GPU, or model weights. Use a new output directory for every full attempt; the
local runner does not safely resume or skip completed calibration stages.

The default packaged setup is `stereo4_6x10_100mm_marker`, a marker-aware 6-by-10
board with 100 mm squares. Use another setup only when it matches the physical
board used in the recording. The other packaged setup currently available is
`stereo4_6x10_22p58mm_marker`.

Set absolute paths and verify the input and output locations:

```bash
CALIBRATION_SEQUENCE_DIR=/absolute/path/to/calibration_sequence
CALIBRATION_OUTPUT_DIR=/absolute/path/to/local_runs/calibration_run_01

test -d "$CALIBRATION_SEQUENCE_DIR"
test -s "$CALIBRATION_SEQUENCE_DIR/metadata.yaml"
find "$CALIBRATION_SEQUENCE_DIR" -maxdepth 1 -type f -name '*.mcap' \
  -print -quit | grep -q .
test ! -e "$CALIBRATION_OUTPUT_DIR"
```

Run calibration with the default setup:

```bash
python -m v2d.pipelines.run_mv_calibration \
  --rosbag_path "$CALIBRATION_SEQUENCE_DIR" \
  --output_dir "$CALIBRATION_OUTPUT_DIR"
```

To select the alternate packaged board definition, add:

```bash
--calibration_setup stereo4_6x10_22p58mm_marker
```

The default invocation uses code baked into the local images. Add `--dev` only
for code development; it mounts the checked-out module sources into both
calibration containers.

Calibration first extracts images and intrinsics, then detects the board,
initializes camera poses with PnP, and refines the extrinsics with bundle
adjustment. Its principal outputs are:

```text
<calibration_output>/
|-- raw/
|   |-- images/
|   `-- edex
`-- extrinsics/
    |-- edex
    `-- calibration_accuracy.json
```

Successful completion prints:

```text
=== Calibration Complete ===
```

Validate the handoff artifacts before reconstruction:

```bash
CALIBRATION_EDEX="$CALIBRATION_OUTPUT_DIR/extrinsics/edex"

test -d "$CALIBRATION_OUTPUT_DIR/raw"
test -s "$CALIBRATION_EDEX"
test -s "$CALIBRATION_OUTPUT_DIR/extrinsics/calibration_accuracy.json"
python -m json.tool "$CALIBRATION_EDEX" >/dev/null
python -m json.tool \
  "$CALIBRATION_OUTPUT_DIR/extrinsics/calibration_accuracy.json" >/dev/null
```

Review `calibration_accuracy.json`, especially the bundle-adjustment
reprojection statistics, before treating the result as a valid calibration.
Pass `CALIBRATION_EDEX` directly to reconstruction as shown below.

## Download the model weights

Calibration needs no model weights. The reconstruction runner uses fixed weight
directories below `data/weights`. Run all commands from `reconstruction` after
building the images:

```bash
python -m v2d.face_detector.docker.run_download_weights \
  --output_dir data/weights/face_detector

python -m v2d.foundation_stereo.docker.run_download_weights \
  --output_dir data/weights/foundation_stereo

python -m v2d.grounding_dino.docker.run_download_weights \
  --output_dir data/weights/grounding_dino

python -m v2d.sam2.docker.run_download_weights \
  --output_dir data/weights/sam2

python -m v2d.foundation_pose.docker.run_download_weights \
  --output_dir data/weights/foundation_pose \
  --backend nvidia_tensorrt \
  --accept_nvidia_model_eula

python -m v2d.detectron2.docker.run_download_weights \
  --output_dir data/weights/detectron2 \
  --model_sizes b

python -m v2d.sam3d_body.docker.run_download_weights \
  --output_dir data/weights/sam3d_body
```

FoundationPose's `--accept_nvidia_model_eula` flag passes explicit acceptance
to the NVIDIA model downloader. Review the applicable model license before
using it.

SAM3D Body is hosted in a gated Hugging Face repository. Request access to
`facebook/sam-3d-body-dinov3` and run `hf auth login` on the host first. Install
the Hugging Face CLI with `python -m pip install huggingface_hub` if `hf` is not
already available. The wrapper uses `HF_TOKEN` when set and otherwise reads the
standard Hugging Face token file.

SOMA-X is used later by the same `v2d_sam3d_body` image. Its public model assets
may be fetched on first use and cached under
`data/weights/sam3d_body/hf_home`. That directory must be writable, and the
first run needs network access unless the cache has already been populated.

### Optional: prebuild FoundationPose engines

The NVIDIA FoundationPose backend builds GPU-compatible TensorRT engines when
they are missing. To pay that cost before a full run, build them explicitly on
the GPU that will execute the pipeline:

```bash
python -m v2d.foundation_pose.docker.run_export_engines \
  --weights_dir data/weights/foundation_pose
```

TensorRT engines are GPU/runtime-specific. Rebuild them after changing the GPU
architecture, TensorRT runtime, or FoundationPose model files. Without this
step, the FoundationPose stage builds missing engines automatically.

## Prepare the reconstruction inputs

The runner requires four inputs:

1. **Sequence directory**: contains at least one `.mcap` rosbag and
   `hoi_metadata.yaml`.
2. **Calibration EDEX**: the validated
   `$CALIBRATION_OUTPUT_DIR/extrinsics/edex` produced above, or a matching
   previously validated calibration.
3. **Object mesh**: pass the mesh directory's `output_aligned.glb`.
4. **New output directory**: must not contain results from an earlier run.

A minimal sequence layout is:

```text
sequence/
|-- recording_0.mcap
`-- hoi_metadata.yaml
```

The metadata must contain a nonempty Grounding DINO prompt:

```yaml
object:
  prompt: blue trash can
```

The legacy `object.bbox` field is ignored and removed from the forwarded
metadata. The supported local runner does not create or consume manual bbox
labels; Grounding DINO always seeds SAM2 from `object.prompt`.

The object mesh directory must contain the pinned aligned mesh. Put the optional
BOP-style symmetry annotation beside it so preprocessing carries both files
together:

```text
object_mesh/
|-- output_aligned.glb
`-- output_symmetry.json       # optional
```

Preprocessing copies the mesh's entire parent directory into its output. Keep
the source mesh directory outside the run output directory. The sequence,
calibration, mesh, and output paths must be distinct; nesting the mesh source in
the output can cause a same-file `copytree` failure.

Before launching, check the inputs and reserve a fresh output path:

```bash
SEQUENCE_DIR=/absolute/path/to/sequence
CALIBRATION_EDEX=${CALIBRATION_EDEX:-/absolute/path/to/calibration/extrinsics/edex}
OBJECT_MESH=/absolute/path/to/object_mesh/output_aligned.glb
OUTPUT_DIR=/absolute/path/to/local_runs/sequence_run_01

test -d "$SEQUENCE_DIR"
find "$SEQUENCE_DIR" -maxdepth 1 -type f -name '*.mcap' -print -quit | grep -q .
test -s "$SEQUENCE_DIR/hoi_metadata.yaml"
test -s "$CALIBRATION_EDEX"
test -s "$OBJECT_MESH"
test ! -e "$OUTPUT_DIR"
```

## Run reconstruction

With the variables from the previous section still set, run:

```bash
python -m v2d.pipelines.run_mv_hoi_reconstruction \
  --rosbag_path "$SEQUENCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --calibration_camera_params_path "$CALIBRATION_EDEX" \
  --obj_mesh_path "$OBJECT_MESH"
```

The default command uses the code baked into the local images. Add `--dev` only
when developing pipeline modules; it mounts the current `reconstruction/modules`
tree into the containers so source edits made after the image build are visible.

The runner is sequential and is not a resumable workflow engine. It does not
inspect an existing output tree and safely skip completed stages. After a
partial failure, diagnose the cause and start the complete runner again with a
new output directory.

## Reconstruction outputs and success checks

The principal output tree is:

```text
<output>/
|-- raw/                         # extracted images and EDEX
|-- preprocess/                  # rectified images/videos, metadata, mesh
|-- face_detector/               # anonymized videos
|-- foundation_stereo/           # per-camera depth
|-- grounding_dino/              # object bounding boxes
|-- sam2/object/                 # object masks
|-- foundation_pose/             # fused object poses
|-- detectron2/                  # person tracks
|-- sam2/human/                  # human masks
|-- sam3d_body/
|   |-- mhr_params_mv.pt
|   |-- mhr_mesh_mv.pt
|   `-- export_soma/soma_params.npz
`-- postprocess/
    |-- ground_plane/
    |-- fused_pointcloud/
    |-- chamfer_human/
    |-- chamfer_object/
    |-- silhouette_mask_human/
    |-- silhouette_mask_object/
    |-- hoi_overlay/
    `-- wis3d/
```

Successful completion prints:

```text
=== Multi-View Reconstruction Complete ===
```

Perform lightweight output checks with:

```bash
test -s "$OUTPUT_DIR/preprocess/prompt.txt"
test -s "$OUTPUT_DIR/foundation_pose/poses.npy"
test -s "$OUTPUT_DIR/sam3d_body/mhr_params_mv.pt"
test -s "$OUTPUT_DIR/sam3d_body/mhr_mesh_mv.pt"
test -s "$OUTPUT_DIR/sam3d_body/export_soma/soma_params.npz"
test -s "$OUTPUT_DIR/postprocess/hoi_overlay/tiled_hoi_overlay.mp4"
test -d "$OUTPUT_DIR/postprocess/wis3d"
```

These checks confirm that the major stages produced artifacts; they are not a
replacement for visual inspection or production accuracy and QC gates.

## Troubleshooting

### Calibration extracts no images

Confirm that the calibration directory directly contains a nonempty `.mcap`
recording from the supported rig and that the rosbag includes all expected
camera streams. Preserve the failed output for diagnosis, then retry the full
calibration pipeline with a new output directory.

### Calibration cannot detect the board

Confirm that the board is visible, sharp, and sufficiently large in the
extracted images from every required camera. Verify that `--calibration_setup`
matches the physical board; using the wrong square size or pattern definition
invalidates the result even if some corners are detected.

### Bundle adjustment does not converge

Inspect the extracted calibration images for missing cameras, insufficient
viewpoint diversity, motion blur, occlusion, or incorrect board detections.
Correct the source or setup selection and retry in a new output directory. Do
not use a partial `extrinsics/edex` from a failed run.

### Calibration finishes without a usable EDEX

Require nonempty, valid JSON at both `raw/edex` and `extrinsics/edex`, plus a
valid `extrinsics/calibration_accuracy.json`. If any is missing or malformed,
treat the calibration as failed and rerun from the source recording in a new
output directory.

### A partial calibration output already exists

The local calibration runner is sequential and does not safely resume. Keep the
partial tree for logs or inspection, but choose a different empty output path
for the next complete attempt.

### Mesh files reported as the same file

If preprocessing reports that `output_symmetry.json` or `output_aligned.glb`
is the same source and destination file, the source mesh directory overlaps the
output tree or the output directory is being reused. Move the mesh template to
a separate source directory and select a new output directory.

### Grounding DINO prompt is missing

The supported runner requires a nonempty `object.prompt` in the source
`hoi_metadata.yaml`. Preprocessing writes it to `preprocess/prompt.txt`. If that
file is absent or empty, correct the source metadata and restart in a new output
directory.

### A Docker image or model file is missing

Run `workflows/mv_hoi/build_images.sh` again and repeat the relevant downloader
from the weight section. Keep the directory names exactly as documented because
the runner resolves them relative to `reconstruction/data/weights`.

### Hugging Face rejects the SAM3D Body download

Confirm that the account has accepted the gated repository terms, run
`hf auth login`, and retry the SAM3D Body downloader. If using `HF_TOKEN`, do not
write the token into a checked-in script or command log.

### FoundationPose spends a long time before tracking

Missing TensorRT engines are built before tracking starts. Use the explicit
engine-build command above and confirm that it runs on the same GPU/runtime as
the pipeline. Delete or force-rebuild engines only when diagnosing an actual
compatibility failure.

### Docker cannot access the GPU

Repeat the GPU check from the requirements section. Fix the NVIDIA driver,
Container Toolkit, Docker daemon, or user permissions before retrying the full
pipeline.

### The host runs out of disk space

Extracted images, depth, masks, and diagnostic videos can be much larger than
the source rosbag. Check free space in both Docker's storage location and the
output filesystem before starting. Remove only outputs you have positively
identified as disposable; then retry with a new output directory.
