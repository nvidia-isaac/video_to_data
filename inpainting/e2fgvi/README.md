# E2FGVI video inpainting stage

This directory contains a self-contained, containerized E2FGVI-HQ stage for the
visual-inpainting investigation. The host process only validates inputs and
orchestrates Docker. Model inference stays in the image.

The image fetches
[`MarionLepert/phantom-E2FGVI`](https://github.com/MarionLepert/phantom-E2FGVI)
and checks out exactly `5b45ffe400288006facb350e00d319bfc6c5cbd3` during the build. That is the
requested Phantom release commit; it is not the current SHA of the upstream
`MCG-NKU/E2FGVI` default branch. The source checkout is left unmodified. A small
inference-only `mmcv` compatibility package maps the three APIs used by E2FGVI
to PyTorch/Torchvision, avoiding old binary MMCV wheels and constructor-time
network downloads.

## Input and output contract

- `input_video`: any constant-resolution video OpenCV can decode.
- `masks`: a NumPy `.npy` file with **boolean** dtype and exact shape
  `[decoded_frame_count, source_height, source_width]`. `True` means “replace
  this pixel.” Integer 0/1 arrays are rejected intentionally.
- `checkpoint`: an E2FGVI-HQ state dictionary, supplied at run time. Weights are
  never baked into or downloaded by this stage.
- `output_video`: the inpainted video. Its decoded frame count, width, height,
  and FPS are checked against the source before the temporary output is moved
  into place.
- `metadata_path`: canonical JSON, defaulting to `<output_video>.json`. It
  contains source/model provenance, input SHA-256 values, geometry, FPS, and all
  inference parameters. Completed metadata also fingerprints the encoded output
  bytes. It contains no timestamps, durations, random temporary paths, or
  machine identifiers, so identical inputs/options and execution image produce
  identical sidecar text.

Before inference can replace an output, the CLI atomically publishes a
`committing` sidecar. The final `completed` sidecar is published only after the
output is decoded, fingerprinted, and all input fingerprints are rechecked. An
interrupted overwrite therefore cannot leave an old completion marker beside a
new output. The host runner likewise resolves `v2d_e2fgvi` once and executes its
immutable `sha256:...` image ID, which is recorded in the sidecar.

The output preserves the source video stream geometry and FPS, but it does not
copy audio. Downscaled inference is composited back into the source-resolution
frames so pixels outside the dilated mask retain source resolution.

## Build

From `video_to_data_internal`:

```bash
python inpainting/e2fgvi/docker/build.py
```

The image is named `v2d_e2fgvi`. Building needs network access for the pinned
source commit and pinned Torchvision wheel. It does not fetch a model checkpoint.
The base is PyTorch 2.7.1 with CUDA 12.8/CuDNN 9 development libraries so its
CUDA stack supports the investigation host's Blackwell (`sm_120`) GPUs.

For the host runner, install the repository's `v2d-docker` package plus these
two editable packages in the same Python environment:

```bash
python -m pip install -e reconstruction/modules/v2d_docker
python -m pip install -e inpainting/e2fgvi/lib -e inpainting/e2fgvi/docker
```

## Validate without Docker, a model import, or a GPU

`--dry-run` decodes the video on the host, validates the boolean mask contract
and checkpoint file, computes the processing plan, hashes the inputs, and prints
the exact deterministic report. It neither invokes Docker nor writes outputs:

```bash
v2d-run-e2fgvi \
  --input-video clip.mp4 \
  --masks masks.npy \
  --checkpoint E2FGVI-HQ-CVPR22.pth \
  --output-video clip_inpainted.mp4 \
  --device cpu \
  --dry-run
```

`--validate-only` runs the same validation CLI inside an already-built image,
without requesting a GPU or importing E2FGVI/Torch. It writes a sidecar whose
status is `validated`, but does not create the video:

```bash
v2d-run-e2fgvi \
  --input-video clip.mp4 \
  --masks masks.npy \
  --checkpoint E2FGVI-HQ-CVPR22.pth \
  --output-video clip_inpainted.mp4 \
  --validate-only
```

## Run inference

Select the physical host GPU explicitly with `--gpu`. The host runner passes
Docker `--gpus device=<physical-index>`, so no other host GPU is exposed and the
selected device is consistently addressed inside the container as `cuda:0`:

```bash
v2d-run-e2fgvi \
  --input-video clip.mp4 \
  --masks masks.npy \
  --checkpoint E2FGVI-HQ-CVPR22.pth \
  --output-video clip_inpainted.mp4 \
  --gpu 1 \
  --device cuda:0 \
  --downscale 1 \
  --max-size 960 \
  --dilation-iterations 4 \
  --dilation-kernel 3 \
  --neighbor-stride 5 \
  --ref-stride 20 \
  --num-ref 5
```

`downscale` is a divisor (`2` halves both dimensions). `max-size=0` disables
the longest-edge cap. `num-ref=-1` uses every eligible reference at
`ref-stride`; a positive value limits references and usually reduces memory.
The default `mp4v` codec can be changed with a four-character `--codec` value.
Use `--overwrite` to replace an existing requested output/sidecar.

The host runner enables strict input/output mount isolation. Keep the source
video, mask, and checkpoint outside the output directory (the investigation's
`shared_arm_mask` and `shared_inpaint` layout already satisfies this).

Legacy completed sidecars can be upgraded without rerunning inference only when
their current video, masks, checkpoint, and decoded output geometry still match
the recorded values:

```bash
v2d-run-e2fgvi \
  --input-video clip.mp4 \
  --masks masks.npy \
  --checkpoint E2FGVI-HQ-CVPR22.pth \
  --output-video clip_inpainted.mp4 \
  --enrich-completed-metadata
```

This adds the output fingerprint atomically and explicitly marks the original
container identity as unrecorded; it never claims the currently installed image
produced a legacy artifact.

The container CLI can also be called directly. Both dash and underscore option
spellings are accepted so it works with `v2d.docker.container.run_in_container`:

```bash
python -m v2d.inpainting.e2fgvi.cli \
  --input-video /data/clip.mp4 \
  --masks /data/masks.npy \
  --checkpoint /data/E2FGVI-HQ-CVPR22.pth \
  --output-video /data/clip_inpainted.mp4 \
  --device cuda:0
```

## Tests

The unit suite covers pure mask/config validation, resizing, reference sampling,
and deterministic metadata. It does not need E2FGVI, Torch, Docker, weights, or
a GPU:

```bash
python -m unittest discover -s inpainting/e2fgvi/tests -v
```

## Licensing

The stage code is Apache-2.0. E2FGVI is fetched as third-party source and is
licensed under its repository's CC BY-NC 4.0 license; review that non-commercial
license before using the model or weights outside this investigation.
