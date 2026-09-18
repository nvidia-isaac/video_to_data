# OSMO Workflows

Workflow definitions for running training, retargeting, and development environments on
an OSMO deployment you can access. Local Docker builds and runs do not require OSMO or
a private container registry; follow [the local setup guide](../docs/SETUP.md).

## Repository-local E2E container operation

The public reconstruction-to-GR00T launcher uses workflow/run.sh e2e-run internally. It starts
or reuses the normal Robotic Grounding image without a TTY, mounts the current checkout and
active run directory, and recreates the container if a stage adds a required MANO or checkpoint
mount. Prefer the repository-root run_e2e.sh commands; e2e-run is the container lifecycle
primitive they use.

~~~bash
./workflow/run.sh e2e-run latest 0 \
  --run-data /absolute/path/to/e2e_run \
  --workdir /workspace/video_to_data/robotic_grounding \
  --recreate-on-mount-change \
  -- true
~~~

**See also:** [data_pipeline.md](data_pipeline.md) for the end-to-end data flow
(raw → retargeted → trained), artifact layout, and downloading remote results.

## Prerequisites

### 1. Build locally or configure a registry

Build the local image from `robotic_grounding/`:

```bash
./workflow/run.sh build latest
```

The base image and NGC login prerequisites are documented in [SETUP.md](../docs/SETUP.md).
For remote jobs, choose a container registry namespace you can push to and your OSMO
workers can pull from:

```bash
export V2D_IMAGE_REGISTRY=registry.example.com/your-namespace
docker login registry.example.com
./workflow/run.sh push latest
```

`V2D_IMAGE_REGISTRY` is required only for `push`, `pull`, and `run_osmo.py --build-image`.
Workflow YAML image defaults deliberately use an invalid placeholder; supply an image
when submitting directly with the OSMO CLI.

### 2. Configure your OSMO deployment and storage

Install and authenticate the OSMO CLI using your deployment's setup instructions.
Select a pool available to your account and provide it with `--pool`.
Configure image-pull credentials and storage credentials for your chosen registry and
input/output URLs in that deployment. The repository does not provide a hosted OSMO
service, pool, or shared dataset bucket.

For retargeting, upload the loaded dataset and object assets from the
[local setup](../docs/SETUP.md) to your own storage. Pass `input_url` and `output_url`
explicitly; their template defaults are empty.

### 3. Configure W&B for training jobs

Training templates reference an OSMO credential named `wandb`. Configure the key named
in the selected YAML (`wandb_api_key` in `train.yaml`, `wandb_pass` in
`train_vega_manip.yaml`), or edit its credential mapping to match your deployment.
OSMO injects it as `WANDB_API_KEY` in the task. Set any W&B entity/project to your own
workspace; no organization access is implied by these examples.

For local training without W&B, use `--logger tensorboard` as shown in the
[package README](../README.md#rl-training).

## Submitting a Job

Choose one image mode:

- `--image <registry>/<namespace>/robotic-grounding:<tag>` submits an existing image.
- `--build-image` builds locally, pushes to `V2D_IMAGE_REGISTRY`, and submits that exact
  remote image using the experiment name as its tag.

These options are mutually exclusive. Every submission requires `--pool`; `--dry-run`
previews the commands without building, pushing, or submitting.

### Remote development

```bash
python scripts/run_osmo.py --experiment-name <your-name> \
  --image <registry>/<namespace>/robotic-grounding:<tag> --pool <your-pool> \
  --workflow-yaml workflow/dev_env.yaml

# Once running:
osmo workflow port-forward <workflow-name> dev-env --port 6000:22
ssh root@localhost -p 6000
```

### Training

```bash
# V2D_IMAGE_REGISTRY must already be exported.
python scripts/run_osmo.py --experiment-name <name> --build-image \
  --pool <your-pool> --workflow-yaml workflow/train.yaml
```

### Retargeting

See [data_pipeline.md](data_pipeline.md) for each stage's inputs and outputs. Load runs
separately in the reconstruction loader image. This workflow runs object URDF generation,
processing, support reconstruction, visualization, and video generation; select a subset
with `--set stages=<stage>`. Artifacts are written beneath your `output_url`, including
`<dataset>_processed/`, `<dataset>_urdfs/`, and `reconstructed_stage/`.

```bash
# Full pipeline
python scripts/run_osmo.py --experiment-name retarget-<dataset> \
  --image <registry>/<namespace>/robotic-grounding:<tag> --pool <your-pool> \
  --workflow-yaml workflow/retarget.yaml --set dataset=<dataset> \
  --set input_url=<object-storage-url> --set output_url=<object-storage-url>

# Process only
python scripts/run_osmo.py --experiment-name retarget-<dataset>-process \
  --image <registry>/<namespace>/robotic-grounding:<tag> --pool <your-pool> \
  --workflow-yaml workflow/retarget.yaml --set dataset=<dataset> --set stages=process \
  --set input_url=<object-storage-url> --set output_url=<object-storage-url>
```

#### Filtering sequences

Use `sequence_pattern` (regex), `sequence_id` (exact), or `max_sequences` to pick a subset.
`sequence_pattern` is applied as both an OSMO-input download regex and a Python-level filter.

```bash
python scripts/run_osmo.py --experiment-name retarget-taco-screw \
  --image <registry>/<namespace>/robotic-grounding:<tag> --pool <your-pool> \
  --workflow-yaml workflow/retarget.yaml --set dataset=taco \
  --set input_url=<object-storage-url> --set output_url=<object-storage-url> \
  --set 'sequence_pattern=.*(screw|skim_off|smear|stir).*'

# Equivalent alternatives:
#   --set sequence_id=taco_screw__screwdriver__toy_20231102_063
#   --set max_sequences=10
```

## Managing Workflows

```bash
osmo workflow list
osmo workflow logs <workflow-name>
osmo workflow cancel <workflow-name>
```
