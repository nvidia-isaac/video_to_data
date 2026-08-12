# OSMO Workflows

Workflow definitions for running training, retargeting, and development environments on OSMO. See the public [OSMO user guide](https://nvidia.github.io/OSMO/main/user_guide/index.html) for platform details.

**See also:** [data_pipeline.md](data_pipeline.md) for the end-to-end data flow (raw → retargeted → trained).

## Prerequisites

### 1. Configure a container registry

Build the workflow image locally, then push it to a registry you control. Set its namespace before pushing or pulling:

```bash
export V2D_IMAGE_REGISTRY=<registry>/<namespace>
```

The checked-in OSMO templates use an intentionally invalid image hostname, so
direct submissions fail safely until you override the image. Use the wrapper
with `--image` or `--build-image`, or submit a template directly with:

```bash
osmo workflow submit workflow/train.yaml \
  --set image="$V2D_IMAGE_REGISTRY/robotic-grounding:<tag>" \
  --pool <your-pool>
```

If you use NGC, install the [NGC CLI](https://docs.ngc.nvidia.com/cli/cmd.html), sign in, and configure Docker access for your chosen registry.

### 2. Configure OSMO

Install the [OSMO CLI](https://nvidia.github.io/OSMO/main/user_guide/getting_started/install/index.html) and set up [credentials](https://nvidia.github.io/OSMO/main/user_guide/getting_started/credentials.html) for your registry and storage provider.

For an Omniverse Nucleus server, configure its credentials only if your deployment requires it; see the [token guide](https://docs.omniverse.nvidia.com/nucleus/latest/config-and-info/api_tokens.html#token-generation).

### 3. Configure W&B for training jobs

OSMO training workflows log to W&B by default. Set `WANDB_API_KEY` in the shell that submits the workflow:

```bash
export WANDB_API_KEY=<your-key>
```

If W&B is not available, do not submit a cloud training run. Use local smoke tests with `--logger tensorboard` from the top-level README.

## Submitting a Job

`run_osmo.py` builds + pushes + submits in one command. Pass `--build-image`
after setting `V2D_IMAGE_REGISTRY`, or pass `--image <registry>/<image>:<tag>`
to submit an existing image. Use `--dry-run` to preview.

### Remote development

```bash
python scripts/run_osmo.py --experiment-name <your-name> --build-image \
  --pool <your-pool> --workflow-yaml workflow/dev_env.yaml

# Once running:
osmo workflow port-forward <workflow-name> dev-env --port 6000:22
ssh root@localhost -p 6000
```

### Training

```bash
python scripts/run_osmo.py --experiment-name <name> --build-image \
  --pool <your-pool> --workflow-yaml workflow/train.yaml
```

### Retargeting

See [data_pipeline.md](data_pipeline.md) for what each stage does. Stages (`load`, `process`, `reconstruct`, `visualize`, `video`) can run together or individually. Configure the workflow's `output_url` for the object-storage location where you want results published.

```bash
# Full pipeline (works for any registered dataset: taco, arctic, oakink2, hot3d, h2o, grab, dexycb)
python scripts/run_osmo.py --experiment-name retarget-<dataset> \
  --image <your-registry>/robotic-grounding:<your-tag> \
  --pool <your-pool> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=<dataset>

# Run only one stage
python scripts/run_osmo.py --experiment-name retarget-<dataset>-<stage> \
  --image <your-registry>/robotic-grounding:<your-tag> \
  --pool <your-pool> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=<dataset> --set stages=<stage>
```

#### Filtering sequences

Use `sequence_pattern` (regex), `sequence_id` (exact), or `max_sequences` to pick a subset. `sequence_pattern` is applied as both an OSMO-input download regex and a Python-level filter.

```bash
python scripts/run_osmo.py --experiment-name retarget-taco-screw \
  --image <your-registry>/robotic-grounding:<your-tag> \
  --pool <your-pool> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=taco \
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
