# OSMO Workflows

Workflow definitions for running training, retargeting, and development environments on OSMO. See the [OSMO user guide](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/index.html) for platform details.

**See also:** [data_pipeline.md](data_pipeline.md) for the end-to-end data flow (raw → retargeted → trained), [data_storage.md](data_storage.md) for storage layout and output-download commands.

## Prerequisites

### 1. Access

- `isaac-amr` NGC group — open a ticket in `#swngc-help`.
- OSMO DLs `access-osmo` and `access-osmo-isaac-dev` via [DLRequest](https://dlrequest/) (ping `#osmo-support` to get approved).

### 2. Configure NGC

1. Install the [NGC CLI](https://docs.ngc.nvidia.com/cli/cmd.html) and [sign in](https://ngc.nvidia.com/signin).
2. Select `nvstaging` > `isaac-amr`, then **Setup** > **Create API Key**.
3. Configure and verify:
   ```bash
   ngc config set              # use the API key
   docker login nvcr.io        # user: $oauthtoken, pass: <api_key>
   ngc user who                # confirm read+write
   ```

### 3. Configure OSMO

Install the [OSMO CLI](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/getting_started/install/index.html) and set up [credentials](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/getting_started/credentials.html), including the [CSS setup](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/appendix/css/index.html#data-credentials-css) for storage access.

Omni-auth credentials are generally not needed; if a workflow complains about them, remove the credentials block from the yaml or follow the [token guide](https://docs.omniverse.nvidia.com/nucleus/latest/config-and-info/api_tokens.html#token-generation).

## Submitting a Job

`run_osmo.py` builds + pushes + submits in one command. Add `--image <tag>` to skip rebuilding with an existing image, or `--dry-run` to preview.

### Remote development

```bash
python scripts/run_osmo.py --experiment-name <your-name> --workflow-yaml workflow/dev_env.yaml

# Once running:
osmo workflow port-forward <workflow-name> dev-env --port 6000:22
ssh root@localhost -p 6000
```

### Training

```bash
python scripts/run_osmo.py --experiment-name <name> --workflow-yaml workflow/train.yaml
```

### Retargeting

See [data_pipeline.md](data_pipeline.md) for what each stage does. Stages (`load`, `process`, `reconstruct`, `visualize`, `video`) can run together or individually. Outputs from one run are published as a single new version of the `v2d_{dataset}_retarget_exp_200` OSMO dataset — see [data_storage.md](data_storage.md) to pull them locally.

```bash
# Full pipeline (works for any registered dataset: taco, arctic, oakink2, hot3d, h2o, grab, dexycb)
python scripts/run_osmo.py --experiment-name retarget-<dataset> \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=<dataset>

# Run only one stage
python scripts/run_osmo.py --experiment-name retarget-<dataset>-<stage> \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=<dataset> --set stages=<stage>
```

#### Filtering sequences

Use `sequence_pattern` (regex), `sequence_id` (exact), or `max_sequences` to pick a subset. `sequence_pattern` is applied as both an OSMO-input download regex and a Python-level filter.

```bash
python scripts/run_osmo.py --experiment-name retarget-taco-screw \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
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
