# OSMO Workflows

This directory contains workflow definitions for running training and development environments on OSMO. See
[OSMO Wikipage](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/index.html) for more detail on OSMO.

## Prerequisites

### 1. Get access to isaac-amr NGC group and OSMO DLs.

1. Submit a ticket in the `#swngc-help` Slack channel by sending a message. They will send you an email giving you access to `isaac-amr` NGC group.
2. In [DLRequest](https://dlrequest/), request to join fto `access-osmo` and `access-osmo-isaac-dev`.
   - **Note:** you will likely need to ping in the `#osmo-support` channel to get your request approved.

### 2. Configure NGC
1. Download and install the [NGC CLI](https://docs.ngc.nvidia.com/cli/cmd.html).
2. Login to NGC [here](https://ngc.nvidia.com/signin).
3. Select `nvstaging` > `isaac-amr`
4. Under your profile dropdown, go to **Setup** > **Create API Key**
5. In terminal, configure NGC:
   ```bash
   ngc config set
   # Use the API key generated above
   ```
6. Login to Docker registry:
   ```bash
   docker login nvcr.io
   # Username: $oauthtoken
   # Password: <your_api_key>
   ```
7. Verify access:
   ```bash
   ngc user who
   # Check you have read and write access
   ```

### 3. Configure OSMO

1. Install the [OSMO CLI](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/getting_started/install/index.html)
2. Setup OSMO Credentials, follow the instructions [here](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/getting_started/credentials.html) to setup credentials
   - **Important:** Make sure to complete the [NVIDIA CSS setup](https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/release-6.0.x/user_guide/appendix/css/index.html#data-credentials-css) for storage access (required for workflows that use storage).

### 4. Omni Auth (Optional)

The current workflows may not require omni-auth credentials. If you are running into missing credentials errors, you can likely remove them from the workflow yaml.

If needed, follow the OSMO documentation for omni-auth setup: https://docs.omniverse.nvidia.com/nucleus/latest/config-and-info/api_tokens.html#token-generation.

## Submitting a Job to OSMO

Use the `run_osmo.py` script to build, push, and submit workflows in one command.

### Remote Development
Launch a remote development environment on OSMO cluster:

```bash
# Build, push Docker image, and submit OSMO workflow
python scripts/run_osmo.py --experiment-name <your-name> --workflow-yaml workflow/dev_env.yaml
```

Once the workflow is running:
```bash
# Port-forwarding
osmo workflow port-forward <workflow-name> dev-env --port 6000:22

# SSH into the remote
ssh root@localhost -p 6000
```

You can now develop remotely inside the OSMO container with full GPU access.

### Submit a Training Job

```bash
python scripts/run_osmo.py --experiment-name test --workflow-yaml workflow/train.yaml
```

This will:
1. Build a Docker image tagged `test`
2. Push the image to NGC registry
3. Submit the `train.yaml` workflow to OSMO

### Using an Existing Image

If you already have an image built and pushed:
```bash
python scripts/run_osmo.py --experiment-name test --workflow-yaml workflow/train.yaml \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:test
```

### Dry Run

Preview commands without executing:
```bash
python scripts/run_osmo.py --experiment-name test --workflow-yaml workflow/train.yaml --dry-run
```

## Task Library Data Storage (CSS / PDX)

Datasets for robot grounding Task Library are stored on NVIDIA CSS (PDX). Workflows download
inputs before the task starts and upload outputs after the task completes.

Object meshes are baked into the Docker image under `assets/meshes/`.

### Storage Location

```
swift://pdx.s8k.io/AUTH_team-isaac/datasets/v2d/
  human_motion_data/
    taco/
      dataset/               # Raw TACO data (Object_Poses, Hand_Poses)
      taco_loaded/            # taco_loader.py output (Parquet)
      taco_processed/         # taco_to_sharpa.py output (Parquet, retargeted)
      support_surfaces/       # reconstruct_support_surfaces.py output (USD)
    arctic/
      dataset/               # Raw ARCTIC data (.mano.npy, .object.npy)
      arctic_loaded/          # arctic_loader.py output (Parquet)
      arctic_processed/       # arctic_to_sharpa.py output (Parquet, retargeted)
      support_surfaces/       # reconstruct_support_surfaces.py output (USD)
    oakink2/
      dataset/               # Raw OakInk2 data
      oakink2_loaded/         # oakink2_loader.py output (Parquet)
      oakink2_processed/      # oakink2_to_sharpa.py output (Parquet, retargeted)
      support_surfaces/       # reconstruct_support_surfaces.py output (USD)
    hot3d/
      dataset/              # Raw Hot3D data
      hot3d_loaded/           # hot3d_loader.py output (Parquet)
      hot3d_processed/        # hot3d_to_sharpa.py output (Parquet, retargeted)
      support_surfaces/       # reconstruct_support_surfaces.py output (USD)
```

### Browsing Available Sequences

Use `list_css_sequences.py` to list sequences stored on CSS (PDX). First configure
credentials, then run the script:

```bash
# Set CSS credentials
source scripts/setup_css_env.sh

# List all datasets and stages
python scripts/list_css_sequences.py

# List a specific dataset
python scripts/list_css_sequences.py --dataset taco

# List a specific dataset and stage
python scripts/list_css_sequences.py --dataset arctic --stage loaded

# Filter sequences by regex pattern
python scripts/list_css_sequences.py --dataset taco --pattern '.*screw.*'
```

## Retargeting Sequences

The retargeting pipeline converts raw hand-object motion data into retargeted robot trajectories.
It runs in three stages:

1. **Load** — raw data (`.npy`, `.pkl`) to Parquet with MANO hand + object data
2. **Process** — loaded Parquet through IK retargeting to produce robot joint trajectories
3. **Reconstruct** — build support surface USD meshes from object still-poses in the loaded data

All workflows use the `HUMAN_MOTION_DATA_DIR` env var to resolve input paths from the
downloaded CSS data. Always pass `--image` to `run_osmo.py` to use an existing Docker image and skip rebuilding.
Replace `<your-tag>` in the examples below with the tag of your most recent image build/push (e.g. :latest).

### Full Pipeline (Load + Process + Reconstruct)

Use `retarget.yaml` to run all three stages in a single workflow:

```bash
# TACO
python scripts/run_osmo.py --experiment-name retarget-taco \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=taco

# ARCTIC
python scripts/run_osmo.py --experiment-name retarget-arctic \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=arctic

# OakInk2
python scripts/run_osmo.py --experiment-name retarget-oakink2 \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=oakink2

# HOT3D
python scripts/run_osmo.py --experiment-name retarget-hot3d \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=hot3d
```

### Single Stage Only

Use `--set stages=<stage>` to run just one stage (`load`, `process`, or `reconstruct`):

```bash
# Load only (raw data -> Parquet)
python scripts/run_osmo.py --experiment-name retarget-taco-load \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=taco --set stages=load

# Process only (IK retargeting)
python scripts/run_osmo.py --experiment-name retarget-arctic-process \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=arctic --set stages=process

# Reconstruct only (support surfaces)
python scripts/run_osmo.py --experiment-name retarget-oakink2-reconstruct \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=oakink2 --set stages=reconstruct
```

### Filtering Sequences

Filter which sequences to download and process using regex patterns, exact IDs, or a max count.
The `sequence_pattern` is used both as an OSMO input `regex` (to limit downloads) and as
a Python-level `--sequence_pattern` filter.

```bash
# Process only sequences matching a pattern
python scripts/run_osmo.py --experiment-name retarget-taco-screw \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=taco \
  --set 'sequence_pattern=.*(screw|skim_off|smear|stir).*'

# Process a single sequence by exact ID
python scripts/run_osmo.py --experiment-name retarget-taco-one \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=taco \
  --set sequence_id=taco_screw__screwdriver__toy_20231102_063

# Limit to first 10 sequences
python scripts/run_osmo.py --experiment-name retarget-taco-small \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<your-tag> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=taco \
  --set max_sequences=10
```

### Pulling Processed Data Locally

After workflows complete, use `sync_css_data.py` to download results to your local repo.
First configure credentials, then run the script:

```bash
# Set CSS credentials
source scripts/setup_css_env.sh

# Pull all outputs (loaded, processed, support_surfaces) for a dataset
python scripts/sync_css_data.py --dataset taco

# Pull only processed data
python scripts/sync_css_data.py --dataset taco --component processed

# Pull all outputs for all datasets
python scripts/sync_css_data.py --dataset all

# Pull only loaded (pre-retarget) data
python scripts/sync_css_data.py --dataset arctic --component loaded

# Pull only reconstructed support surfaces
python scripts/sync_css_data.py --dataset taco --component support_surfaces

# Filter sequences by regex pattern
python scripts/sync_css_data.py --dataset taco --component processed --pattern '.*screw.*'

# Preview what would be downloaded without downloading
python scripts/sync_css_data.py --dataset all --dry-run
```

## Managing Workflows

### List Running Workflows

```bash
osmo workflow list
```
