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

## Managing Workflows

### List Running Workflows

```bash
osmo workflow list
```
