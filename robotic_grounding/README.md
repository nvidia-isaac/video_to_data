# Robotic Grounding

## Docker Usage

Make sure you have access to `nvcr.io/nvstaging/isaac-amr`. You can request it by asking in `#swngc-help` slack channel.

```bash
chmod +x workflow/run.sh # Make run bash executable
./workflow/run.sh build [version] # Build Docker image and tag it with [version]
./workflow/run.sh push [version] # Push Docker image to NVIDIA registry
./workflow/run.sh pull [version] # Pull Docker image from NVIDIA registry
./workflow/run.sh start [version] [gpu] # Run and enter the Container with specific version and GPU
./workflow/run.sh shell [version] [gpu] # Enter Container from new shell with specific version and GPU
./workflow/run.sh stop [version] [gpu] # Stop the Container with specific version and GPU
```

Development should be **inside the Container**.

## Retargeting
```bash
python scripts/retarget/arctic_to_sharpa.py # Check the file for arguments
python scripts/retarget/vis_retargeted.py # Need to run retargeting and save results first
```

## RL training
```bash
python scripts/rsl_rl/dummy_agent.py
python scripts/rsl_rl/train.py
python scripts/rsl_rl/play.py
```

## RL Tasks
- `Sharpa-V2P-v0-Play`
- `Sharpa-V2P-v0`
