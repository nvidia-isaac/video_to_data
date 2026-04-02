# Video to Data

Monorepo for Video to Data (V2D) — converting human demonstration videos into simulation environments and physics-grounded robot training data.

## Packages

| Package | Description |
|---|---|
| [`robotic_grounding/`](robotic_grounding/) | RL training framework built on NVIDIA Isaac Lab 2.3.1 with RSL-RL (PPO) |

## Development

Each package has its own `workflow/` directory with Docker and setup scripts. See each package's README for details.

```bash
cd robotic_grounding

# First-time host setup
bash workflow/setup_deps.sh

# Build and start Docker container
./workflow/run.sh build [version]
./workflow/run.sh start [version] [gpu_id]
```
