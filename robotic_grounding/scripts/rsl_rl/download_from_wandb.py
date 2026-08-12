import os
import json
import wandb
import re
import argparse


def extract_checkpoint_number(filename: str) -> int:
    """Extract the checkpoint number from a filename."""
    match = re.search(r"(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else -1


def download_run(
    run_id: str, checkpoint: str | None = None, log_dir: str | None = None
) -> str | None:
    """Download a run from wandb."""
    # 0. Initialize wandb API
    api = wandb.Api()
    try:
        run = api.run(run_id)
    except wandb.errors.CommError as e:
        print(f"⚠️ [Wandb] Error: Could not access run '{run_id}': {e}")
        return None

    # 1. Get configs
    configs = run.config
    log_dir = (
        os.path.join(log_dir, configs["log_dir"].split("/")[-1])
        if log_dir is not None
        else configs["log_dir"]
    )
    alg_cfg = configs["alg_cfg"]
    policy_cfg = configs["policy_cfg"]
    env_cfg = configs["env_cfg"]
    runner_cfg = configs["runner_cfg"]

    # 2. Create a folder based on log_dir
    print(f"📥 [Wandb] Creating folder {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    # 3. Dump cfgs to log_dir/params
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    print(f"📥 [Wandb] Dumping configs to {params_dir}")
    for name, cfg in [
        ("alg_cfg", alg_cfg),
        ("policy_cfg", policy_cfg),
        ("env_cfg", env_cfg),
        ("runner_cfg", runner_cfg),
    ]:
        with open(os.path.join(params_dir, f"{name}.json"), "w") as f:
            json.dump(cfg, f, indent=4)

    # 4. Download checkpoint files to log_dir
    files = [f for f in run.files() if f.name.endswith(".pt")]
    if not files:
        print("⚠️ [Wandb] No checkpoint files found")
        return None

    if checkpoint:
        matched = [f for f in files if checkpoint in f.name]
        if not matched:
            print(f"⚠️ [Wandb] No checkpoint matching '{checkpoint}' found")
            return None
        files = matched
    else:
        # Download only the latest checkpoint (highest iteration number)
        files = [max(files, key=lambda f: extract_checkpoint_number(f.name))]

    print(f"📥 [Wandb] Downloading {[f.name for f in files]} to {log_dir}")
    for file in files:
        file.download(root=log_dir, replace=True)

    return os.path.join(log_dir, files[0].name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download a run from wandb")
    parser.add_argument(
        "run_id", help="Wandb run ID (e.g. <entity>/<project>/<run_id>)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint name to download (default: latest)",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Log directory to download to (default: None)",
    )
    args = parser.parse_args()

    download_run(args.run_id, args.checkpoint, args.log_dir)
