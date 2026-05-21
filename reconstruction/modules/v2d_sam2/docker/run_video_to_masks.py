"""Host-side wrapper for v2d.sam2.lib.video_to_masks.

When the prompts JSON contains ``mask_path`` entries, those host paths are
outside the directories that ``run_in_container`` would mount by default
(it only mounts the parents of args it knows about: video, prompts file,
weights, masks output). To make the masks visible inside the container,
the wrapper:

  1. Reads the prompts JSON to discover every unique mask parent dir.
  2. Mounts each one read-only at a synthetic container path.
  3. Rewrites the prompts JSON (into a tempfile) with the container-side
     mask paths, and passes that rewritten JSON to the lib.

The lib itself stays simple — it opens whatever path it sees in the JSON.
"""
import json
import os
import shutil
import tempfile

from v2d.docker.container import run_in_container
from v2d.sam2.docker._config import IMAGE_NAME, MODULES_DIR


def _rewrite_prompts_for_container(
    prompts_path: str,
) -> tuple[str, list[str], str | None]:
    """Return (prompts_path_for_run, extra_volumes, tempdir_to_cleanup).

    Each unique host directory that holds a mask gets its own read-only
    mount at ``/data/mask_dir_<i>``, and each ``mask_path`` in the JSON is
    rewritten to the container-side path. The rewritten JSON is written to
    a tempfile that the caller is responsible for cleaning up (via the
    returned tempdir).

    Rewriting is unconditional whenever any ``mask_path`` exists — being
    under the prompts JSON's parent dir doesn't help, because that dir is
    mounted at a different path inside the container.

    When no prompt carries a ``mask_path``, returns the original
    prompts_path and an empty volume list (no rewrite needed).
    """
    with open(prompts_path) as f:
        data = json.load(f)
    prompt_list = data.get("prompts") or []

    mask_dirs: list[str] = []
    seen: set[str] = set()
    for p in prompt_list:
        mp = p.get("mask_path")
        if not mp:
            continue
        host_dir = os.path.dirname(os.path.abspath(mp))
        if host_dir not in seen:
            seen.add(host_dir)
            mask_dirs.append(host_dir)

    if not mask_dirs:
        return prompts_path, [], None

    host_to_container = {
        d: f"/data/mask_dir_{i}" for i, d in enumerate(mask_dirs)
    }
    extra_volumes = [f"{d}:{c}:ro" for d, c in host_to_container.items()]

    for p in prompt_list:
        mp = p.get("mask_path")
        if not mp:
            continue
        host_dir = os.path.dirname(os.path.abspath(mp))
        p["mask_path"] = os.path.join(
            host_to_container[host_dir], os.path.basename(mp),
        )

    tempdir = tempfile.mkdtemp(prefix="sam2_prompts_")
    rewritten_path = os.path.join(
        tempdir, os.path.basename(prompts_path) or "prompts.json"
    )
    with open(rewritten_path, "w") as f:
        json.dump(data, f, indent=2)
    return rewritten_path, extra_volumes, tempdir


def run_video_to_masks(
    video_path: str,
    prompts_path: str,
    masks_dir: str,
    weights_dir: str,
    dev: bool = False,
) -> None:
    rewritten_path, extra_volumes, tempdir = _rewrite_prompts_for_container(
        prompts_path,
    )
    try:
        run_in_container(
            image=IMAGE_NAME,
            module="v2d.sam2.lib.video_to_masks",
            inputs={
                "video_path":   video_path,
                "prompts_path": rewritten_path,
                "weights_dir":  weights_dir,
            },
            outputs={"masks_dir": masks_dir},
            dev=dev,
            modules_dir=MODULES_DIR,
            gpus=True,
            extra_volumes=extra_volumes or None,
        )
    finally:
        if tempdir is not None:
            shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--masks_dir", type=str, required=True, help="Output directory for masks")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to SAM2 weights directory")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_video_to_masks(args.video_path, args.prompts_path, args.masks_dir, args.weights_dir, dev=args.dev)
