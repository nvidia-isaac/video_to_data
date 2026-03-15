import os
import subprocess


def run_in_container(
    image: str,
    module: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    extra_args: dict[str, object] | None = None,
    dev: bool = False,
    modules_dir: str | None = None,
    gpus: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """
    Run a Python module inside a Docker container with file arguments.

    inputs:     arg_name → host_path  (read file mounts; None values are skipped)
    outputs:    arg_name → host_path  (write file mounts; parent dirs created; None skipped)
    extra_args: arg_name → value      (non-path CLI args)
                  None or False → omit entirely
                  True          → add as a bare flag (--arg_name, no value)
                  other         → add as --arg_name str(value)
    env:        extra environment variables passed via -e

    Each unique host directory gets one volume mount at /data/<arg_name>, where the
    arg_name is taken from the first argument that references that directory. Subsequent
    arguments in the same directory reuse the same mount.
    """
    inputs  = {k: os.path.abspath(v) for k, v in inputs.items()  if v is not None}
    outputs = {k: os.path.abspath(v) for k, v in outputs.items() if v is not None}

    for path in outputs.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Map each unique host directory to a container mount point.
    # First arg to reference a directory claims the mount name.
    dir_to_mount: dict[str, str] = {}
    for arg_name, path in {**inputs, **outputs}.items():
        host_dir = os.path.dirname(path)
        if host_dir not in dir_to_mount:
            dir_to_mount[host_dir] = f"/data/{arg_name}"

    cmd = ["docker", "run", "--rm"]
    if gpus:
        cmd += ["--gpus", "all"]
    cmd += [
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
    ]
    if env:
        for key, value in env.items():
            cmd += ["-e", f"{key}={value}"]
    for host_dir, container_dir in dir_to_mount.items():
        cmd += ["-v", f"{host_dir}:{container_dir}"]
    if dev:
        if modules_dir is None:
            raise ValueError("modules_dir must be provided when dev=True")
        cmd += ["-v", f"{modules_dir}:/workspace"]

    cmd += [image, "python", "-m", module]

    for arg_name, path in {**inputs, **outputs}.items():
        host_dir = os.path.dirname(path)
        container_path = f"{dir_to_mount[host_dir]}/{os.path.basename(path)}"
        cmd += [f"--{arg_name}", container_path]

    if extra_args:
        for arg_name, value in extra_args.items():
            if value is None or value is False:
                pass
            elif value is True:
                cmd += [f"--{arg_name}"]
            else:
                cmd += [f"--{arg_name}", str(value)]

    subprocess.run(cmd, check=True)
