# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from pathlib import Path


def _base_dir(path: str) -> str:
    """
    Return the deepest directory component of path that contains no glob characters.
    For plain paths this equals dirname(path). For paths like /out/*/*.glb it
    returns /out, so the container mount covers the full glob subtree.
    """
    parts = Path(path).parts
    clean = []
    for part in parts[:-1]:  # skip the filename/last component
        if "*" in part or "?" in part:
            break
        clean.append(part)
    return str(Path(*clean)) if clean else str(Path(path).parent)


def _host_paths_overlap(first: str, second: str) -> bool:
    """Return whether two mount paths are equal or one contains the other."""

    first_real = os.path.realpath(first)
    second_real = os.path.realpath(second)
    try:
        common = os.path.commonpath((first_real, second_real))
    except ValueError:
        # Different drives on platforms where commonpath cannot compare them.
        return False
    return common in (first_real, second_real)


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
    extra_volumes: list[str] | None = None,
    network_disabled: bool = False,
    gpu_device: int | None = None,
    strict_io_isolation: bool = False,
    input_directories: set[str] | None = None,
    input_files: set[str] | None = None,
    output_directories: set[str] | None = None,
    atomic_output_directories: set[str] | None = None,
) -> None:
    """
    Run a Python module inside a Docker container with file arguments.

    inputs:     arg_name → host_path  (read file mounts; None values are skipped)
    outputs:    arg_name → host_path  (write file mounts; parent dirs created; None skipped)
    extra_args: arg_name → value      (non-path CLI args)
                  None or False → omit entirely
                  True          → add as a bare flag (--arg_name, no value)
                  other         → add as --arg_name str(value)
    gpus:      legacy flag exposing every host GPU; retained for existing callers
    gpu_device: expose exactly one physical host GPU index. When supplied, this
                takes precedence over ``gpus`` and emits ``--gpus device=<index>``.
    env:        extra environment variables passed via -e
    extra_volumes: raw -v arguments appended after all other mounts (e.g.
                  anonymous volumes to preserve image-built artifacts under a dev mount)
    network_disabled: run with Docker networking disabled for offline stages.
    strict_io_isolation: reject equal or nested input/output host-directory
                         mount roots before creating output directories or
                         launching Docker. This is opt-in because some legacy
                         wrappers intentionally keep inputs and outputs under
                         one run directory.
    input_directories: names in ``inputs`` whose values are directories. Each
                       directory is mounted directly and its container argument
                       is the mount root. Unlisted values retain legacy file or
                       glob handling.
    input_files: names in ``inputs`` whose values are existing regular files.
                 Each file is bind-mounted directly and read-only, rather than
                 exposing its parent directory. This lets strict callers keep
                 outputs elsewhere beneath the file's parent without granting
                 the container access to sibling host files.
    output_directories: names in ``outputs`` whose values are directories. Each
                        directory is created and mounted directly. Unlisted
                        values retain legacy file handling through their parent.
    atomic_output_directories: names in ``outputs`` whose directory leaf must
                               not be created by the host runner. Its parent is
                               created/mounted and the leaf path is passed to
                               the container, allowing an atomic directory
                               rename into place.

    Each unique input directory gets a read-only mount at ``/data/<arg_name>``
    and each unique output directory gets a read-write mount at the same legacy
    namespace. The arg_name is taken from the first argument that references
    that directory, and subsequent arguments of the same kind reuse the mount.
    In non-strict mode, an exact shared input/output root is deliberately
    coalesced back to one read-write mount. Such a root is writable through the
    output either way, and retaining one container alias preserves legacy
    relative-symlink behavior. Callers requiring host-level isolation must opt
    into ``strict_io_isolation``, which rejects shared or nested roots.
    """
    inputs = {k: os.path.abspath(v) for k, v in inputs.items() if v is not None}
    outputs = {k: os.path.abspath(v) for k, v in outputs.items() if v is not None}
    duplicate_path_arguments = inputs.keys() & outputs.keys()
    if duplicate_path_arguments:
        raise ValueError(
            "input and output path arguments must use distinct names: "
            f"{sorted(duplicate_path_arguments)}"
        )
    input_directory_names = set(input_directories or ())
    input_file_names = set(input_files or ())
    output_directory_names = set(output_directories or ())
    atomic_output_directory_names = set(atomic_output_directories or ())
    unknown_input_paths = (input_directory_names | input_file_names) - inputs.keys()
    unknown_output_directories = (
        output_directory_names | atomic_output_directory_names
    ) - outputs.keys()
    if unknown_input_paths or unknown_output_directories:
        raise ValueError(
            "typed path argument names must identify present path arguments: "
            f"inputs={sorted(unknown_input_paths)}, "
            f"outputs={sorted(unknown_output_directories)}"
        )
    ambiguous_input_types = input_directory_names & input_file_names
    if ambiguous_input_types:
        raise ValueError(
            "input arguments cannot be both directories and exact files: "
            f"{sorted(ambiguous_input_types)}"
        )
    ambiguous_output_directories = (
        output_directory_names & atomic_output_directory_names
    )
    if ambiguous_output_directories:
        raise ValueError(
            "output directory arguments cannot be both direct and atomic: "
            f"{sorted(ambiguous_output_directories)}"
        )
    invalid_input_files = sorted(
        name for name in input_file_names if not os.path.isfile(inputs[name])
    )
    if invalid_input_files:
        raise FileNotFoundError(
            "exact input files must exist and be regular files: "
            f"{invalid_input_files}"
        )

    # Build input and output namespaces independently. Strict callers keep them
    # separate; compatibility mode later coalesces exact shared directory roots.
    input_dir_to_mount: dict[str, str] = {}
    for arg_name, path in inputs.items():
        if arg_name in input_file_names:
            continue
        host_dir = path if arg_name in input_directory_names else _base_dir(path)
        if host_dir not in input_dir_to_mount:
            input_dir_to_mount[host_dir] = f"/data/{arg_name}"
    input_file_to_mount: dict[str, str] = {}
    for arg_name, path in inputs.items():
        if arg_name not in input_file_names:
            continue
        if path not in input_file_to_mount:
            input_file_to_mount[path] = (
                f"/data/{arg_name}/{os.path.basename(path)}"
            )
    output_dir_to_mount: dict[str, str] = {}
    for arg_name, path in outputs.items():
        host_dir = path if arg_name in output_directory_names else _base_dir(path)
        if host_dir not in output_dir_to_mount:
            output_dir_to_mount[host_dir] = f"/data/{arg_name}"
    overlapping_host_dirs = sorted(
        (input_path, output_dir)
        for input_path in (*input_dir_to_mount, *input_file_to_mount)
        for output_dir in output_dir_to_mount
        if _host_paths_overlap(input_path, output_dir)
    )
    if strict_io_isolation and overlapping_host_dirs:
        raise ValueError(
            "strict I/O isolation forbids input/output host-directory overlap: "
            f"{overlapping_host_dirs}"
        )

    # Preserve one namespace for exact shared roots in compatibility mode.
    # Keeping two aliases would not make the input read-only (the RW alias sees
    # the same host data), but it would corrupt relative symlinks persisted by
    # legacy stages because their paths would encode the container-only aliases.
    shared_host_dirs = (
        set(input_dir_to_mount) & set(output_dir_to_mount)
        if not strict_io_isolation
        else set()
    )
    for host_dir in shared_host_dirs:
        output_dir_to_mount[host_dir] = input_dir_to_mount[host_dir]

    target_hosts: dict[str, set[str]] = {}
    for mapping in (input_dir_to_mount, input_file_to_mount, output_dir_to_mount):
        for host_path, target in mapping.items():
            target_hosts.setdefault(target, set()).add(host_path)
    duplicate_targets = {
        target for target, host_paths in target_hosts.items() if len(host_paths) > 1
    }
    if duplicate_targets:
        raise ValueError(
            "input and output arguments must not claim the same container mount "
            f"target: {sorted(duplicate_targets)}"
        )

    for arg_name, path in outputs.items():
        directory = path if arg_name in output_directory_names else _base_dir(path)
        os.makedirs(directory, exist_ok=True)

    cmd = ["docker", "run", "--rm"]
    if network_disabled:
        cmd += ["--network", "none"]
    if gpu_device is not None:
        if (
            isinstance(gpu_device, bool)
            or not isinstance(gpu_device, int)
            or gpu_device < 0
        ):
            raise ValueError("gpu_device must be a non-negative physical GPU index")
        cmd += ["--runtime=nvidia", "--gpus", f"device={gpu_device}"]
    elif gpus:
        cmd += ["--runtime=nvidia", "--gpus", "all"]
    cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    if env:
        for key, value in env.items():
            cmd += ["-e", f"{key}={value}"]
    for host_dir, container_dir in input_dir_to_mount.items():
        if host_dir in shared_host_dirs:
            continue
        cmd += ["-v", f"{host_dir}:{container_dir}:ro"]
    for host_file, container_file in input_file_to_mount.items():
        cmd += ["-v", f"{host_file}:{container_file}:ro"]
    for host_dir, container_dir in output_dir_to_mount.items():
        cmd += ["-v", f"{host_dir}:{container_dir}"]
    if dev:
        if modules_dir is None:
            raise ValueError("modules_dir must be provided when dev=True")
        cmd += ["-v", f"{modules_dir}:/workspace"]
    if extra_volumes:
        for vol in extra_volumes:
            cmd += ["-v", vol]

    cmd += [image, "python", "-m", module]

    for arg_name, path in inputs.items():
        if arg_name in input_file_names:
            cmd += [f"--{arg_name}", input_file_to_mount[path]]
            continue
        host_dir = path if arg_name in input_directory_names else _base_dir(path)
        container_path = input_dir_to_mount[host_dir]
        if arg_name not in input_directory_names:
            rel = os.path.relpath(path, host_dir)
            container_path = f"{container_path}/{rel}"
        cmd += [f"--{arg_name}", container_path]
    for arg_name, path in outputs.items():
        host_dir = path if arg_name in output_directory_names else _base_dir(path)
        container_path = output_dir_to_mount[host_dir]
        if arg_name not in output_directory_names:
            rel = os.path.relpath(path, host_dir)
            container_path = f"{container_path}/{rel}"
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
