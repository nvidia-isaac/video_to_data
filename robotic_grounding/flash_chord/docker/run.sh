#!/bin/bash
# Manage the flash_chord Docker container (modeled on video_to_data/.../workflow/run.sh).
#
# Usage:
#   docker/run.sh build [tag] [gpu]           - Build the image flash_chord:<tag>
#   docker/run.sh start [tag] [gpu]           - Start a detached container and open a shell
#   docker/run.sh shell [tag] [gpu]           - Open another shell in the running container
#   docker/run.sh exec  [tag] [gpu] -- <cmd>  - Run a command in the running container
#   docker/run.sh stop  [tag] [gpu]           - Stop (and remove) the container  [alias: kill]
#
#   tag : image tag, i.e. flash_chord:<tag> (e.g. latest, user_xyz).  Default: latest
#   gpu : GPU index (e.g. 0) or "all".                                 Default: 0
#
# Build args CUDA_TAG / UV_TAG can be overridden via the environment.
set -euo pipefail

CMD="${1:-}"
if [[ $# -gt 0 ]]; then
    shift
fi
TAG="latest"
GPU_DEVICE="0"
REMAINING_ARGS=()

CUDA_TAG="${CUDA_TAG:-13.0.2-devel-ubuntu24.04}"
UV_TAG="${UV_TAG:-0.11.21}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_V2D_ASSETS_DIR="${REPO_ROOT}/../source/robotic_grounding/robotic_grounding/assets"
V2D_ASSETS_MOUNT=()
if [[ -n "${V2D_ASSETS_DIR:-}" ]]; then
    if [[ ! -d "${V2D_ASSETS_DIR}" ]]; then
        echo "V2D_ASSETS_DIR is not a directory: ${V2D_ASSETS_DIR}" >&2
        exit 1
    fi
    V2D_ASSETS_MOUNT=(-v "${V2D_ASSETS_DIR}:/v2d/assets:ro")
elif [[ -d "${DEFAULT_V2D_ASSETS_DIR}" ]]; then
    V2D_ASSETS_MOUNT=(-v "${DEFAULT_V2D_ASSETS_DIR}:/v2d/assets:ro")
fi

usage() {
    cat >&2 <<USAGE
Usage: docker/run.sh {build|start|shell|exec|stop} [tag] [gpu]

  build [tag] [gpu]              Build the image flash_chord:<tag>
  start [tag] [gpu]              Start a detached container and open a shell
  shell [tag] [gpu]              Open another shell in the running container
  exec  [tag] [gpu] -- <cmd>     Run a command in the running container
  stop  [tag] [gpu]              Stop (and remove) the container  [alias: kill]

  tag : image tag, i.e. flash_chord:<tag> (e.g. latest, user_xyz).  Default: latest
  gpu : GPU index (e.g. 0) or "all".                                 Default: 0

Environment:
  CUDA_TAG / UV_TAG               Override Docker build base versions
  V2D_ASSETS_DIR                  Optional V2D assets root mounted read-only at /v2d/assets

Examples:
  docker/run.sh build                 # build flash_chord:latest
  docker/run.sh build user_xyz       # build flash_chord:user_xyz
  docker/run.sh start latest 0        # start container + open shell
  docker/run.sh shell                 # open another shell (latest, gpu 0)
  docker/run.sh exec latest 0 -- python scripts/debug/view_scene.py --parquet /data/reference.parquet --viewer gl
  docker/run.sh exec -- pytest tests/test_configuration.py -q
  docker/run.sh stop latest 0
USAGE
}

parse_optional_tag_gpu() {
    if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
        TAG="$1"
        shift
    fi
    if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
        GPU_DEVICE="$1"
        shift
    fi
    REMAINING_ARGS=("$@")
}

parse_optional_tag_gpu "$@"

IMAGE_NAME="flash_chord:${TAG}"
CONTAINER_NAME="flash_chord-${TAG}-gpu${GPU_DEVICE}"

# `--runtime=nvidia` is required on this host for the NVIDIA Container Toolkit to inject
# the GPU *and* the GL/EGL driver libs (`--gpus` alone leaves the container CPU-only).
if [[ "${GPU_DEVICE}" == "all" ]]; then
    GPU_FLAG=(--runtime=nvidia --gpus all)
else
    GPU_FLAG=(--runtime=nvidia --gpus "device=${GPU_DEVICE}")
fi

is_running() {
    docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"
}

case "${CMD}" in
    build)
        echo "Building ${IMAGE_NAME}"
        DOCKER_BUILDKIT=1 docker build \
            -f "${REPO_ROOT}/docker/Dockerfile" \
            --build-arg "CUDA_TAG=${CUDA_TAG}" \
            --build-arg "UV_TAG=${UV_TAG}" \
            -t "${IMAGE_NAME}" \
            "${REPO_ROOT}"
        echo "Build complete: ${IMAGE_NAME}"
        ;;

    start)
        if is_running; then
            echo "Container ${CONTAINER_NAME} already running. Opening a shell..."
        else
            if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
                echo "Image ${IMAGE_NAME} not found. Build it first:" >&2
                echo "  docker/run.sh build ${TAG}" >&2
                exit 1
            fi

            echo "Starting ${CONTAINER_NAME} (image: ${IMAGE_NAME})"

            # Run as the host user so files created in the bind-mounted repo are
            # owned by you, not root. A per-container passwd/group gives that UID a
            # name. No secrets are written; /etc/shadow is never mounted.
            HOST_UID="$(id -u)"; HOST_GID="$(id -g)"
            XHOST_USER="$(id -un)"
            HOST_USER="${XHOST_USER}"; HOST_GROUP="$(id -gn)"
            [[ "${HOST_USER}" =~ ^[a-z_][a-z0-9_-]*$ ]] || HOST_USER="user"
            [[ "${HOST_GROUP}" =~ ^[a-z_][a-z0-9_-]*$ ]] || HOST_GROUP="usergroup"

            IDMAP_DIR="${HOME:-/tmp}/.cache/flash_chord/${CONTAINER_NAME}"
            mkdir -p "${IDMAP_DIR}"
            printf 'root:x:0:0:root:/root:/bin/bash\n%s:x:%s:%s:%s:/tmp:/bin/bash\n' \
                "${HOST_USER}" "${HOST_UID}" "${HOST_GID}" "${HOST_USER}" > "${IDMAP_DIR}/passwd"
            printf 'root:x:0:\n%s:x:%s:\n' "${HOST_GROUP}" "${HOST_GID}" > "${IDMAP_DIR}/group"
            chmod 0644 "${IDMAP_DIR}/passwd" "${IDMAP_DIR}/group"

            # Let the container reach the host X server (native GUI) with local-user scope.
            if [[ -n "${DISPLAY:-}" ]]; then
                xhost +SI:localuser:"${XHOST_USER}" >/dev/null 2>&1 || true
            fi

            # Render backend (set here so every `docker exec` shell inherits it):
            # native GLX when a display is present, headless EGL otherwise.
            if [[ -n "${DISPLAY:-}" ]]; then
                RENDER_ENV=(-e "MUJOCO_GL=glfw" -e "PYOPENGL_PLATFORM=glx")
            else
                RENDER_ENV=(-e "MUJOCO_GL=egl" -e "PYOPENGL_PLATFORM=egl")
            fi

            # Optional SSH agent forwarding (handy for git over SSH in-container).
            SSH_ENV=()
            if [[ -n "${SSH_AUTH_SOCK:-}" && -S "${SSH_AUTH_SOCK}" ]]; then
                SSH_ENV=(-v "${SSH_AUTH_SOCK}:/ssh-agent" -e "SSH_AUTH_SOCK=/ssh-agent")
            fi

            # WANDB_API_KEY: use env if set (must be exported), else read from host home.
            WANDB_ENV=()
            WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
            if [[ -z "${WANDB_API_KEY_VALUE}" && -f "${HOME}/.wandb_api_key" ]]; then
                WANDB_API_KEY_VALUE="$(cat "${HOME}/.wandb_api_key")"
            fi
            if [[ -n "${WANDB_API_KEY_VALUE}" ]]; then
                WANDB_ENV=(-e "WANDB_API_KEY=${WANDB_API_KEY_VALUE}")
            fi

            docker run -dit --rm \
                "${GPU_FLAG[@]}" \
                --name "${CONTAINER_NAME}" \
                --network host \
                --user "${HOST_UID}:${HOST_GID}" \
                -e HOME=/tmp \
                -e "USER=${HOST_USER}" \
                -e NVIDIA_DRIVER_CAPABILITIES=all \
                -e "DISPLAY=${DISPLAY:-}" \
                "${RENDER_ENV[@]}" \
                "${SSH_ENV[@]}" \
                "${WANDB_ENV[@]}" \
                "${V2D_ASSETS_MOUNT[@]}" \
                -v "${REPO_ROOT}:/workspace" \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v "${IDMAP_DIR}/passwd:/etc/passwd:ro" \
                -v "${IDMAP_DIR}/group:/etc/group:ro" \
                --entrypoint /bin/bash \
                "${IMAGE_NAME}" >/dev/null
        fi

        exec docker exec -it "${CONTAINER_NAME}" /bin/bash
        ;;

    shell)
        if ! is_running; then
            echo "Error: ${CONTAINER_NAME} is not running." >&2
            echo "Start it first:  docker/run.sh start ${TAG} ${GPU_DEVICE}" >&2
            exit 1
        fi
        exec docker exec -it "${CONTAINER_NAME}" /bin/bash
        ;;

    exec)
        EXEC_ARGS=("${REMAINING_ARGS[@]}")
        if [[ "${EXEC_ARGS[0]:-}" == "--" ]]; then
            EXEC_ARGS=("${EXEC_ARGS[@]:1}")
        fi
        if [[ ${#EXEC_ARGS[@]} -eq 0 ]]; then
            echo "Usage: docker/run.sh exec [tag] [gpu] -- <command>" >&2
            exit 1
        fi
        if ! is_running; then
            echo "Error: ${CONTAINER_NAME} is not running." >&2
            echo "Start it first:  docker/run.sh start ${TAG} ${GPU_DEVICE}" >&2
            exit 1
        fi
        exec docker exec -it "${CONTAINER_NAME}" "${EXEC_ARGS[@]}"
        ;;

    stop|kill)
        if ! is_running; then
            echo "Container ${CONTAINER_NAME} is not running."
            exit 0
        fi
        docker stop "${CONTAINER_NAME}" >/dev/null
        echo "Stopped ${CONTAINER_NAME} (removed; it was started with --rm)."
        ;;

    *)
        usage
        exit 1
        ;;
esac
