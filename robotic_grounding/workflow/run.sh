#!/bin/bash

# Script to manage robotic-grounding Docker container
# Usage:
#   ./run.sh build [version]              - Build x86_64 image (default version: latest)
#   ./run.sh build-aarch64 [version]      - Build aarch64 image (default version: latest)
#   ./run.sh push [version]               - Push x86_64 image to NVIDIA registry
#   ./run.sh push-aarch64 [version]       - Push aarch64 image to NVIDIA registry
#   ./run.sh pull [version]               - Pull x86_64 image from NVIDIA registry
#   ./run.sh pull-aarch64 [version]       - Pull aarch64 image from NVIDIA registry
#   ./run.sh start [version] [gpu]        - Start x86_64 container (default version: latest, gpu: 0)
#   ./run.sh start-aarch64 [version] [gpu] - Start aarch64 container
#   ./run.sh shell [version] [gpu]        - Enter shell of a running container
#   ./run.sh shell-aarch64 [version] [gpu] - Enter shell of a running aarch64 container
#   ./run.sh exec [version] [gpu] -- <cmd>  - Run a command in a running container
#   ./run.sh e2e-run [version] [gpu] [mounts] -- <cmd> - Internal noninteractive E2E operation
#   ./run.sh stop [version] [gpu]         - Stop the running container
#   ./run.sh stop-aarch64 [version] [gpu] - Stop the running aarch64 container

set -e

# Detect arch suffix from the subcommand — affects IMAGE_NAME and CONTAINER_NAME.
# e.g. start-aarch64 uses robotic-grounding:latest-aarch64 and a distinct container name.
ARCH_SUFFIX=""
CMD="$1"
if [[ "$CMD" == *-aarch64 ]]; then
    ARCH_SUFFIX="-aarch64"
    CMD="${CMD%-aarch64}"
fi

VERSION=${2:-latest}
GPU_DEVICE=${3:-0}
E2E_MODE="0"
E2E_RUN_DATA=""
E2E_MANO_DIR=""
E2E_CHECKPOINT=""
E2E_WORKDIR="/workspace/video_to_data/robotic_grounding"
E2E_RECREATE_ON_MOUNT_CHANGE="0"
E2E_COMMAND=()

if [ "$CMD" = "e2e-run" ]; then
    E2E_MODE="1"
    shift 3 2>/dev/null || true
    while [ $# -gt 0 ]; do
        case "$1" in
            --run-data)
                [ $# -ge 2 ] || { echo "Error: --run-data requires a path." >&2; exit 2; }
                E2E_RUN_DATA="$2"
                shift 2
                ;;
            --mano-dir)
                [ $# -ge 2 ] || { echo "Error: --mano-dir requires a path." >&2; exit 2; }
                E2E_MANO_DIR="$2"
                shift 2
                ;;
            --checkpoint)
                [ $# -ge 2 ] || { echo "Error: --checkpoint requires a path." >&2; exit 2; }
                E2E_CHECKPOINT="$2"
                shift 2
                ;;
            --workdir)
                [ $# -ge 2 ] || { echo "Error: --workdir requires a path." >&2; exit 2; }
                E2E_WORKDIR="$2"
                shift 2
                ;;
            --recreate-on-mount-change)
                E2E_RECREATE_ON_MOUNT_CHANGE="1"
                shift
                ;;
            --)
                shift
                E2E_COMMAND=("$@")
                break
                ;;
            *)
                echo "Error: unknown e2e-run option: $1" >&2
                exit 2
                ;;
        esac
    done
    [ -d "$E2E_RUN_DATA" ] || { echo "Error: E2E run-data directory does not exist: $E2E_RUN_DATA" >&2; exit 2; }
    if [ -n "$E2E_MANO_DIR" ]; then
        [ -d "$E2E_MANO_DIR" ] || { echo "Error: MANO directory does not exist: $E2E_MANO_DIR" >&2; exit 2; }
    fi
    if [ -n "$E2E_CHECKPOINT" ]; then
        [ -f "$E2E_CHECKPOINT" ] || { echo "Error: checkpoint does not exist: $E2E_CHECKPOINT" >&2; exit 2; }
    fi
    [ ${#E2E_COMMAND[@]} -gt 0 ] || { echo "Error: e2e-run requires a command after --." >&2; exit 2; }
    E2E_RUN_DATA="$(cd "$E2E_RUN_DATA" && pwd -P)"
    if [ -n "$E2E_MANO_DIR" ]; then
        E2E_MANO_DIR="$(cd "$E2E_MANO_DIR" && pwd -P)"
    fi
    if [ -n "$E2E_CHECKPOINT" ]; then
        E2E_CHECKPOINT_DIR="$(cd "$(dirname "$E2E_CHECKPOINT")" && pwd -P)"
        E2E_CHECKPOINT="${E2E_CHECKPOINT_DIR}/$(basename "$E2E_CHECKPOINT")"
    fi
    CMD="start"
fi

IMAGE_NAME="robotic-grounding${ARCH_SUFFIX}:${VERSION}"
CONTAINER_NAME="robotic-grounding${ARCH_SUFFIX}-${VERSION}-gpu${GPU_DEVICE}"
NGC_LOCATION="nvcr.io/nvstaging/isaac-amr"

wait_for_container_removal() {
    local wait_index
    for wait_index in {1..100}; do
        if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    echo "Error: timed out waiting for container removal: ${CONTAINER_NAME}" >&2
    return 1
}

case "$CMD" in
    build)
        echo "Building Docker image: ${IMAGE_NAME}"
        cd "$(dirname "$0")/.."

        if [ -n "$ARCH_SUFFIX" ]; then
            SETUP_OK=true

            if ! docker buildx version &>/dev/null; then
                echo ""
                echo "ERROR: docker buildx plugin is not installed."
                echo "  Fix:  sudo apt-get install -y docker-buildx-plugin"
                SETUP_OK=false
            fi

            if ${SETUP_OK} && ! docker buildx ls 2>/dev/null | grep -q "multiarch"; then
                echo ""
                echo "WARNING: 'multiarch' buildx builder not found."
                echo "  Fix:  docker buildx create --name multiarch --use"
                echo "        docker buildx inspect --bootstrap"
                SETUP_OK=false
            fi

            if ${SETUP_OK} && ! docker buildx ls 2>/dev/null | grep -q "linux/arm64"; then
                echo ""
                echo "WARNING: arm64 platform not available in buildx."
                echo "  Fix:  docker run --privileged --rm tonistiigi/binfmt --install arm64"
                SETUP_OK=false
            fi

            if ! ${SETUP_OK}; then
                echo ""
                echo "One-time setup (run in order):"
                echo "  1. sudo apt-get install -y docker-buildx-plugin"
                echo "  2. docker run --privileged --rm tonistiigi/binfmt --install arm64"
                echo "  3. docker buildx create --name multiarch --use"
                echo "  4. docker buildx inspect --bootstrap"
                exit 1
            fi

            docker buildx build --platform linux/arm64 --load \
                -t ${IMAGE_NAME} -f workflow/Dockerfile.aarch64 .
        else
            docker build -t ${IMAGE_NAME} -f workflow/Dockerfile .
        fi

        echo "Build complete: ${IMAGE_NAME}"
        ;;

    push)
        echo "Pushing ${IMAGE_NAME} to NVIDIA registry..."
        docker tag ${IMAGE_NAME} ${NGC_LOCATION}/${IMAGE_NAME}
        docker push ${NGC_LOCATION}/${IMAGE_NAME}
        echo "Push complete!"
        echo "Removing local images to free disk space..."
        docker rmi ${NGC_LOCATION}/${IMAGE_NAME} ${IMAGE_NAME} || true
        ;;

    pull)
        echo "Pulling ${NGC_LOCATION}/${IMAGE_NAME}..."
        docker pull ${NGC_LOCATION}/${IMAGE_NAME}
        docker tag ${NGC_LOCATION}/${IMAGE_NAME} ${IMAGE_NAME}
        echo "Pull complete: ${IMAGE_NAME}"
        ;;

    start)
        echo "Starting container: ${CONTAINER_NAME} (image: ${IMAGE_NAME})"

        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Container is already running."
            if [ "$E2E_MODE" = "1" ]; then
                REPO_MOUNT_SOURCE="$(cd "$(dirname "$0")/../.." && pwd -P)"
                MOUNT_TABLE="$(docker inspect -f '{{range .Mounts}}{{printf "%s|%s\n" .Source .Destination}}{{end}}' "${CONTAINER_NAME}")"
                EXPECTED_MOUNTS=(
                    "${REPO_MOUNT_SOURCE}|/workspace/video_to_data"
                    "${E2E_RUN_DATA}|/workspace/e2e"
                )
                if [ -n "$E2E_MANO_DIR" ]; then
                    EXPECTED_MOUNTS+=("${E2E_MANO_DIR}|/workspace/mano")
                fi
                if [ -n "$E2E_CHECKPOINT" ]; then
                    EXPECTED_MOUNTS+=("${E2E_CHECKPOINT}|/workspace/checkpoints/$(basename "$E2E_CHECKPOINT")")
                fi
                MISSING_MOUNT=""
                for EXPECTED_MOUNT in "${EXPECTED_MOUNTS[@]}"; do
                    if ! grep -Fxq "$EXPECTED_MOUNT" <<<"$MOUNT_TABLE"; then
                        MISSING_MOUNT="$EXPECTED_MOUNT"
                        break
                    fi
                done
                if [ -n "$MISSING_MOUNT" ]; then
                    if [ "$E2E_RECREATE_ON_MOUNT_CHANGE" = "1" ]; then
                        echo "Recreating ${CONTAINER_NAME}; required mount changed: ${MISSING_MOUNT}"
                        docker stop "$CONTAINER_NAME"
                        docker rm "$CONTAINER_NAME" 2>/dev/null || true
                        wait_for_container_removal
                    else
                        echo "Error: refusing to reuse ${CONTAINER_NAME}; missing mount ${MISSING_MOUNT}" >&2
                        echo "Stop it with './workflow/run.sh stop ${VERSION} ${GPU_DEVICE}' and retry." >&2
                        exit 2
                    fi
                fi
            fi
        fi

        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            # A previous startup may have left a stopped named container (for example,
            # if Docker exited between ``stop`` and the replacement ``run``). Remove only
            # that exact stopped lifecycle container before reusing its deterministic name.
            if docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' \
                | grep -q "^${CONTAINER_NAME}$"; then
                docker rm "$CONTAINER_NAME" 2>/dev/null || true
                wait_for_container_removal
            fi
            echo "Creating and starting new container..."
            cd "$(dirname "$0")/.."
            xhost +local:docker > /dev/null 2>&1 || true

            SSH_AGENT_MOUNT=""
            SSH_AGENT_ENV=""
            if [ -n "${SSH_AUTH_SOCK}" ] && [ -S "${SSH_AUTH_SOCK}" ]; then
                SSH_AGENT_MOUNT="-v ${SSH_AUTH_SOCK}:/ssh-agent"
                SSH_AGENT_ENV="-e SSH_AUTH_SOCK=/ssh-agent"
            else
                echo "Warning: SSH agent socket not found; agent forwarding disabled."
            fi

            # WANDB_API_KEY: use env if set (must be exported), else read from host home
            WANDB_API_KEY_VALUE="${WANDB_API_KEY}"
            if [ -z "${WANDB_API_KEY_VALUE}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
                WANDB_API_KEY_VALUE=$(cat "${HOME}/.wandb_api_key")
            fi
            WANDB_API_KEY_ENV=""
            if [ -n "${WANDB_API_KEY_VALUE}" ]; then
                WANDB_API_KEY_ENV="-e WANDB_API_KEY=${WANDB_API_KEY_VALUE}"
            fi

            # Optional: mount external human_motion_data datasets (e.g. from another repo).
            # Set HUMAN_MOTION_DATA_DIR on the host to an absolute path before calling run.sh:
            #   HUMAN_MOTION_DATA_DIR=/path/to/human_motion_data ./workflow/run.sh start
            # Mount each dataset separately so an external root does not hide datasets
            # committed in the image or repository mount (for example, whole_body/).
            DATA_MOUNTS=()
            CONTAINER_DATA_DIR="/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data"
            if [ -n "${HUMAN_MOTION_DATA_DIR}" ]; then
                for DATASET_DIR in "${HUMAN_MOTION_DATA_DIR}"/*/; do
                    DATASET_DIR="${DATASET_DIR%/}"
                    [ -d "${DATASET_DIR}" ] || continue
                    DATASET_NAME="$(basename "${DATASET_DIR}")"
                    DATA_MOUNTS+=(-v "${DATASET_DIR}:${CONTAINER_DATA_DIR}/${DATASET_NAME}")
                    echo "Mounting external data: ${DATASET_DIR} → ${CONTAINER_DATA_DIR}/${DATASET_NAME}"
                done
            fi

            # Build per-container passwd/group so the host UID has a name
            # inside the container (avoids the "I have no name!" bash prompt
            # and keeps tools that call getpwuid() happy: git, ssh, etc.).
            # No secrets are written: passwords use the "x" placeholder and
            # /etc/shadow is NOT mounted into the container.
            HOST_UID="$(id -u)"
            HOST_GID="$(id -g)"
            HOST_USERNAME="$(id -un)"
            HOST_GROUPNAME="$(id -gn)"
            if ! [[ "${HOST_USERNAME}" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
                HOST_USERNAME="user"
            fi
            if ! [[ "${HOST_GROUPNAME}" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
                HOST_GROUPNAME="usergroup"
            fi
            CACHE_ROOT="${HOME:-/tmp}/.cache/robotic-grounding"
            CONTAINER_PASSWD_DIR="${CACHE_ROOT}/${CONTAINER_NAME}"
            mkdir -p "${CONTAINER_PASSWD_DIR}"
            chmod 0700 "${CACHE_ROOT}" "${CONTAINER_PASSWD_DIR}" 2>/dev/null || true
            umask 0022
            # Per-container shadowed /etc/passwd and /etc/group. We add an
            # entry for the Isaac Sim image's owner (uid/gid 1234, name
            # isaac-sim) so that --group-add 1234 below resolves to a real
            # name inside the container; cosmetic but keeps tools that call
            # getgrgid() happy.
            cat > "${CONTAINER_PASSWD_DIR}/passwd" <<EOF
root:x:0:0:root:/root:/bin/bash
isaac-sim:x:1234:1234::/isaac-sim:/bin/bash
${HOST_USERNAME}:x:${HOST_UID}:${HOST_GID}:${HOST_USERNAME}:/tmp:/bin/bash
EOF
            cat > "${CONTAINER_PASSWD_DIR}/group" <<EOF
root:x:0:
isaac-sim:x:1234:${HOST_USERNAME}
${HOST_GROUPNAME}:x:${HOST_GID}:
EOF
            chmod 0644 "${CONTAINER_PASSWD_DIR}/passwd" "${CONTAINER_PASSWD_DIR}/group"

            # Per-container writable dirs for Kit's data/cache/logs. The
            # base image owns /isaac-sim/kit/{data,cache,logs} as
            # uid:gid 1234:1234 mode 0750/0755, so as a non-root user we
            # can't write there even with --group-add 1234 (group has r-x,
            # not w). Bind-mount writable host dirs over those paths so
            # Kit's user.config.json, shader cache, and log files land
            # somewhere we own.
            KIT_DATA_DIR="${CONTAINER_PASSWD_DIR}/kit-data"
            KIT_CACHE_DIR="${CONTAINER_PASSWD_DIR}/kit-cache"
            KIT_LOGS_DIR="${CONTAINER_PASSWD_DIR}/kit-logs"
            mkdir -p "${KIT_DATA_DIR}" "${KIT_CACHE_DIR}" "${KIT_LOGS_DIR}"

            E2E_MOUNTS=()
            DOCKER_TTY_ARGS=(-it)
            CONTAINER_COMMAND=()
            if [ "$E2E_MODE" = "1" ]; then
                DOCKER_TTY_ARGS=()
                CONTAINER_COMMAND=(-lc "exec sleep infinity")
                E2E_MOUNTS=(-v "${E2E_RUN_DATA}:/workspace/e2e")
                echo "Mounting E2E run data: ${E2E_RUN_DATA} → /workspace/e2e"
                if [ -n "$E2E_MANO_DIR" ]; then
                    E2E_MOUNTS+=(-v "${E2E_MANO_DIR}:/workspace/mano:ro")
                    echo "Mounting MANO models: ${E2E_MANO_DIR} → /workspace/mano (read-only)"
                fi
                if [ -n "$E2E_CHECKPOINT" ]; then
                    E2E_MOUNTS+=(-v "${E2E_CHECKPOINT}:/workspace/checkpoints/$(basename "$E2E_CHECKPOINT"):ro")
                    echo "Mounting checkpoint: ${E2E_CHECKPOINT} → /workspace/checkpoints/$(basename "$E2E_CHECKPOINT") (read-only)"
                fi
            fi

            docker run --rm "${DOCKER_TTY_ARGS[@]}" \
                --runtime=nvidia \
                --gpus device=${GPU_DEVICE} \
                --network host \
                --name ${CONTAINER_NAME} \
                --user "${HOST_UID}:${HOST_GID}" \
                -v "$(pwd)/..:/workspace/video_to_data" \
                "${DATA_MOUNTS[@]}" \
                "${E2E_MOUNTS[@]}" \
                --group-add 1234 \
                -v "${HOME}/.ssh:/tmp/.ssh:ro" \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v "${CONTAINER_PASSWD_DIR}/passwd:/etc/passwd:ro" \
                -v "${CONTAINER_PASSWD_DIR}/group:/etc/group:ro" \
                -v "${KIT_DATA_DIR}:/isaac-sim/kit/data" \
                -v "${KIT_CACHE_DIR}:/isaac-sim/kit/cache" \
                -v "${KIT_LOGS_DIR}:/isaac-sim/kit/logs" \
                -e HOME=/tmp \
                -e "USER=${HOST_USERNAME}" \
                -e DISPLAY=${DISPLAY} \
                ${SSH_AGENT_MOUNT} \
                ${SSH_AGENT_ENV} \
                ${WANDB_API_KEY_ENV} \
                -e "ACCEPT_EULA=Y" \
                -d \
                --entrypoint /bin/bash \
                ${IMAGE_NAME} \
                "${CONTAINER_COMMAND[@]}"
        fi

        if [ "$E2E_MODE" = "1" ]; then
            docker exec --workdir "$E2E_WORKDIR" "$CONTAINER_NAME" "${E2E_COMMAND[@]}"
        else
            docker exec -it ${CONTAINER_NAME} /bin/bash
        fi
        ;;

    shell)
        echo "Entering shell of container: ${CONTAINER_NAME}"

        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Error: Container ${CONTAINER_NAME} is not running."
            echo "Use './run.sh start${ARCH_SUFFIX}' to start the container first."
            exit 1
        fi

        docker exec -it ${CONTAINER_NAME} /bin/bash
        ;;

    exec)
        shift 3 2>/dev/null || true
        [ "$1" = "--" ] && shift
        if [ $# -eq 0 ]; then
            echo "Usage: $0 exec${ARCH_SUFFIX} [version] [gpu] -- <command>"
            exit 1
        fi

        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Error: Container ${CONTAINER_NAME} is not running."
            echo "Use './run.sh start${ARCH_SUFFIX}' to start the container first."
            exit 1
        fi

        docker exec -it ${CONTAINER_NAME} "$@"
        ;;

    stop)
        echo "Stopping container: ${CONTAINER_NAME}"

        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Container ${CONTAINER_NAME} is not running."
            exit 0
        fi

        docker stop ${CONTAINER_NAME}
        echo "Container stopped successfully."
        ;;

    *)
        echo "Usage: $0 {build|push|pull|start|shell|exec|stop}[-aarch64] [version] [gpu]"
        echo "       $0 e2e-run [version] [gpu] --run-data PATH [--mano-dir PATH] [--checkpoint FILE] [--workdir PATH] [--recreate-on-mount-change] -- <command>"
        echo ""
        echo "  build [version]               - Build x86_64 image (default: latest)"
        echo "  build-aarch64 [version]        - Build aarch64 image (default: latest-aarch64)"
        echo "  push [version]                - Push x86_64 image to NGC"
        echo "  push-aarch64 [version]         - Push aarch64 image to NGC"
        echo "  pull [version]                - Pull x86_64 image from NGC"
        echo "  pull-aarch64 [version]         - Pull aarch64 image from NGC"
        echo "  start [version] [gpu]         - Start x86_64 container (default gpu: 0)"
        echo "  start-aarch64 [version] [gpu]  - Start aarch64 container"
        echo "  shell [version] [gpu]         - Enter shell of a running container"
        echo "  shell-aarch64 [version] [gpu]  - Enter shell of a running aarch64 container"
        echo "  exec [version] [gpu] -- <cmd> - Run a command in a running container"
        echo "  e2e-run [version] [gpu] ...   - Internal noninteractive start/reuse + exec"
        echo "  stop [version] [gpu]          - Stop the running container"
        echo "  stop-aarch64 [version] [gpu]   - Stop the running aarch64 container"
        exit 1
        ;;
esac
