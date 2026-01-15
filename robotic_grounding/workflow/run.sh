#!/bin/bash

# Script to manage robotic-grounding Docker container
# Usage:
#   ./run.sh build [version]         - Build the Docker image (default: latest)
#   ./run.sh push [version]          - Push the Docker image to NVIDIA registry (default: latest)
#   ./run.sh pull [version]          - Pull the Docker image from NVIDIA registry (default: latest)
#   ./run.sh start [version] [gpu]   - Run the container and enter the shell (default version: latest, gpu: 0)
#   ./run.sh shell [version] [gpu]   - Enter the shell of a running container with specific version and GPU
#   ./run.sh stop [version] [gpu]    - Stop the running container

set -e

VERSION=${2:-latest}
GPU_DEVICE=${3:-0}
IMAGE_NAME="robotic-grounding:${VERSION}"
CONTAINER_NAME="robotic-grounding-${VERSION}-gpu${GPU_DEVICE}"
NGC_LOCATION="nvcr.io/nvstaging/isaac-amr"

case "$1" in
    build)
        echo "Building Docker image: ${IMAGE_NAME}"
        cd "$(dirname "$0")/.."
        docker build -t ${IMAGE_NAME} -f workflow/Dockerfile .
        echo "Build complete!"
        ;;

    push)
        echo "Pushing Docker image to NVIDIA registry (version: ${VERSION})..."
        echo "Tagging ${IMAGE_NAME} for registry..."
        docker tag ${IMAGE_NAME} ${NGC_LOCATION}/${IMAGE_NAME}
        docker push ${NGC_LOCATION}/${IMAGE_NAME}
        echo "Push complete!"
        ;;

    pull)
        echo "Pulling Docker image from NVIDIA registry (version: ${VERSION})..."
        docker pull ${NGC_LOCATION}/${IMAGE_NAME}
        echo "Tagging as ${IMAGE_NAME}..."
        docker tag ${NGC_LOCATION}/${IMAGE_NAME} ${IMAGE_NAME}
        echo "Pull complete!"
        ;;

    start)
        echo "Starting container: ${CONTAINER_NAME}"

        # Check if container is running
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Container is already running. Entering shell..."
        else
            echo "Creating and starting new container..."
            cd "$(dirname "$0")/.."
            # Create docker-specific pre-commit cache directory if it doesn't exist
            mkdir -p ~/.cache/pre-commit-docker

            # Allow X11 forwarding from Docker containers
            xhost +local:docker > /dev/null 2>&1 || true

            docker run --rm -it \
                --runtime=nvidia \
                --gpus device=${GPU_DEVICE} \
                --network host \
                --name ${CONTAINER_NAME} \
                -v $(pwd):/workspace/robotic_grounding \
                -v ~/.ssh:/root/.ssh:ro \
                -v ~/.gitconfig:/root/.gitconfig \
                -v ~/.cache/pre-commit-docker:/root/.cache/pre-commit \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -e DISPLAY=${DISPLAY} \
                -e "ACCEPT_EULA=Y" \
                -d \
                ${IMAGE_NAME} \
                tail -f /dev/null
        fi

        # Enter the shell
        docker exec -it ${CONTAINER_NAME} /bin/bash
        ;;

    shell)
        echo "Entering shell of container: ${CONTAINER_NAME}"

        # Check if container is running
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Error: Container ${CONTAINER_NAME} is not running."
            echo "Use './run.sh start' to start the container first."
            exit 1
        fi

        docker exec -it ${CONTAINER_NAME} /bin/bash
        ;;

    stop)
        echo "Stopping container: ${CONTAINER_NAME}"

        # Check if container is running
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "Container ${CONTAINER_NAME} is not running."
            exit 0
        fi

        docker stop ${CONTAINER_NAME}
        echo "Container stopped successfully."
        ;;

    *)
        echo "Usage: $0 {build|pull|push|start|shell|stop} [version] [gpu]"
        echo ""
        echo "  build [version]         - Build the Docker image (default: latest)"
        echo "  pull [version]          - Pull the Docker image from NVIDIA registry (default: latest)"
        echo "  push [version]          - Push the Docker image to NVIDIA registry (default: latest)"
        echo "  start [version] [gpu]   - Run the container and enter the shell (default version: latest, gpu: 0)"
        echo "  shell                   - Enter the shell of a running container"
        echo "  stop                    - Stop the running container"
        exit 1
        ;;
esac
