a# vLLM + QTIP Development Environment Setup (via Docker)

This document outlines the steps for setting up a vLLM development environment with QTIP integration using Docker. The target platform is WSL2 on Windows, with CUDA-enabled GPU support.

## Environment Overview

| Component | Details |
|-----------|---------|
| Platform | Windows 11 with WSL2 (Debian) |
| GPU | NVIDIA RTX (CUDA 12.8) |
| Container Runtime | Docker Desktop with WSL2 integration |
| Build Target | vllm-dev image with QTIP Python fallback code |

## Preparation

### Enable Docker WSL2 Integration

Ensure Docker Desktop is installed and WSL2 integration is enabled for the desired distro (e.g., Ubuntu 20.04).

## Dockerfile Changes I Made

### Added libnuma-dev to fix fastsafetensors build

```dockerfile
FROM base as dev
RUN apt-get update && apt-get install -y libnuma-dev
```


## Docker Image Build

### 1. Build the Dev Image

```bash
DOCKER_BUILDKIT=1 docker build . \
  -f docker/Dockerfile \
  --target dev \
  -t vllm-dev
```

This command builds the dev stage from `vllm/docker/Dockerfile`, creating an image tagged `vllm-dev`.

### Explanation

```bash
DOCKER_BUILDKIT=1 docker build . \
  --file docker/Dockerfile \
  --target dev \
  --tag vllm-dev
```

- `DOCKER_BUILDKIT=1`: Enables the BuildKit backend for improved performance and caching.
- `--file`: Specifies the path to the Dockerfile.
- `--target dev`: Builds only up to the dev stage.
- `--tag vllm-dev`: Names the output image vllm-dev.

> **Note:** The first build may take 20+ minutes due to dependency compilation.

## Running the Container

After the image is built:

```bash
docker run -it --gpus all --rm --ipc=host \
  -v $(pwd):/workspace \
  vllm-dev \
  /bin/bash
```

### Explanation

- `-it`: Launches an interactive terminal.
- `--gpus all`: Enables GPU access inside the container.
- `--ipc=host`: Shares host memory (important for PyTorch multiprocessing).
- `--rm`: Automatically removes the container upon exit.
- `-v $(pwd):/workspace`: Mounts the current host directory to /workspace inside the container.
- `/bin/bash`: Starts a bash shell.


### cuda part, not done yet, needs better computer
## Install vllm in developer mode
```

export SETUPTOOLS_SCM_PRETEND_VERSION=0.1

pip install -e .[dev] 
```

## Summary

This setup allows you to:

- Build vLLM with CUDA 12.8+ toolchain
- Develop and test QTIP inference modules in isolation
- Avoid polluting the host Python environment

All development is done inside `/workspace`, and changes are instantly visible on the Windows side via volume mounting.