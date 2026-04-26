"""GPU inventory helpers via the Docker/NVIDIA container runtime.

We query nvidia-smi inside a short-lived container so the orchestrator
itself doesn't need any NVIDIA tooling installed.
"""

import logging

import docker

log = logging.getLogger(__name__)

# nvidia-smi image — tiny, official, always available where NVIDIA runtime is
_NVIDIA_SMI_IMAGE = "nvidia/cuda:12.3.1-base-ubuntu22.04"


def _docker_client() -> docker.DockerClient:
    return docker.from_env()


def get_free_gpu_memory() -> dict[int, float]:
    """Return {gpu_index: free_memory_gb} for every visible GPU.

    Runs nvidia-smi in a throwaway container using the NVIDIA runtime.
    Returns an empty dict if the NVIDIA runtime is unavailable.
    """
    try:
        client = _docker_client()
        output = client.containers.run(
            _NVIDIA_SMI_IMAGE,
            command=(
                "nvidia-smi --query-gpu=index,memory.free "
                "--format=csv,noheader,nounits"
            ),
            remove=True,
            runtime="nvidia",
            stdout=True,
            stderr=False,
        )
        gpus: dict[int, float] = {}
        for line in output.decode().strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            idx, free_mb = parts
            gpus[int(idx)] = float(free_mb) / 1024.0
        return gpus

    except docker.errors.DockerException as e:
        log.warning("Could not query GPU memory via Docker: %s", e)
        return {}


def pick_gpu(required_gb: float, in_use: set[int]) -> int | None:
    """Return the GPU index with the most free VRAM >= required_gb.

    Excludes GPUs already assigned to a running model.
    Falls back to GPU 0 if GPU info is unavailable.
    """
    free = get_free_gpu_memory()
    if not free:
        log.warning("No GPU info available — defaulting to GPU 0")
        return 0

    candidates = {
        idx: gb
        for idx, gb in free.items()
        if idx not in in_use and gb >= required_gb
    }
    if not candidates:
        return None

    return max(candidates, key=lambda i: candidates[i])
