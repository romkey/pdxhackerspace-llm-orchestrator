"""Model lifecycle management: start, stop, evict, idle-watch.

vLLM instances run as sibling Docker containers on the llm-net network.
The orchestrator manages them via the Docker SDK.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import docker
import docker.errors
import httpx

from config import ModelConfig, OrchestratorConfig
from gpu import pick_gpu

log = logging.getLogger(__name__)

CONTAINER_PREFIX = "vllm-"
VLLM_INTERNAL_PORT = 8000   # port vLLM listens on inside its container
DOCKER_NETWORK = "llm-net"  # containers must share this network


@dataclass
class RunningModel:
    config: ModelConfig
    container_id: str
    container_name: str
    gpu: int
    started_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used = time.time()

    def idle_seconds(self) -> float:
        return time.time() - self.last_used

    def base_url(self) -> str:
        # Reach the vLLM container by name on the shared Docker network
        return f"http://{self.container_name}:{VLLM_INTERNAL_PORT}"


class ModelLoadingError(Exception):
    """Raised when the requested model is currently being loaded."""
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Model {model_name!r} is currently loading")


class ModelManager:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self._running: dict[str, RunningModel] = {}
        self._loading: set[str] = set()
        self._lock = asyncio.Lock()
        self._docker = docker.from_env()
        self._ensure_network()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure(self, model_name: str) -> RunningModel:
        """Return a running model, starting it if necessary.

        Raises:
            ValueError: model name not in config
            ModelLoadingError: model is already loading (client should retry)
            RuntimeError: model failed to start
        """
        async with self._lock:
            if model_name in self._running:
                self._running[model_name].touch()
                return self._running[model_name]

            model_cfg = self.cfg.model_by_name(model_name)
            if model_cfg is None:
                raise ValueError(f"Unknown model: {model_name!r}")

            if model_name in self._loading:
                raise ModelLoadingError(model_name)

            await self._evict_if_needed()
            self._loading.add(model_name)

        # Start outside the lock so other models aren't blocked during load
        try:
            entry = await self._start(model_cfg)
        finally:
            async with self._lock:
                self._loading.discard(model_name)

        async with self._lock:
            self._running[model_name] = entry

        return entry

    def is_loading(self, model_name: str) -> bool:
        return model_name in self._loading

    def status(self) -> dict:
        from gpu import get_free_gpu_memory
        return {
            "running": {
                name: {
                    "container": entry.container_name,
                    "gpu": entry.gpu,
                    "idle_seconds": round(entry.idle_seconds(), 1),
                    "uptime_seconds": round(time.time() - entry.started_at, 1),
                }
                for name, entry in self._running.items()
            },
            "loading": list(self._loading),
            "gpus": {str(k): round(v, 2) for k, v in get_free_gpu_memory().items()},
        }

    async def stop(self, model_name: str) -> bool:
        async with self._lock:
            return self._stop_unlocked(model_name)

    async def shutdown_all(self) -> None:
        async with self._lock:
            for name in list(self._running):
                self._stop_unlocked(name)

    # ------------------------------------------------------------------
    # Background idle watcher
    # ------------------------------------------------------------------

    async def idle_watcher(self) -> None:
        log.info(
            "Idle watcher started (timeout=%ds, interval=%ds)",
            self.cfg.idle_timeout,
            self.cfg.idle_check_interval,
        )
        while True:
            await asyncio.sleep(self.cfg.idle_check_interval)
            async with self._lock:
                idle = [
                    name
                    for name, entry in self._running.items()
                    if entry.idle_seconds() >= self.cfg.idle_timeout
                ]
            for name in idle:
                log.info("Stopping idle model %r", name)
                await self.stop(name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_network(self) -> None:
        """Create the shared Docker network if it doesn't already exist."""
        try:
            self._docker.networks.get(DOCKER_NETWORK)
            log.info("Docker network %r already exists", DOCKER_NETWORK)
        except docker.errors.NotFound:
            self._docker.networks.create(DOCKER_NETWORK, driver="bridge")
            log.info("Created Docker network %r", DOCKER_NETWORK)

    def _container_name(self, model_name: str) -> str:
        # Sanitize model name for use as a Docker container name
        safe = model_name.replace("/", "-").replace(":", "-").replace("_", "-")
        return f"{CONTAINER_PREFIX}{safe}"

    def _ensure_image(self, image: str) -> None:
        """Pull the vLLM image if it is not already present locally."""
        try:
            self._docker.images.get(image)
            return
        except docker.errors.ImageNotFound:
            log.info("Docker image %r not found locally; pulling it now", image)

        try:
            self._docker.images.pull(image)
            log.info("Pulled Docker image %r", image)
        except docker.errors.APIError as e:
            explanation = e.explanation or str(e)
            raise RuntimeError(
                f"Failed to pull Docker image {image!r}: {explanation}"
            ) from e
        except docker.errors.DockerException as e:
            raise RuntimeError(f"Failed to pull Docker image {image!r}: {e}") from e

    async def _evict_if_needed(self) -> None:
        """Evict the least-recently-used model if at capacity.

        Must be called while holding self._lock.
        """
        if len(self._running) < self.cfg.max_concurrent_models:
            return

        lru_name = min(self._running, key=lambda n: self._running[n].last_used)
        log.info(
            "At capacity (%d). Evicting LRU model %r",
            self.cfg.max_concurrent_models,
            lru_name,
        )
        self._stop_unlocked(lru_name)

    def _stop_unlocked(self, model_name: str) -> bool:
        entry = self._running.pop(model_name, None)
        if entry is None:
            return False
        log.info("Stopping container %r for model %r", entry.container_name, model_name)
        try:
            container = self._docker.containers.get(entry.container_id)
            container.stop(timeout=15)
            container.remove(force=True)
        except docker.errors.NotFound:
            pass  # already gone
        except docker.errors.DockerException as e:
            log.warning("Error stopping container %r: %s", entry.container_name, e)
        return True

    async def _start(self, model_cfg: ModelConfig) -> RunningModel:
        in_use = {e.gpu for e in self._running.values()}
        gpu = pick_gpu(model_cfg.vram_gb, in_use)

        if gpu is None:
            raise RuntimeError(
                f"No GPU with {model_cfg.vram_gb} GB free available "
                f"for model {model_cfg.name!r}"
            )

        container_name = self._container_name(model_cfg.name)

        # Remove any stale container with the same name
        try:
            old = self._docker.containers.get(container_name)
            log.warning("Removing stale container %r", container_name)
            old.remove(force=True)
        except docker.errors.NotFound:
            pass

        environment = {
            "HF_TOKEN": self.cfg.hf_token or "",
            "HUGGING_FACE_HUB_TOKEN": self.cfg.hf_token or "",
        }

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._ensure_image(self.cfg.vllm_image))

        log.info(
            "Starting vLLM container %r (image=%s) for model %r on GPU %d",
            container_name, self.cfg.vllm_image, model_cfg.name, gpu,
        )

        # Run in a thread — docker SDK is synchronous
        try:
            container = await loop.run_in_executor(
                None,
                lambda: self._docker.containers.run(
                    self.cfg.vllm_image,
                    command=[
                        model_cfg.hf_name,
                        "--port", str(VLLM_INTERNAL_PORT),
                        "--gpu-memory-utilization", str(self.cfg.gpu_memory_utilization),
                    ],
                    name=container_name,
                    detach=True,
                    network=DOCKER_NETWORK,
                    environment=environment,
                    device_requests=[
                        docker.types.DeviceRequest(
                            device_ids=[str(gpu)],
                            capabilities=[["gpu"]],
                        )
                    ],
                    volumes={
                        self.cfg.hf_cache_dir: {
                            "bind": "/root/.cache/huggingface",
                            "mode": "rw",
                        }
                    },
                    shm_size="2g",          # vLLM needs shared memory
                    restart_policy={"Name": "no"},
                ),
            )
        except docker.errors.ImageNotFound as e:
            raise RuntimeError(
                f"vLLM image {self.cfg.vllm_image!r} is not available to Docker. "
                "Pull it on the host or set VLLM_IMAGE to an available image."
            ) from e
        except docker.errors.APIError as e:
            explanation = e.explanation or str(e)
            raise RuntimeError(
                f"Failed to create vLLM container {container_name!r}: {explanation}"
            ) from e
        except docker.errors.DockerException as e:
            raise RuntimeError(
                f"Failed to start vLLM container {container_name!r}: {e}"
            ) from e

        log.info("Container %r started (id=%s)", container_name, container.short_id)

        entry = RunningModel(
            config=model_cfg,
            container_id=container.id,
            container_name=container_name,
            gpu=gpu,
        )

        await self._wait_for_health(entry)
        return entry

    async def _wait_for_health(self, entry: RunningModel) -> None:
        url = f"{entry.base_url()}/health"
        deadline = time.time() + self.cfg.startup_timeout

        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                # Check if the container exited unexpectedly
                try:
                    loop = asyncio.get_event_loop()
                    container = await loop.run_in_executor(
                        None,
                        lambda: self._docker.containers.get(entry.container_id),
                    )
                    if container.status not in ("running", "created"):
                        logs = container.logs(tail=50).decode(errors="replace")
                        raise RuntimeError(
                            f"Container {entry.container_name!r} exited unexpectedly "
                            f"(status={container.status}).\nLast logs:\n{logs}"
                        )
                except docker.errors.NotFound:
                    raise RuntimeError(
                        f"Container {entry.container_name!r} disappeared during startup"
                    )

                try:
                    r = await client.get(url, timeout=5)
                    if r.status_code == 200:
                        log.info(
                            "Model %r healthy (container %r)",
                            entry.config.name, entry.container_name,
                        )
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(3)

        raise RuntimeError(
            f"Model {entry.config.name!r} did not become healthy within "
            f"{self.cfg.startup_timeout}s"
        )
