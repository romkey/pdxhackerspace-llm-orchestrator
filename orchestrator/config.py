"""Load and validate all configuration from environment / .env file."""

import json
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str        # name clients use in requests
    hf_name: str     # HuggingFace model ID
    vram_gb: float   # VRAM required for GPU selection


@dataclass
class OrchestratorConfig:
    models: list[ModelConfig]
    hf_token: str | None
    hf_cache_dir: str
    idle_timeout: int
    max_concurrent_models: int
    gpu_memory_utilization: float
    orchestrator_port: int
    startup_timeout: int
    idle_check_interval: int
    log_level: str
    vllm_image: str

    def model_by_name(self, name: str) -> ModelConfig | None:
        return next((m for m in self.models if m.name == name), None)


def load_config() -> OrchestratorConfig:
    models_raw = os.environ.get("MODELS")
    if not models_raw:
        raise RuntimeError(
            "MODELS environment variable is not set. "
            "Copy .env.example to .env and fill it in."
        )

    try:
        models_data = json.loads(models_raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse MODELS JSON: {e}") from e

    models: list[ModelConfig] = []
    for entry in models_data:
        try:
            models.append(
                ModelConfig(
                    name=entry["name"],
                    hf_name=entry["hf_name"],
                    vram_gb=float(entry["vram_gb"]),
                )
            )
        except KeyError as e:
            raise RuntimeError(f"Model entry missing required key {e}: {entry}") from e

    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate model names found in MODELS config")

    return OrchestratorConfig(
        models=models,
        hf_token=os.environ.get("HUGGINGFACE_TOKEN"),
        hf_cache_dir=os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface")),
        idle_timeout=int(os.environ.get("IDLE_TIMEOUT_SECONDS", "600")),
        max_concurrent_models=int(os.environ.get("MAX_CONCURRENT_MODELS", "2")),
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")),
        orchestrator_port=int(os.environ.get("ORCHESTRATOR_PORT", "8000")),
        startup_timeout=int(os.environ.get("STARTUP_TIMEOUT_SECONDS", "180")),
        idle_check_interval=int(os.environ.get("IDLE_CHECK_INTERVAL_SECONDS", "60")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        vllm_image=os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:latest"),
    )
