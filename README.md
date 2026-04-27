# pdxhackerspace-llm-orchestrator

On-demand vLLM model lifecycle manager with LiteLLM routing for local GPU clusters.

Models start automatically when first requested, each in its own Docker container, and shut down after a configurable idle period. LiteLLM sits in front as the OpenAI-compatible API gateway, handling auth, rate limiting, and retries during model cold starts.

---

## Architecture

```
Client (OpenAI SDK / curl)
  │
  ▼
LiteLLM  :4000              ← API gateway: auth, rate limiting, spend tracking, retries
  │  HTTP (OpenAI-compatible)   (runs in Docker on llm-net)
  ▼
Orchestrator  :8000          ← model lifecycle manager + proxy
  │  Docker SDK               (runs in Docker on llm-net, has Docker socket access)
  ▼
vLLM containers              ← one per loaded model, spun up/down on demand
  vllm-llama-3-8b            (each joins llm-net, GPU assigned via NVIDIA runtime)
  vllm-mistral-7b
  ...
```

All services run in Docker. The orchestrator manages vLLM containers as siblings via the Docker socket — no host binaries required.

**The orchestrator and all vLLM containers must run on the GPU machine** (or machines). LiteLLM can run anywhere with network access to the orchestrator.

---

## Requirements

### GPU machine
- Linux (Ubuntu 22.04+ recommended)
- NVIDIA GPU(s) with CUDA 12+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and configured
- Docker Engine + Docker Compose v2
- Docker configured to use the NVIDIA runtime:
  ```bash
  # Verify the NVIDIA runtime is available
  docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
  ```

### LiteLLM machine (can be the same machine)
- Docker Engine + Docker Compose v2
- Network access to the GPU machine on the orchestrator port

### Docker image platforms

The published orchestrator image is built as a multi-platform Linux image for `linux/amd64` and `linux/arm64`. That covers x86_64 Linux hosts, Docker Desktop on Intel and Apple Silicon Macs, and ARM64 NVIDIA systems such as DGX Spark and Jetson Thor.

Macs can run the orchestrator and LiteLLM control-plane containers for development, but GPU-backed vLLM model containers still require a Linux host with NVIDIA Container Toolkit. For DGX Spark and Jetson Thor, keep `VLLM_IMAGE` configurable and choose a vLLM image tag that publishes `linux/arm64` support for the target hardware. You can verify an upstream image manifest with:

```bash
docker buildx imagetools inspect vllm/vllm-openai:latest
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/romkey/pdxhackerspace-llm-orchestrator.git
cd pdxhackerspace-llm-orchestrator

cp .env.example .env
$EDITOR .env
```

At minimum, set:
- `MODELS` — JSON array of models to serve (see below)
- `HUGGINGFACE_TOKEN` — required for gated models (Llama, Mistral, etc.)
- `HF_CACHE_DIR` — absolute path on the host for the HuggingFace model cache
- `ORCHESTRATOR_URL` — only needed if LiteLLM runs on a different machine

### 2. Start the stack

On the GPU machine (runs both orchestrator and LiteLLM):
```bash
docker compose up -d
docker compose logs -f
```

Or separately — orchestrator on the GPU machine, LiteLLM elsewhere:
```bash
# On the GPU machine
docker compose up -d orchestrator

# On the LiteLLM machine (set ORCHESTRATOR_URL in .env first)
docker compose up -d litellm
```

### 3. Test

```bash
# List available models
curl http://localhost:4000/v1/models

# Chat completion — triggers container + model load on first call
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The first request to a model takes 60–180 seconds while the vLLM container starts and loads the model weights. LiteLLM retries automatically on 503 responses during this window. Subsequent requests are fast.

---

## Configuration

All configuration lives in `.env`. **Never commit `.env`** — it is gitignored.

### Model definitions (`MODELS`)

```bash
MODELS='[
  {"name": "llama-3-8b",  "hf_name": "meta-llama/Llama-3.1-8B-Instruct",  "vram_gb": 18},
  {"name": "mistral-7b",  "hf_name": "mistralai/Mistral-7B-Instruct-v0.3", "vram_gb": 16},
  {"name": "qwen-14b",    "hf_name": "Qwen/Qwen2.5-14B-Instruct",          "vram_gb": 30}
]'
```

| Field | Description |
|-------|-------------|
| `name` | Name clients use in the `model` field of API requests |
| `hf_name` | HuggingFace model ID |
| `vram_gb` | VRAM required; used to select which GPU to assign |

### Key settings

| Variable | Default | Description |
|----------|---------|-------------|
| `IDLE_TIMEOUT_SECONDS` | `600` | Seconds idle before a model container is stopped |
| `MAX_CONCURRENT_MODELS` | `2` | Max models loaded at once (LRU eviction when exceeded) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of GPU VRAM vLLM may use per model |
| `ORCHESTRATOR_PORT` | `8000` | Port the orchestrator API listens on |
| `STARTUP_TIMEOUT_SECONDS` | `180` | Max seconds to wait for a vLLM container to become healthy |
| `HF_CACHE_MIN_FREE_GB` | `10` | Minimum free disk space required in `HF_CACHE_DIR` before starting a model |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | Docker image used for vLLM containers |
| `VLLM_EXTRA_ARGS` | _(none)_ | Optional extra arguments appended to `vllm serve` |
| `HF_CACHE_DIR` | _(required)_ | Absolute host path to HuggingFace model cache |
| `ORCHESTRATOR_URL` | `http://orchestrator:8000` | URL LiteLLM uses to reach the orchestrator |
| `LITELLM_PORT` | `4000` | External port LiteLLM listens on |
| `LITELLM_MASTER_KEY` | _(none)_ | Optional API key clients must send to LiteLLM |

See `.env.example` for the full list with comments.

### LiteLLM routing (`litellm/config.yaml`)

`litellm/config.yaml` is safe to commit — it contains no secrets. The orchestrator URL is injected at runtime via `${ORCHESTRATOR_URL}`.

To add a model, add it to both `MODELS` in `.env` and `model_list` in `litellm/config.yaml`, then restart both services.

---

## Docker Network

All containers communicate over a Docker bridge network named `llm-net`. The orchestrator creates this network automatically at startup if it doesn't exist. vLLM containers are addressed by their container name (e.g. `vllm-llama-3-8b`) within the network — no port mapping to the host is required for internal traffic.

## Security Note: Docker Socket

The orchestrator mounts `/var/run/docker.sock` to manage vLLM containers. This gives the orchestrator container root-equivalent access to the Docker daemon on the host. This is a standard pattern for container orchestration but you should be aware of the implication: a compromised orchestrator container could affect other containers or the host.

Mitigations if this is a concern:
- Run the orchestrator on a dedicated machine used only for LLM serving
- Use Docker socket proxy (e.g. [Tecnativa/docker-socket-proxy](https://github.com/Tecnativa/docker-socket-proxy)) to limit which Docker API calls the orchestrator can make

---

## Multi-GPU Machines

The orchestrator queries GPU free memory via a short-lived `nvidia/cuda` container and automatically assigns each model to the GPU with the most available VRAM, skipping GPUs already in use. No manual GPU pinning is needed.

Check current GPU and model state:

```bash
curl http://localhost:8000/status | jq
```

---

## Multiple GPU Machines

Run one orchestrator per GPU machine. Add each as a separate backend in `litellm/config.yaml`:

```yaml
model_list:
  # llama on machine 1
  - model_name: llama-3-8b
    litellm_params:
      model: openai/llama-3-8b
      api_base: "http://gpu-machine-1:8000/v1"
      api_key: "none"

  # same model on machine 2 — LiteLLM load-balances automatically
  - model_name: llama-3-8b
    litellm_params:
      model: openai/llama-3-8b
      api_base: "http://gpu-machine-2:8000/v1"
      api_key: "none"
```

---

## Orchestrator API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check — returns `{"status": "ok"}` |
| `GET /status` | Running models, loading models, GPU free memory |
| `GET /v1/models` | Configured models in OpenAI list format |
| `POST /v1/chat/completions` | Proxied to vLLM (container started on demand) |
| `POST /v1/completions` | Proxied to vLLM |

A `503 Retry-After: 10` response means the model container is currently starting. LiteLLM handles retries automatically.

---

## Monitoring

```bash
# Orchestrator status (running models, loading, GPU memory)
curl http://localhost:8000/status | jq

# Live logs
docker compose logs -f orchestrator
docker compose logs -f litellm

# Individual vLLM container logs
docker logs vllm-llama-3-8b -f

# All vLLM containers
docker ps --filter name=vllm-

# GPU utilization
watch -n2 nvidia-smi
```

---

## Troubleshooting

**"Could not connect to Docker daemon"**
- Confirm `/var/run/docker.sock` exists and the Docker service is running
- Check the socket mount in `docker-compose.yml`

**Container starts but model never becomes healthy**
- Check the vLLM container logs: `docker logs vllm-<model-name>`
- Confirm `HUGGINGFACE_TOKEN` is set and has access to the model
- Increase `STARTUP_TIMEOUT_SECONDS` for large models or slow storage
- Confirm `HF_CACHE_DIR` is an absolute path (tilde `~` does not expand in Docker mounts)
- Confirm `HF_CACHE_DIR` has enough free disk space for the model weights and temporary download files

**vLLM exits with CUDA out of memory during startup**
- Lower `GPU_MEMORY_UTILIZATION`
- Reduce context length with `VLLM_EXTRA_ARGS="--max-model-len 8192"`
- Try `VLLM_EXTRA_ARGS="--max-model-len 8192 --enforce-eager"` to avoid CUDA graph capture overhead during testing
- Use a smaller or quantized model if weights plus KV cache do not fit on the GPU

**"No GPU with N GB free"**
- Another model container is using the GPU — wait for it to idle out or lower `IDLE_TIMEOUT_SECONDS`
- Increase `MAX_CONCURRENT_MODELS` if you have multiple GPUs
- Lower `vram_gb` in `MODELS` if your estimate is too conservative

**LiteLLM times out before the model is ready**
- Cold starts always take 60–180s; LiteLLM retries up to 8 times with 10s back-off
- If still failing, increase `timeout` in `litellm/config.yaml`

**NVIDIA runtime not found**
- Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Verify: `docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi`

---

## Development (no Docker)

```bash
cd orchestrator
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in .env
uvicorn main:app --reload --port 8000
```

The orchestrator will still use the Docker socket to manage vLLM containers even when running outside Docker, as long as Docker is available on the development machine.

---

## License

MIT
