"""PDX Hackerspace LLM Orchestrator.

Proxies OpenAI-compatible requests to on-demand vLLM processes.
Models are started on first request and shut down after idling.
"""

import asyncio
import json
import logging
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import load_config
from lifecycle import ModelLoadingError, ModelManager

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

cfg = load_config()

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("orchestrator")

manager = ModelManager(cfg)
app = FastAPI(title="LLM Orchestrator")

_HOP_BY_HOP = {
    "host", "content-length", "transfer-encoding",
    "connection", "keep-alive", "te", "trailers", "upgrade",
}

# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    log.info(
        "Orchestrator starting. %d model(s) configured, max %d concurrent.",
        len(cfg.models),
        cfg.max_concurrent_models,
    )
    for m in cfg.models:
        log.info(
            "  - %-20s  %-55s  vram=%.0fGB",
            m.name, m.hf_name, m.vram_gb,
        )
    asyncio.create_task(manager.idle_watcher())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    log.info("Orchestrator shutting down — stopping all models")
    await manager.shutdown_all()


# ---------------------------------------------------------------------------
# Health / status endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    """Running models, models currently loading, and GPU free memory."""
    return manager.status()


@app.get("/v1/models")
async def list_models() -> dict:
    """All configured models in OpenAI list format."""
    return {
        "object": "list",
        "data": [
            {"id": m.name, "object": "model", "owned_by": "orchestrator"}
            for m in cfg.models
        ],
    }


# ---------------------------------------------------------------------------
# Proxy — all other /v1/* traffic forwarded to the appropriate vLLM process
# ---------------------------------------------------------------------------


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    # ---- Resolve model name ------------------------------------------------
    model_name: str | None = None
    body_bytes: bytes = b""

    if request.method in ("POST", "PUT"):
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes) if body_bytes else {}
            model_name = body_json.get("model")
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
    else:
        model_name = request.query_params.get("model")

    # Fall back to explicit header
    if not model_name:
        model_name = request.headers.get("x-model")

    if not model_name:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "model name required — set the 'model' field in the request body "
                    "or pass an X-Model header"
                )
            },
        )

    # ---- Ensure model is running -------------------------------------------
    try:
        entry = await manager.ensure(model_name)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except ModelLoadingError:
        # Tell LiteLLM to retry shortly
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "10"},
            content={"error": f"Model {model_name!r} is loading — please retry shortly"},
        )
    except RuntimeError as e:
        log.exception("Failed to start model %r", model_name)
        return JSONResponse(status_code=503, content={"error": str(e)})

    # ---- Stream proxy to vLLM ---------------------------------------------
    target_url = f"{entry.base_url()}/v1/{path}"
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                content=body_bytes,
                params=dict(request.query_params),
            ) as resp:
                entry.touch()
                status_code = resp.status_code
                content_type = resp.headers.get("content-type", "application/json")

                # Buffer small non-streaming responses; stream everything else
                if "text/event-stream" in content_type:
                    async def stream_sse():
                        async for chunk in resp.aiter_bytes(chunk_size=512):
                            yield chunk

                    return StreamingResponse(
                        content=stream_sse(),
                        status_code=status_code,
                        media_type=content_type,
                    )
                else:
                    body = await resp.aread()
                    return JSONResponse(
                        content=json.loads(body) if body else {},
                        status_code=status_code,
                    )

    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={"error": f"Could not connect to backend for model {model_name!r}"},
        )
    except Exception as e:
        log.exception("Proxy error for model %r at %s", model_name, target_url)
        return JSONResponse(status_code=500, content={"error": str(e)})
