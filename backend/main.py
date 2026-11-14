"""
VoiceGuard AI - FastAPI entry point (Phase 1: skeleton only).
No AASIST model is loaded yet and NO predictions are made.
Run (from the backend folder, venv active):   python main.py
"""
from __future__ import annotations

import platform
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import load_calibration, settings

START_TIME = time.time()

app = FastAPI(title="VoiceGuard AI", version="0.1.0-phase1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def gpu_info() -> dict:
    """Describe PyTorch / GPU state without ever crashing the server."""
    try:
        import torch
    except ImportError:
        return {"torch_installed": False}

    info: dict = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        try:
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            info.update(
                device_name=props.name,
                compute_capability=f"{props.major}.{props.minor}",
                vram_total_mb=round(total_b / 2**20),
                vram_free_mb=round(free_b / 2**20),
            )
        except Exception as exc:
            info["gpu_query_error"] = f"{type(exc).__name__}: {exc}"
    return info


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "phase": 1,
        "model_loaded": False,
        "model_note": "AASIST is not loaded yet (added in Phase 2). "
                      "No detection is available.",
        "checkpoint_present": settings.checkpoint_path.is_file(),
        "calibrated": load_calibration() is not None,
        "device_preference": settings.device_preference,
        "gpu": gpu_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uptime_s": round(time.time() - START_TIME, 1),
    }


# Must stay after all API routes.
if (settings.frontend_dir / "index.html").is_file():
    app.mount(
        "/",
        StaticFiles(directory=str(settings.frontend_dir), html=True),
        name="frontend",
    )


if __name__ == "__main__":
    import uvicorn

    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"WARNING: host is {settings.host!r}. This exposes the server to "
            "your network. Keep development builds on 127.0.0.1."
        )
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)