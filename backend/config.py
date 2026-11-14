"""
VoiceGuard AI - central configuration (Phase 1).
Override any value with an environment variable starting with VOICEGUARD_.
No detection thresholds are defined here; they come from calibration (Phase 8).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent   # ...\VoiceGuard\backend
PROJECT_DIR = BASE_DIR.parent                # ...\VoiceGuard


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VOICEGUARD_{name}", default)


def _env_tuple(name: str, default: tuple) -> tuple:
    raw = os.environ.get(f"VOICEGUARD_{name}")
    if raw is None:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    cors_origins: tuple
    frontend_dir: Path
    model_dir: Path
    upstream_dir: Path
    checkpoint_path: Path
    calibration_path: Path
    hf_repo_id: str
    sample_rate: int
    window_samples: int
    live_hop_seconds: float
    max_upload_mb: int
    max_audio_seconds: int
    allowed_extensions: tuple
    max_concurrent_inference: int
    device_preference: str

    @property
    def window_seconds(self) -> float:
        """64600 samples / 16000 Hz = 4.0375 s."""
        return self.window_samples / self.sample_rate


def load_settings() -> Settings:
    model_dir = BASE_DIR / "aasist_model"
    port = int(_env("PORT", "8000"))
    return Settings(
        host=_env("HOST", "127.0.0.1"),
        port=port,
        cors_origins=_env_tuple(
            "CORS_ORIGINS",
            (f"http://127.0.0.1:{port}", f"http://localhost:{port}"),
        ),
        frontend_dir=PROJECT_DIR / "frontend",
        model_dir=model_dir,
        upstream_dir=PROJECT_DIR / "third_party" / "aasist",
        checkpoint_path=model_dir / "AASIST.pth",
        calibration_path=BASE_DIR / "calibration.json",
        hf_repo_id="SpeechAntiSpoofingBenchmarks/AASIST",
        sample_rate=16000,
        window_samples=64600,
        live_hop_seconds=float(_env("LIVE_HOP_SECONDS", "2.5")),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", "25")),
        max_audio_seconds=int(_env("MAX_AUDIO_SECONDS", "600")),
        allowed_extensions=(".wav", ".flac", ".mp3", ".ogg"),
        max_concurrent_inference=int(_env("MAX_CONCURRENT_INFERENCE", "1")),
        device_preference=_env("DEVICE", "auto").lower(),
    )


settings = load_settings()


def load_calibration(path: Optional[Path] = None) -> Optional[dict]:
    """Return calibrated thresholds, or None if calibration is not done."""
    path = path or settings.calibration_path
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None