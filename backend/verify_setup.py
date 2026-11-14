"""
VoiceGuard AI - environment check (Phase 1).
Run from the backend folder with the venv active:  python verify_setup.py
Prints real PASS / WARN / FAIL results from your machine.
"""
from __future__ import annotations

import hashlib
import importlib
import platform
import sys
import time
from pathlib import Path

from config import settings

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_statuses: list[str] = []


def report(status: str, name: str, detail: str = "") -> None:
    _statuses.append(status)
    print(f"[{status}] {name}" + (f"  -  {detail}" if detail else ""))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> None:
    section("Python")
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 10):
        report(PASS, "Python version", ver)
    elif (v.major, v.minor) > (3, 10):
        report(WARN, "Python version", f"{ver} (team standard was 3.10; 3.11 is OK)")
    else:
        report(FAIL, "Python version", f"{ver} - need 3.10 or newer")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    report(PASS if in_venv else WARN, "Virtual environment active", sys.prefix)
    print(f"      OS: {platform.platform()}")


PACKAGES = [
    (["numpy"], "numpy"),
    (["librosa"], "librosa"),
    (["soundfile"], "soundfile"),
    (["fastapi"], "fastapi"),
    (["uvicorn"], "uvicorn[standard]"),
    (["python_multipart", "multipart"], "python-multipart"),
    (["aiofiles"], "aiofiles"),
    (["huggingface_hub"], "huggingface_hub"),
]


def check_packages() -> None:
    section("Python packages")
    for modules, pip_name in PACKAGES:
        last_exc = None
        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
                report(PASS, f"import {mod_name}", str(getattr(mod, "__version__", "version unknown")))
                break
            except Exception as exc:
                last_exc = exc
        else:
            report(FAIL, f"import {modules[0]}", f"{type(last_exc).__name__}: {last_exc}  ->  pip install {pip_name}")


def check_torch() -> None:
    section("PyTorch / GPU")
    try:
        import torch
    except Exception as exc:
        report(FAIL, "import torch", f"{type(exc).__name__}: {exc}")
        return

    build_cuda = torch.version.cuda
    report(PASS, "import torch", f"{torch.__version__} (built for CUDA {build_cuda})")
    try:
        import torchaudio
        report(PASS, "import torchaudio", torchaudio.__version__)
    except Exception as exc:
        report(WARN, "import torchaudio", f"{type(exc).__name__}: {exc}")

    if build_cuda is None:
        report(WARN, "PyTorch build", "CPU-only build installed - the GPU will NOT be used.")
        print("      GPU status: NOT READY (CPU fallback only)")
        return
    if not torch.cuda.is_available():
        report(WARN, "torch.cuda.is_available()", "False although this is a CUDA build - "
                                                  "usually an outdated NVIDIA driver.")
        print("      GPU status: NOT READY (CPU fallback only)")
        return

    report(PASS, "torch.cuda.is_available()", "True")
    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"      GPU name:           {props.name}")
    print(f"      Compute capability: {props.major}.{props.minor}")
    print(f"      VRAM total / free:  {total_b / 2**20:.0f} MB / {free_b / 2**20:.0f} MB")
    if "3050" in props.name:
        report(PASS, "RTX 3050 detected", props.name)
    else:
        report(WARN, "RTX 3050 detected", f"GPU 0 is '{props.name}'")

    try:
        x = torch.randn(1024, 1024, device="cuda")
        _ = x @ x
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = x @ x
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / 10
        report(PASS, "GPU compute test", f"1024x1024 matmul executed on GPU, ~{ms:.2f} ms each")
        del x, _
        torch.cuda.empty_cache()
        print("      GPU status: READY")
    except Exception as exc:
        report(FAIL, "GPU compute test", f"{type(exc).__name__}: {exc}")
        print("      GPU status: NOT READY")


def check_audio_formats() -> None:
    section("Audio decoding (libsndfile via soundfile)")
    try:
        import soundfile as sf
    except Exception:
        report(FAIL, "soundfile", "not importable (see package check above)")
        return
    print(f"      libsndfile: {sf.__libsndfile_version__}")
    fmts = sf.available_formats()
    for name in ("WAV", "FLAC", "OGG", "MP3"):
        if name in fmts:
            report(PASS, f"decoder present: {name}", fmts[name])
        else:
            report(WARN, f"decoder present: {name}", "not available in this libsndfile build")
    print("      NOTE: 'decoder present' does NOT mean VoiceGuard handles the format")
    print("            correctly. Each format is verified by tests later.")


def check_files() -> None:
    section("Model files and project layout")
    ckpt = settings.checkpoint_path
    if ckpt.is_file():
        size_kb = ckpt.stat().st_size / 1024
        digest = sha256_of(ckpt)
        status = PASS if 100 < size_kb < 5000 else WARN
        report(status, "HF checkpoint AASIST.pth", f"{size_kb:.0f} KB (expected roughly 1,300 KB)")
        print(f"      SHA-256: {digest}")
    else:
        digest = None
        report(WARN, "HF checkpoint AASIST.pth", f"missing at {ckpt}  ->  python scripts\\fetch_model_files.py")

    upstream_model = settings.upstream_dir / "models" / "AASIST.py"
    upstream_ckpt = settings.upstream_dir / "models" / "weights" / "AASIST.pth"
    report(PASS if upstream_model.is_file() else WARN,
           "Upstream code models/AASIST.py",
           str(upstream_model) if upstream_model.is_file()
           else "missing -> git clone https://github.com/clovaai/aasist.git third_party\\aasist")
    if upstream_ckpt.is_file():
        up_digest = sha256_of(upstream_ckpt)
        print(f"      Upstream checkpoint SHA-256: {up_digest}")
        if digest is None:
            report(WARN, "Checkpoint identical to upstream?", "HF checkpoint not downloaded yet")
        elif digest == up_digest:
            report(PASS, "Checkpoint identical to upstream?", "YES - same bytes as clovaai/aasist")
        else:
            report(WARN, "Checkpoint identical to upstream?", "NO - hashes differ; we will inspect in Phase 2")
    else:
        print("      (no upstream checkpoint at models/weights/AASIST.pth - skipped hash comparison)")

    index = settings.frontend_dir / "index.html"
    report(PASS if index.is_file() else WARN, "frontend/index.html",
           "found" if index.is_file() else "missing - copy your prototype HTML here")


def main() -> int:
    print("VoiceGuard AI - environment check")
    print(f"Backend folder: {Path(__file__).resolve().parent}")
    check_python()
    check_packages()
    check_torch()
    check_audio_formats()
    check_files()

    section("Summary")
    print(f"PASS: {_statuses.count(PASS)}   WARN: {_statuses.count(WARN)}   FAIL: {_statuses.count(FAIL)}")
    return 1 if FAIL in _statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())