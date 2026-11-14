"""
VoiceGuard AI - download the pretrained AASIST files from Hugging Face.
Run from the backend folder with the venv active:
    python scripts\fetch_model_files.py
Prints what the repo really contains, downloads the wanted files that exist,
and records the exact commit in aasist_model/SOURCE.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402

WANTED = ["AASIST.pth", "aasist.py", "_net.py", "LICENSE", "meta.yaml"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None, help="commit sha or branch (default: main)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install -r backend\\requirements.txt")
        return 1

    repo = settings.hf_repo_id
    api = HfApi()
    try:
        info = api.model_info(repo, revision=args.revision)
        commit = info.sha
        files = api.list_repo_files(repo, revision=commit)
    except Exception as exc:
        print(f"ERROR: could not reach Hugging Face ({type(exc).__name__}: {exc}).")
        return 1

    print(f"Repository : {repo}")
    print(f"Commit     : {commit}")
    print("Files in the repository right now:")
    for name in sorted(files):
        print(f"   - {name}")

    if "AASIST.pth" not in files:
        print("\nERROR: AASIST.pth is not in the repository. Send me this output.")
        return 1

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    print(f"\nDownloading into {settings.model_dir} ...")
    for name in WANTED:
        if name not in files:
            print(f"   (not in repo, skipped) {name}")
            continue
        local = Path(hf_hub_download(repo_id=repo, filename=name, revision=commit,
                                     local_dir=str(settings.model_dir)))
        digest = sha256_of(local)
        size = local.stat().st_size
        downloaded[name] = {"bytes": size, "sha256": digest}
        print(f"   ok  {name:<12} {size:>10,} bytes   sha256={digest}")

    ckpt = downloaded["AASIST.pth"]
    if ckpt["bytes"] < 100_000:
        print("\nERROR: AASIST.pth is suspiciously small (expected roughly 1.3 MB). Delete it and retry.")
        return 1

    source = {"repo": repo, "commit": commit, "repo_files": sorted(files), "downloaded": downloaded}
    (settings.model_dir / "SOURCE.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"\nWrote provenance record: {settings.model_dir / 'SOURCE.json'}")
    print("Done. Next: git clone the official repo, then run  python verify_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())