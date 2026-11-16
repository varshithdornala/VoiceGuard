# VoiceGuard

Deepfake voice detection prototype built by **Team Exito** (Cybersecurity).
Upload an audio clip and VoiceGuard predicts whether the voice is real (bonafide) or synthetic (spoof), using the pretrained AASIST anti-spoofing model.

Repository: https://github.com/varshithdornala/VoiceGuard

## Team

| Member | Role | GitHub |
|---|---|---|
| Chamala Nidhish Kumar (Team Leader) | Frontend and UX | (code pushed via teammate's account) |
| Chinta Eswara Naga Srinivas | AI Model and Detection | [srinivas2008chinta-ai](https://github.com/srinivas2008chinta-ai) |
| Mallu Rama Sumanth Reddy | Backend and API | [sumanthrdy007r-hue](https://github.com/sumanthrdy007r-hue) |
| Dornala Varshith | Setup, Testing and Deployment | [varshithdornala](https://github.com/varshithdornala) |

## How it works

Browser (HTML page) -> FastAPI backend -> audio decoding and preprocessing -> AASIST model (PyTorch, GPU) -> score and verdict returned to the page.

## Technology

- **Model:** AASIST ([clovaai/aasist](https://github.com/clovaai/aasist)), pretrained checkpoint
- **ML framework:** PyTorch 2.7.0 + torchaudio (CUDA 12.6 build)
- **Backend:** FastAPI + uvicorn
- **Audio:** soundfile / libsndfile (WAV, FLAC, OGG, MP3), librosa, numpy, scipy
- **Frontend:** single HTML page (`frontend/index.html`) served by the backend
- **Python:** 3.11

## Project structure

```
backend/
  main.py              FastAPI app and routes
  config.py            settings and calibration
  verify_setup.py      environment and model check
  requirements.txt
  scripts/fetch_model_files.py
  aasist_model/        AASIST.pth checkpoint, LICENSE, meta.yaml, SOURCE.json
frontend/
  index.html           user interface
third_party/aasist/    upstream AASIST code (clovaai/aasist)
```

## Setup (Windows, Python 3.11)

```powershell
git clone https://github.com/varshithdornala/VoiceGuard.git
cd VoiceGuard
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r backend\requirements.txt
python backend\verify_setup.py
```

The checkpoint is included in `backend/aasist_model/`. To re-download it, run `python backend\scripts\fetch_model_files.py`.
No NVIDIA GPU? Install the CPU build of PyTorch instead; the model is small enough to run on CPU.

## Run

```powershell
uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000 in a browser.

## Verification results

Environment check (`verify_setup.py`): **22 PASS, 1 WARN, 0 FAIL**
(the one warning only notes Python 3.11 versus the team's earlier 3.10 standard; 3.11 is fine).

| Item | Result |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.7.0+cu126 (CUDA 12.6) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM, compute capability 8.9 |
| GPU compute test | 1024x1024 matmul in about 0.55 ms |
| Audio decoders | WAV, FLAC, OGG, MP3 present |
| Checkpoint | `AASIST.pth`, SHA-256 `51d2d9cf...1a1c0`, identical to the official clovaai/aasist checkpoint |

## Test results

TODO: replace with real results from our own test clips.

| Clip | Type | Score | Verdict |
|---|---|---|---|
| TODO | Real voice | TODO | TODO |
| TODO | AI-generated | TODO | TODO |
| TODO | MP3-compressed | TODO | TODO |

Average analysis time per clip: TODO

## Limitations

- Accuracy drops on attack types not seen in training, heavily compressed or noisy audio, and very short clips.
- Scores are probabilistic and should support a human decision, not replace it.
- Results may vary across accents, languages and recording conditions.

## Privacy

TODO: state whether uploaded audio is stored or deleted after analysis.

## Acknowledgements

- AASIST: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks", ICASSP 2022. Code and checkpoint from [clovaai/aasist](https://github.com/clovaai/aasist) (see `third_party/aasist/LICENSE`).
