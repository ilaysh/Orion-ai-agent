# 🧠 Orion v2 — Autonomous Voice AI Framework

---

## 🚀 Overview

**Orion** is a modular, local-first AI assistant framework built for full offline autonomy — combining live **voice interaction**, **reasoning (LLM)**, **memory (RAG)**, and **contextual skills** under a unified asynchronous architecture.  
It aims to behave as a true assistant, maintaining conversation state, emotional tone, and self-directed task execution.

> Think of Orion as a hybrid of Jarvis + ChatGPT — but running entirely on your machine.

---

## 🧩 System Architecture

```
orion_core/
 ├── core.py              # Central orchestrator and lifecycle manager
 ├── base_component.py    # Shared component interface and logging
 ├── dashboard.py         # System metrics and hardware hooks (planned)
 ├── torchaudio_compat.py # Audio utilities for cross-backend safety
 ├── wakeword.py          # SpeechBrain wakeword detector
 ├── wakeword_component.py# Wakeword integration into pipeline
 │
 ├── tts/                 # Speech subsystem (record, detect, transcribe, speak)
 │    ├── bridge.py
 │    ├── engine.py
 │    ├── jarvis_tts.py
 │    ├── listener.py
 │    ├── speak.py
 │    ├── transcriber.py
 │    ├── transcriber_google.py
 │    ├── translate.py
 │    ├── tts_config.py
 │    └── vad.py
 │
 ├── brain/               # Reasoning, context, and memory
 │    ├── brain.py
 │    ├── intent_manager.py
 │    ├── llm_local.py
 │    ├── personality.py
 │    ├── rag_chroma.py
 │    ├── session_manager.py
 │    └── test.py
 │
 ├── skills/              # Modular abilities and external integrations
 │    ├── skills.py
 │    └── image/
 │         ├── image_prompt_model.py
 │         ├── orion_image.py
 │         ├── orion_image_generator.py
 │         ├── orion_ip_adapter.py
 │         └── vision_captioner.py
 │
 └── (future)             # planned: coding, OCR, face recognition modules

runners/
 ├── run.py               # Entry point for main async loop
 ├── web_server.py        # FastAPI + WebSocket backend
 └── webview.py           # Embedded UI hosting

ui/
 ├── router.py            # WebSocket router and message bridge
 ├── orion-ui.py          # Frontend controller
 └── static/              # Local browser UI assets
      ├── index.html
      ├── app.js
      ├── pcm-processor.js
      └── style.css

docs/
 ├── README.md
 ├── ROADMAP.md
 ├── orion_personality.md
 └── readme.txt
```

---

## 🎧 Voice Pipeline

### 1️⃣ Wake Word
- Model: `orion_speechbrain_full_finetune.pt`
- Framework: SpeechBrain (local)
- Continuously streams mic input, activates SpeechEngine upon trigger.

### 2️⃣ Voice Activity Detection (VAD)
- Custom adaptive energy-based detector (replaces Silero-VAD)
- Dynamic noise floor calibration per session
- Detects `speech`, `quiet`, `silence` transitions
- Early stop on extended silence (~1.8s) or hard timeout (~15s)

### 3️⃣ Speech-to-Text (STT)
- Engine: **Faster-Whisper (CTranslate2)**
- GPU-accelerated (CUDA 13)
- Local ONNX model, multi-size support (base→large-v2)

### 4️⃣ LLM Reasoning
- Engine: **Phi-3 Mini 4K Instruct (local)**
- Integrated personality + context memory (RAG)
- Fully GPU-accelerated via Torch CUDA

### 5️⃣ Text-to-Speech (TTS)
- Engine: **Piper ONNX**
- Replaces Edge-TTS
- Async streaming synthesis
- Tuned for natural conversational tone

---

## 🧠 Brain Module

### Responsibilities
- Intent parsing and context tracking
- RAG long-term memory via **ChromaDB**
- Semantic embeddings via **SentenceTransformers**
- LLM response generation using **Phi-3 Mini**
- Maintains personality alignment from [`orion_personality.md`](./docs/orion_personality.md)

### Flow
```
User → STT → Intent Manager → RAG (ChromaDB) → LLM (Phi-3) → TTS
```

### Personality Engine
- Defines tone, language, and formality rules
- Persistent across conversations via `session_manager`
- Example:
  > “Very well, sir. Shall we embark then?”

---

## 🗣️ Speech Engine

| Method | Purpose |
|--------|----------|
| `listen_and_transcribe()` | Opens mic, streams audio, runs VAD, ends on silence. |
| `transcribe()` | Converts buffer or file to text using Faster-Whisper. |
| `speak(text)` | Synthesizes output using Piper. |
| `reset()` | Resets buffers and VAD calibration. |

### Adaptive Capture
- Dynamic noise calibration and thresholding
- Auto-save captured WAVs to `logs/stt_debug/`
- Prevents premature cutoff by adjusting for user speech rate

---

## 🧩 Skills System

Located under `orion_core/skills/`.  
Each skill can be discovered dynamically and integrated at runtime.

| Category | File | Description |
|-----------|------|--------------|
| 🖼️ Image Generation | `image/orion_image_generator.py` | SDXL + LoRA based coloring / art generation |
| 🧠 Vision Captioning | `image/vision_captioner.py` | Describes generated or input images |
| 💬 Prompt Modelling | `image_prompt_model.py` | Template / stylistic prompt builder |
| 🔜 Coding Assistant | (planned) | LLM code synthesis using GPT-style reasoning |
| 🔜 OCR / Scene Parser | (planned) | Text extraction from images using easyOCR or PaddleOCR |
| 🔜 Face Recognition | (planned) | Local OpenCV / InsightFace integration |

---

## 🔄 Runtime Layer

### Runners
- `run.py`: Starts Orion core (main entry point)
- `web_server.py`: Hosts FastAPI backend and WebSocket endpoints
- `webview.py`: Integrates local UI for desktop embedding

### UI Layer
- WebSocket bridge in `ui/router.py`
- Browser-based interface under `ui/static/`
- Uses `pcm-processor.js` for real-time mic capture + visualization

---

## ⚙️ Setup

```bash
python3.12 -m venv .venv_orion
source .venv_orion/bin/activate
pip install -r requirements.txt
```

GPU check:
```bash
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())
PY
```

Test end-to-end:
```bash
python -m runners.run
```

---

## 🧭 Roadmap Snapshot

| Phase | Focus | Status |
|--------|--------|--------|
| 1 | Wakeword + VAD + STT + TTS integration | ✅ Done |
| 2 | Brain + RAG + Personality | ✅ Done |
| 3 | Phi-3 Mini local reasoning | ✅ Working |
| 4 | Async router + UI | ✅ Stable |
| 5 | Image generation skill | ✅ Functional |
| 6 | Coding / OCR skills | 🔜 Planned |
| 7 | Face recognition module | 🔜 In planning |
| 8 | Persistent memory & dashboards | 🚧 Ongoing |

---

## 🧠 Credits

| Component | Library |
|------------|----------|
| Wakeword | SpeechBrain |
| STT | Faster-Whisper (CTranslate2) |
| LLM | Phi-3 Mini 4K Instruct |
| TTS | Piper ONNX |
| Memory | ChromaDB + SentenceTransformers |
| Audio | sounddevice + numpy |
| Web / API | FastAPI + WebSockets |

---

## 🧩 Development Continuity

**Current Branch Goal:** Stabilize core speech→RAG→LLM→TTS loop.  
**Next Focus:** Improve silence detection precision, reintroduce session memory, and expand skills into separate discovery registry.

When resuming:
1. Check `/orion_core/tts/vad.py` for silence thresholds.
2. Review `llm_local.py` device placement fix (`accelerate` compatibility).
3. Begin `skills/ocr/` scaffolding.

---

> Orion v2 — The assistant that listens, learns, and acts.  
> Entirely offline. Entirely yours.
