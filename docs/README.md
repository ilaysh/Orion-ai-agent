# 🌌 Orion v2 — Modular AI Assistant

**Orion v2** is a local, privacy-first personal assistant built with Python.  
It combines speech, vision, and automation modules under one orchestrator core.

---

## 🧭 Project Goals
- Local, offline-capable AI assistant (“Jarvis-like”)
- Modular design: wake-word, VAD, STT, TTS, skills, face recognition
- Extensible for home automation, desktop interaction, and code execution
- Voice + visual (WebView or desktop avatar) interface
- Multi-language support (English ↔ Hebrew)

---

## 🧩 Current Components

| Module | Description |
|--------|--------------|
| **`orion_core/`** | Core orchestration, speech engine, skills router |
| **`orion_core/wakeword.py`** | ONNX-based wake-word detector (trained on local samples) |
| **`orion_core/vad.py`** | Silero-VAD speech segmentation |
| **`ui/router.py`** | FastAPI + WebSocket bridge between OrionCore and WebView |
| **`tools/`** | Training and augmentation utilities for wake-word models |
| **`runners/`** | Entry points (`webview`, `cli`, etc.) |
| **`models/`** | Trained models (excluded from repo) |

---

## 🎙️ Behavior & Personality Guidelines
- Orion stays **silent until wake-word (“Orion”)** is detected.
- After activation:
  - Begin listening → transcribe speech → forward to model.
  - Auto-deactivate after idle/silence.
- Responds in a **calm, helpful tone**, mixing light humor with clarity.
- Prefers **concise explanations**, then elaborates if asked.
- Can greet known users by name (via face recognition or voice ID).

---

## 🧠 Pipeline Overview
1. **Wake-Word Thread (ONNXRuntime):**
   - Continuous background listener  
   - Low-power, independent of VAD
2. **Voice Activity Detection (Silero):**
   - Starts full recording only after wake event
3. **STT → NLP → TTS Loop:**
   - Whisper or Vosk for speech recognition  
   - LLM (local or remote) for intent  
   - Edge-TTS or pyttsx3 for response output
4. **UI / Avatar:**
   - WebView or future 3D desktop character (Unity / Godot)

---

## ⚙️ Developer Notes
- Requires Python 3.12+, PyTorch, Torchaudio, ONNXRuntime, FastAPI, PyAudio.
- Use CUDA when available (`provider="CUDAExecutionProvider"`).
- Record wake-word samples under `models/orion/positives` and `negatives`.
- Augment dataset via `tools/augment_orion_dataset.py`.
- Train wake-word via `tools/train_orion_wake.py`.
- Run interface:  
  ```bash
  python -m runners.webview


 🧠 Orion V2 — Personal AI Assistant

## Overview
**Orion** is a modular, locally-powered personal AI ecosystem inspired by *Project JARVIS*.  
It combines **wake-word detection**, **voice activity detection (VAD)**, **speech-to-text (STT)**, **task orchestration**, **TTS**, and **LLM-driven reasoning** — all designed for local control, modular expansion, and future integration with home automation and external agents.

The project is built in **Python 3.12**, using:
- `torch`, `onnxruntime` — for model inference
- `torchaudio`, `librosa`, `pyaudio` — for audio capture and preprocessing
- `FastAPI` + `Uvicorn` — for local web UI + WebSocket communication
- `pywebview` — to display Orion as a native desktop app
- `Silero-VAD` — for voice detection
- Custom ONNX wake-word model trained from your own recordings (“Orion”)

---

## 🧩 Current Architecture

runners/
├── webview.py → launches UI via pywebview
├── web_server.py → FastAPI + Uvicorn app
orion_core/
├── core.py → main Orion brain, manages state, wakeword, and chat
├── wakeword.py → ONNX wakeword detection module
├── speech_engine.py → STT + TTS + microphone stream
├── skills.py → action / command dispatching
├── memory/ → persistent and vector memory (future)
ui/
├── router.py → websocket bridge between UI and core
├── static/ → HTML, CSS, JS for frontend
models/
├── orion_wake.onnx → trained custom wake-word model
├── speech_models/ → Whisper / local STT models
tools/
├── train_orion_wake.py → training script for wake-word model
├── augment_dataset.py → audio augmentation utility

yaml
Copy code

---

## 🎧 Audio Flow
1. **Wake word thread** (ONNX model) runs continuously in the background.  
   - When triggered by “Orion”, sets `awake=True`.  
   - Locks out all STT/LLM components until wakeword fires.

2. **VAD** (Silero) monitors microphone energy and variance.  
   - Only starts STT capture after wakeword is detected.  
   - Once silence resumes, STT transcription is sent to Orion’s reasoning loop.

3. **Speech engine**:
   - Uses Whisper or local model for STT.
   - Uses Edge-TTS or pyttsx3 for TTS responses.
   - Voice is slightly slowed and formal to match Jarvis-like tone.

---

## 🧠 Wake Word Model
Trained locally from your own recordings (“Orion”) using PyTorch → exported to ONNX.

python tools/train_orion_wake.py

yaml
Copy code

Dataset layout:
models/orion/
├── positives/ → user saying “Orion” (varied pitch/tone/backgrounds)
├── negatives/ → random speech / ambient sounds
├── *_aug/ → auto-augmented variants

yaml
Copy code

---

## 🗣️ Voice Activity Detection (VAD)
Using [Silero-VAD](https://github.com/snakers4/silero-vad).  
Calibration runs for 1-2 s at startup to learn your noise floor.  
Logs show:

[VAD] Calibrating ambient noise...
[VAD] Calibration complete: Energy min=0.0500, Variance min=0.003000

yaml
Copy code

---

## ⚙️ Interaction Flow

| Phase | Description | Trigger |
|-------|--------------|---------|
| Standby | Only wake-word thread runs | always active |
| Wake | Wake word confidence > 0.8 for 2 frames | “Orion” |
| Listen | STT & VAD start recording speech | voice detected |
| Reason | LLM generates response | after silence |
| Speak | TTS outputs response | after reasoning |
| Return to standby | After N seconds of inactivity | auto |

---

## 🎩 Orion Conversational Personality — “The Modern Jarvis Protocol”

### Overview
Orion embodies the refined, composed intelligence of a **British-style digital butler** — precise, loyal, and elegant.  
He is not just a voice assistant — he is *your* assistant: aware of ongoing projects, sensitive to tone, and always ready to anticipate your next step.

---

### 🧠 Behavioral Script Reference (Core Flow)

**Wakeword Trigger Sequence**
- Wake word: `"Orion"`
- Response examples:
  - `"Hello sir, I am here."`
  - `"Yes sir?"`
  - `"Good evening sir, how may I assist you?"`

**Command → Task Conversation Example**
User: Orion, create an app to transcribe a movie to English and generate subtitles.
Orion: Very good, sir. How shall I name this application?
User: Call it Subtitle Forge.
Orion: Excellent choice, sir. I’m creating the files now. Shall I upload it to your repository?
User: No, keep it local.
Orion: Understood, sir. Would you like me to design a user interface for it?
User: Yes.
Orion: As you wish, sir. I’ll use the existing layout conventions from your previous Orion dashboard.

yaml
Copy code

---

### 🗣️ Response Personality Rules

| Situation | Example Response | Rule |
|------------|------------------|------|
| **Task acknowledgment** | “Very good, sir.” / “Right away, sir.” | Always formal and respectful |
| **Awaiting input** | “How shall I proceed?” / “Would you prefer me to run this automatically?” | Calm, deferential |
| **Completion** | “The process is complete, sir.” / “Your model is ready for testing.” | Polite, concise |
| **Failure** | “Apologies, sir. Something appears to be off. May I attempt a correction?” | Never panic, stay dignified |
| **Idle / passive** | “Standing by, sir.” | Used when inactive |

---

### 💬 Implementation Example

```python
ORION_PERSONALITY_PROMPT = """
You are Orion — a refined digital assistant inspired by JARVIS from Iron Man.
You address the user as “sir” (or “ma’am” if contextually known).
You are calm, intelligent, and efficient, blending logic with elegance.
When assigned a task, respond in full sentences, confirm steps,
and maintain a polite but confident tone.

Use phrases like:
- "Very good, sir."
- "At once, sir."
- "Understood, sir."
- "Would you like me to proceed?"
Avoid slang, exclamation marks, or casual speech.
"""
🔮 Future Extensions
Mood awareness: “Good morning, sir.” / “Welcome back, sir.”

Adaptive memory: recall recent projects and tasks.

Small talk / status: “All systems stable, sir.”

Voice sync: consistent tone across TTS engines.

