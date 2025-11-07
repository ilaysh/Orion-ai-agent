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


🗣️ Speech Engine Behavior & Calibration

VAD calibration runs once on engine initialization (SpeechEngine.init()).

Orion automatically learns ambient noise and adjusts dynamically.

Silence timeout: ~5 s default, adaptive based on speech rhythm.

Short sentences: transcribed immediately after silence (< 1 s).

Long sentences: captured until stable quiet detected (≈ 8 s max).

listen_and_transcribe() simply opens the mic → records → calls _transcribe(audio).

Core only calls transcribe() and never re-calibrates unless requested.

🔔 Wake-Word Response Logic

Wakeword detected → Orion immediately replies using personality rules (e.g.,
“Yes sir?”, “Hello sir, I’m here.”, or “Good evening sir, how may I assist you?”).

After greeting, Orion enters Listening state (LISTEN) for user speech.

If no text is transcribed within 6 s:

Orion politely rechecks: “Did you call me, sir?”

If still silent after another 4 s → returns to Idle (IDLE).

Once a valid transcript arrives, Orion routes it to intent handling → skills → response.

🎩 Personality & Tone (Jarvis-Style)

Always addresses user as “sir” or “ma’am” when known.

Tone: calm, confident, concise.

Phrases:

“Very good, sir.”

“At once, sir.”

“Would you like me to proceed?”

When idle or waiting: “Standing by, sir.”

When re-engaging after silence: “Do you still need me, sir?”

The prompt template lives in orion_core/prompt_templates/orion_personality.md and is loaded into the LLM/TTS chain.

In future updates: personality weights can adjust tone (e.g., warm ↔ formal) or recognize users via face ID.

🧭 Next Immediate Steps (Phase 1.5)

✅ Finalize VAD + STT integration (stable across sessions).

🔜 Implement wakeword greeting response in core.handle_wake_event().

🔜 Add post-silence check (“Did you call me, sir?” if no STT text).

🔜 Load personality prompt automatically into chat pipeline.

🔜 Add simple skill responses (e.g., “What’s the time?”, “Check the weather”).

🧩 Orion Skill: Image Generation (orion_image_generator.py)
Description

Orion’s image generation skill enables the creation of high-quality photo-realistic and semi-realistic (anime-inspired) artwork using local Stable Diffusion XL (SDXL) and RealVisXL models.
It supports both text-to-image and image-to-image (reference-based) generation, with optional automatic refinement for improved detail and realism.

Core Features
Feature	Description
Text-to-Image	Generate new scenes, portraits, or concepts from natural language prompts.
Image-to-Image	Use a reference image (e.g. character art, design) to maintain color and tone but reimagine pose, lighting, and realism.
Dual Style Presets	--style realistic → cinematic, photographic look
--style semi → semi-realistic, anime-inspired rendering
Dual Output Mode	--make-both creates both styles in one run.
Auto-Refinement Pass	A second low-strength img2img pass enhances details, corrects hands, eyes, props, and texture realism. The first image is archived in /beta/.
Automatic Directory Handling	Input and output default to ~/Pictures and ~/Pictures/orion_outputs/.
GPU Optimized	Uses memory-efficient features (VAE tiling, CPU offload, attention slicing). Compatible with CUDA 13.
Command Examples
Generate a semi-realistic Sakura (with reference image)
python orion_image_generator.py \
  "a young woman inspired by Sakura Haruno from Naruto, pink short hair, red outfit, facing the camera, kunai between her teeth, confident expression" \
  --from-img SakuraShip.webp \
  --style semi \
  --strength 0.7 \
  --use-refiner

Generate both styles
python orion_image_generator.py \
  "portrait of a young woman inspired by Sakura Haruno from Naruto" \
  --from-img SakuraShip.webp \
  --make-both

Skip refinement
python orion_image_generator.py \
  "portrait of a character inspired by Mavuika from Wuthering Waves" \
  --from-img Mavuika.webp \
  --no-refine

File Output
Type	Path
Final image	~/Pictures/orion_outputs/
Beta (first pass)	~/Pictures/orion_outputs/beta/

Files are automatically timestamped and include style suffixes when --make-both is active.

Technical Notes

Models Used:

Base: SG161222/RealVisXL_V4.0

Alternative: stabilityai/stable-diffusion-xl-base-1.0

Refiner: stabilityai/stable-diffusion-xl-refiner-1.0

Recommended Strength Ranges:

Text-only: N/A

Img2Img (reference): 0.55–0.7

Auto-refine: 0.4–0.5

Recommended Steps:

Base: 60–80

Refinement: 30–45

Guidance Scale:

7.0–8.5 depending on desired prompt adherence.

Future Skill Integration

Once Orion’s prompt understanding improves, this skill will serve as the backend for:

Natural prompt requests such as

“Create an image of Mavuika facing the camera in a forest.”

Automatic style adaptation (realistic / semi / anime) inferred from user preference.

Self-generated artistic descriptors via Orion’s internal language model.

Integration Plan (for Orion’s AI Core)

Purpose:
Allow Orion to autonomously generate or refine images when the user requests it naturally.

Trigger Examples:

“Orion, create a realistic portrait of Mavuika.”

“Draw Sakura from Naruto in a new pose.”

“Make a semi-realistic coloring page for Leroy.”

Internal Behavior (planned):

Orion interprets the request using NLP intent classification → image.create.

The system determines style (realistic, semi, or coloring).

Orion composes a prompt using descriptive tags (e.g., lighting, pose, mood).

The assistant calls:

python orion_image_generator.py "<generated prompt>" [--style ...] [--from-img ...] [--make-both]


Once the image is created, Orion previews or opens the output file from
~/Pictures/orion_outputs/.

Next Stages:

Add a Prompt Builder module to dynamically generate detailed artistic prompts.

Integrate with the Vector Memory so Orion remembers user preferences (e.g. “Leroy likes Paw Patrol coloring pages”).

Connect with voice commands for natural spoken requests.