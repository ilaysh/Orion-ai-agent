# ⚙️ Orion Skills Directory
*(Active + Experimental + Planned modules)*

---

## 🧩 Core Skills (Active)

| Skill | Path | Description |
|--------|------|--------------|
| 🖼️ Image Generation | `skills/image/orion_image_generator.py` | Generates art, coloring pages, or creative visuals via SDXL + LoRA |
| 🎨 Vision Captioning | `skills/image/vision_captioner.py` | Converts images into natural language descriptions |
| 🧠 Prompt Modeling | `skills/image/image_prompt_model.py` | Dynamically builds descriptive prompts for visual or text generation |
| 💬 Intent Manager | `brain/intent_manager.py` | Parses voice input into actionable intents |
| 🗂️ Session Memory | `brain/session_manager.py` | Maintains conversational continuity and saves logs |
| 🧭 Personality Engine | `brain/personality.py` | Controls tone, language, and personality rules |

---

## 🧠 System Interaction Skills (Planned)

| Skill | Category | Description | Status |
|--------|-----------|-------------|--------|
| ⚙️ Device Control | Hardware API | Control PC RGB, fans, mouse color via OpenRGB / dbus | 🔜 Planned |
| 💡 Smart Home Integration | IoT / MQTT | Interface with smart lights or sensors | 🔜 Planned |
| 🧾 File Operations | OS / Local | Search, open, or organize files by voice | 🔜 Planned |
| 🧠 Dashboard | System | Real-time CPU/GPU/fan stats with voice overlay | 🚧 Partial |
| 🔧 Self Diagnostics | System | Self-check of pipeline health, latency, and mic state | 🔜 Planned |

---

## 👁️ Vision & Perception Skills (Future)

| Skill | Path | Description | Status |
|--------|------|-------------|--------|
| 🧍 Face Recognition | `skills/vision/face_recognition.py` | Detects and identifies faces using OpenCV + embeddings | 🔜 Planned |
| 🧾 OCR Parsing | `skills/vision/ocr_parser.py` | Extracts text from documents/images using PaddleOCR | 🔜 Planned |
| 🔍 Scene Understanding | (future) | Combines vision + text + audio for full context | 🔜 Research |

---

## 💻 Cognitive & Creative Skills (Future)

| Skill | Category | Description | Status |
|--------|-----------|-------------|--------|
| 🧰 Coding Assistant | Development | Generate runnable projects in multiple languages | 🔜 Design stage |
| 🧱 Code Reviewer | Dev Tools | Evaluate and debug user codebases | 🔜 Future |
| 🎭 Personality Emulator | LLM Fine-Tuning | Modify tone dynamically (e.g. teacher, assistant, friend) | 🚧 Partial |
| 🗓️ Planner | Reasoning | Multi-step goal execution and scheduling | 🔜 Future milestone |

---

## 🧠 Skill Discovery System
Each skill module will:
- Register automatically in `skills/skills.py`
- Expose a `manifest` (`name`, `category`, `intent`, `description`)
- Allow Orion to **discover, enable, or disable** skills dynamically.

Example manifest:
```python
manifest = {
    "name": "Face Recognition",
    "category": "Vision",
    "description": "Detects known faces and greets them by name.",
    "triggers": ["scan face", "who is here"]
}

🧩 Next Implementation Targets

 Skill discovery and registry system

 Face recognition (OpenCV + embeddings)

 OCR / text-in-image recognition

 Device control (fans, RGB, mouse color)

 Code generation + local app creation

 Emotion synthesis in TTS