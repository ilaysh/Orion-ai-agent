# 🧭 Orion v2 Development Roadmap
*(Internal reference for continuity between project sessions)*

---

## 🌍 Phase 1 — Core Systems (✅ Completed)
**Goal:** Establish stable real-time voice → reasoning → response loop.

| Module | Description | Status |
|---------|--------------|--------|
| Wakeword | SpeechBrain-based wake trigger | ✅ Stable |
| VAD | Adaptive energy + silence detection (custom) | ✅ Stable |
| STT | Faster-Whisper (CTranslate2) GPU-accelerated | ✅ Stable |
| LLM | Phi-3 Mini 4K Instruct (local, CUDA 13) | ✅ Working |
| TTS | Piper ONNX (Edge-TTS replaced) | ✅ Stable |
| Router/UI | Async FastAPI + WebSocket interface | ✅ Functional |
| Personality | Persistent personality profiles | ✅ Integrated |

---

## 🧠 Phase 2 — Context and Intelligence (🚧 Ongoing)
**Goal:** Give Orion long-term awareness, intent understanding, and emotional continuity.

| Module | Description | Status |
|---------|--------------|--------|
| RAG Memory | ChromaDB persistent storage | ✅ Indexed |
| Sentence Embeddings | SentenceTransformers for retrieval | ✅ Integrated |
| Intent Manager | Lightweight intent + action matching | ✅ Functional |
| Session Manager | Maintain conversational continuity | 🚧 Partial |
| Personality Refinement | Context-adaptive tone control | 🚧 In Progress |
| Emotion Modeling | Affect-driven phrasing and TTS tone | 🔜 Planned |

**Upcoming Improvements**
- Improve VAD-to-STT transition timing.
- Memory recall tuning (relevance, freshness scoring).
- LLM personality embedding injection refinement.

---

## ⚙️ Phase 3 — Autonomy and Skills (🔜 Development)
**Goal:** Enable Orion to perform actions beyond dialogue.

| Category | Example Skill | Description | Status |
|-----------|----------------|--------------|--------|
| Image Generation | Coloring book / creative SDXL tools | ✅ Working |
| Dashboard | System metrics, fan/mouse/RGB control | 🚧 Partial |
| File & System | Voice-controlled file operations | 🔜 Planned |
| Coding Assistant | Generate runnable projects | 🔜 Planned |
| Face Recognition | OpenCV + InsightFace integration | 🔜 Planned |
| OCR / Vision | Text extraction via PaddleOCR | 🔜 Planned |
| Device Control API | Smart device + PC peripheral management | 🔜 Planned |
| Autonomous Actions | Multi-step task planning | 🔜 Future milestone |

---

## 🧩 Phase 4 — Interface and Experience (🧱 Upcoming)
**Goal:** Improve user interaction, usability, and visual feedback.

- ✅ Functional local web UI
- 🚧 Add real-time waveform visualization
- 🔜 Dynamic personality avatar integration
- 🔜 Emotion display (facial / voice)
- 🔜 Multi-session dashboard UI
- 🔜 “Memory View” interface for context inspection

---

## 🧱 Phase 5 — Long-Term Goals
**Goal:** True autonomy, long-term contextual retention, and persistent operation.

- Offline self-learning through session re-ingestion
- Fully autonomous skill discovery / self-expansion
- Secure remote interface for device orchestration
- Continuous multi-device syncing (desktop, dashboard, Android)

---

### 🔄 Current Development Focus
> **November 2025**
> - Fine-tune VAD silence thresholds  
> - Finalize session memory persistence  
> - Add skills registry system  
> - Begin face recognition and coding assistant integration

---

> Orion is now self-contained: wakeword → VAD → STT → RAG → LLM → TTS → response.
> Next: let it *act*, *remember*, and *decide*.
