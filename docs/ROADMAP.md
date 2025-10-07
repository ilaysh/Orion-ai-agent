# Orion Development Roadmap

This roadmap defines the staged development of Orion v2 — starting with core stability, then intelligence, then self-expansion.  
It is intended to be a living document. Orion itself may later update this roadmap when self-modification is enabled.

---

## Phase 1 – Core Stability (Immediate Priorities)
- [ ] **STT (Speech-to-Text)**  
  - Fix VAD (robust silence detection).  
  - Finalize Whisper integration (works on CUDA 13).  
  - Add microphone selector in UI.  
  - Support Hebrew + English seamlessly.  

- [ ] **TTS (Text-to-Speech)**  
  - Ensure playback does not feed back into microphone.  
  - Correct “speaking → listening” transitions.  
  - Normalize audio levels.  

- [ ] **State Machine**  
  - Core owns transitions: `idle → listening → thinking → speaking → listening`.  
  - Router/UI only mirrors state.  
  - Add timeout + end-of-speech handling.  

---

## Phase 2 – Intelligence Backbone
- [ ] **Vector DB integration**  
  - Store face embeddings, user profiles, and skills metadata.  
  - Store conversation memory + long-term summaries.  

- [ ] **Model integration**  
  - Add 7B/13B LLM (local or API).  
  - Replace hardcoded skills with model-based intent parsing.  
  - Allow model to suggest Python snippets for skills.  

- [ ] **Skill execution layer**  
  - Standard interface: `intent → parameters → execute → result`.  
  - Keep built-in skills for system/time/math.  
  - Model can register new skills dynamically.  

---

## Phase 3 – Self-Expansion & Autonomy
- [ ] **Self-creating skills**  
  - Orion can ask clarifying questions.  
  - Generate new `skills/*.py` files.  
  - Register skill automatically.  
  - Optionally push to Git branch + PR.  

- [ ] **Authentication / personalization**  
  - Fac
