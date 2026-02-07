# orion_core/brain/llm/coder_engine.py
import os
import gc
from llama_cpp import Llama

class CoderEngine:
    def __init__(self):
        self.model = None
        # Ensure this points to your specific model file
        self.model_path = os.path.abspath("models/deepseek-coder-6.7b-instruct.Q5_K_M.gguf")
        
        # TUNED FOR STABILITY (RTX 3060/4060/5060)
        self.config = {
            "n_ctx": 2048,          # Lowered from 4096 to prevent OOM crash
            "n_gpu_layers": -1,     # Offload everything to GPU
            "n_batch": 256,         # Lowered from 512 for smoother swapping
            "verbose": False,       # Reduce log spam
            "n_threads": 6
        }

    async def load(self):
        if self.model: return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"⚠️ Architect Model missing: {self.model_path}")

        print(f"[CoderEngine] 🚀 Loading Architect (DeepSeek)...")
        try:
            self.model = Llama(model_path=self.model_path, **self.config)
        except Exception as e:
            print(f"[CoderEngine] ❌ Init Failed: {e}")
            # Fallback: If 2048 is still too big, try 1024
            if "context" in str(e).lower() or "memory" in str(e).lower():
                print("[CoderEngine] 📉 Retrying with minimal context (1024)...")
                self.config["n_ctx"] = 1024
                self.model = Llama(model_path=self.model_path, **self.config)
            else:
                raise e

    async def unload(self):
        if self.model:
            print("[CoderEngine] 🔻 Unloading Architect...")
            self.model.close()
            del self.model
            self.model = None
            gc.collect()

    async def generate_code(self, prompt: str, system_prompt: str, max_new_tokens=1024, temperature=0.1) -> str:
        if not self.model: await self.load()
        
        # DeepSeek-Coder specific formatting
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        output = self.model(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "```\n\n"]
        )
        
        # Return just the raw code/text, no conversational wrapper
        return output['choices'][0]['text']

_engine = CoderEngine()
def get_coder_engine():
    return _engine