# orion_core/brain/llm/chat_engine.py
import asyncio
import os

# CRITICAL FIX: Stop JAX from stealing 90% of VRAM on startup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# LAZY IMPORT: vllm will be imported only when load() is called to avoid eager initialization

_chat_instance = None

def get_chat_engine():
    global _chat_instance
    if _chat_instance is None:
        _chat_instance = ChatEngine()
    return _chat_instance

class ChatEngine:
    def __init__(self):
        self.engine = None
        # Using AWQ model - standard model is too large (~15GB) for 16GB VRAM
        self.model_path = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"

    async def load(self):
        if self.engine: return
        
        # LAZY IMPORT: Import vllm only when needed
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        
        print(f"[ChatEngine] 🚀 Loading Coder Brain: {self.model_path}")
        
        # VRAM CONFIGURATION (RTX 4060 Ti 16GB)
        # 1. gpu_memory_utilization=0.6 -> Reserves ~9.6GB for Brain. 
        #    Leaves ~6.4GB for Whisper (1.5GB) + System + Screen.
        # 2. max_model_len=8192 -> 8k Context is safer for stability.
        #    (We can try pushing back to 16k later if stable).
        args = AsyncEngineArgs(
            model=self.model_path,
            quantization="awq",
            dtype="float16",
            gpu_memory_utilization=0.6, 
            max_model_len=8192,
            enforce_eager=True,
            disable_log_stats=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        print("[ChatEngine] ✔ Coder Brain loaded (8k Context Active).")

    async def unload(self):
        if self.engine:
            import gc
            import torch
            self.engine = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[ChatEngine] 💤 Hibernated.")

    async def generate_chat(self, prompt: str, system_prompt: str, **kwargs) -> str:
        if not self.engine: await self.load()
        
        # Import SamplingParams here since it's only used in this method
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        request_id = f"req_{asyncio.get_event_loop().time()}"
        results_generator = self.engine.generate(
            full_prompt,
            sampling_params,
            request_id
        )

        final_text = ""
        async for request_output in results_generator:
            final_text = request_output.outputs[0].text

        return final_text.strip()