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
        self.model_name = "Orion"
        self.client = AsyncOpenAI(
            base_url="http://0.0.0.0:8000/v1",
            api_key="EMPTY"
        )
        
   
    async def unload(self):
        print("[ChatEngine] 💤 Unloading requested, but API Server manages VRAM.")
        pass

    async def load(self):
        print(f"[ChatEngine] 🚀 Connecting to local API Server: {self.model_name}")
        print("[ChatEngine] 🔥 Warming up Gemma 4 neural pathways...")
        try:
            from orion_core.brain.personality import Personality
            p = Personality()
            real_system_prompt = p.get_system_prompt([])
            # --- FIXED FOR NATIVE GEMMA 4 TOKENS ---
            warmup_prompt = f"<|turn>system\n{real_system_prompt}\n\nYOU MUST RESPOND ONLY IN VALID JSON.<turn|>\n"
            await self.client.completions.create(
                model=self.model_name,
                prompt=warmup_prompt,
                max_tokens=1
            )
            print("[ChatEngine] ✅ Neural Warmup Complete.")
        except Exception as e:
            print(f"[ChatEngine] ⚠️ API Warmup skipped: {e}")

    async def generate_chat(self, prompt: str, system_prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.1)

        # --- FIXED FOR NATIVE GEMMA 4 PREFILL SCHEMA ---
        full_prompt = (
            f"<|turn>system\n{system_prompt}\n\nYOU MUST RESPOND ONLY IN VALID JSON.<turn|>\n"
            f"<|turn>user\n{prompt}<turn|>\n"
            f"<|turn>model\n```json\n{{\n"
        )
        
        try:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "repetition_penalty": 1.15,
                    "frequency_penalty": 0.1,
                    # --- UPDATED EOG STOP TOKENS ---
                    "stop": ["<turn|>", "<eos>"]
                }
            )
            full_response = response.choices[0].text
        except Exception as e:
            print(f"[ChatEngine] 💥 API Error: {e}")
            full_response = ""
            
        final_text = "{\n" + full_response
        
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