# tries Ollama first; falls back to tiny transformers model
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct"  # change if needed

_tokenizer = None
_model = None


def _fallback():
    global _tokenizer, _model
    if _model:
        return _tokenizer, _model
    name = "microsoft/phi-3-mini-4k-instruct"
    _tokenizer = AutoTokenizer.from_pretrained(name)
    _model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto"
    )
    return _tokenizer, _model


def generate(prompt: str, model_path=None, temperature=0.7, max_new_tokens=512):
    """
    Generate text response from a local LLM (Ollama or HF model).
    Includes safety for truncated decoding.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model_path = model_path or "microsoft/Phi-3-mini-4k-instruct"
    print(f"[LLM] 🚀 Generating with {model_path} (temperature={temperature})")

    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        if not model.device.type.startswith("cuda"):
            model = model.to("cuda")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok
        )

        # 🧠 Run generation
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95
        )

        text = out[0]["generated_text"]
        # 🔒 Safe decoding & marker handling
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>", 1)[-1]
        elif "Assistant:" in text:
            text = text.split("Assistant:", 1)[-1]
        elif "Response:" in text:
            text = text.split("Response:", 1)[-1]

        text = text.strip()
        print(f"[LLM] ✅ Generated {len(text)} chars")
        return text

    except Exception as e:
        print(f"[LLM] ⚠️ Generation failed: {e}")
        return "I'm sorry, I encountered an error while generating the response."
