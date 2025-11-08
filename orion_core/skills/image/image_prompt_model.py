# orion_prompt_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"  # light, local model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto")


def generate_coloring_prompt(subject: str, child_name: str = "Leroy") -> str:
    system = "You write short, descriptive prompts for coloring-book images (black and white line art)."
    user = f"Create a Stable Diffusion prompt for a coloring page for {child_name} about {subject}."
    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
            {"role": "user", "content": user}],
        return_tensors="pt"
    ).to(model.device)
    output = model.generate(inputs, max_new_tokens=150)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split("Assistant:")[-1].strip()
