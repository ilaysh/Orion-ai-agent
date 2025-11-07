#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎭 Orion Coloring Prompt Helper (v4.0)
Generates descriptive prompts for SDXL coloring page creation.
"""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "coloring_knowledge"
KNOWLEDGE_DIR.mkdir(exist_ok=True)
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

print(f"[Model] Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto")


def generate_prompt(character: str, show: str) -> str:
    system = (
        "You are an expert at writing short, vivid Stable Diffusion prompts for children's coloring pages. "
        "Always describe pose, mood, and activity naturally."
    )
    user = f"Write a prompt for a coloring page of {character} from {show}."
    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
            {"role": "user", "content": user}],
        return_tensors="pt"
    ).to(model.device)
    output = model.generate(inputs, max_new_tokens=150)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split("Assistant:")[-1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("character")
    parser.add_argument("--show", default="general")
    args = parser.parse_args()

    char, show = args.character, args.show
    db_path = KNOWLEDGE_DIR / f"{show.lower()}.json"
    db = json.loads(db_path.read_text()) if db_path.exists() else {}

    if char.lower() not in db:
        print(f"[Info] Generating new prompt for {char} ({show})...")
        prompt = generate_prompt(char, show)
        db[char.lower()] = prompt
        db_path.write_text(json.dumps(db, indent=2))
        print(f"[Saved] → {db_path}")
    else:
        prompt = db[char.lower()]
        print(f"[Cache] Found existing prompt for {char}")

    print("\n[Prompt]")
    print(prompt)


if __name__ == "__main__":
    main()
