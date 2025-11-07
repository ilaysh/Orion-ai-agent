#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Orion Coloring Book Generator (v4.0)
Creates full-color render for use in coloring pages (via SDXL or LoRA).
"""

import torch
import argparse
import subprocess
import os
from diffusers import StableDiffusionXLPipeline
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_coloring_image(prompt: str, out_path: Path, steps=60, guidance=7.5,
                            width=1024, height=1024, negative_prompt="",
                            base_model="stabilityai/stable-diffusion-xl-base-1.0",
                            lora=None, lora_scale=0.8, seed=42):
    print("======================================================================")
    print(f"🐾 Generating coloring page for prompt: {prompt}")
    print("======================================================================")

    print(f"[Torch] Device: {device}")
    print(f"[Model] Loading base model: {base_model}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    if lora and Path(lora).exists():
        print(f"[LoRA] Loading character LoRA: {lora}")
        pipe.load_lora_weights(lora)
        pipe.fuse_lora(lora_scale=lora_scale)
        print("✅ Character LoRA loaded successfully")

    print(f"[Prompt] {prompt}")
    print(f"[Generate] Steps={steps} CFG={guidance} Size={width}x{height}")

    # Strong negative prompt to prevent duplicates
    negative = negative_prompt
    if not negative:
        negative = (
            "multiple characters, two dogs, duplicate, double, twins, "
            "group, crowd, extra limbs, deformed, merged bodies, "
            "collage, split image, text, watermark"
        )

    image = pipe(prompt=prompt, num_inference_steps=steps,
                 guidance_scale=guidance,
                 generator=torch.manual_seed(42),
                 negative_prompt=negative,
                 width=width, height=height).images[0]
    image.save(out_path)
    print(f"✅ Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--show", default="general")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora", help="Optional LoRA path")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--negative", default="")
    args = parser.parse_args()

    out_dir = Path.home() / "Pictures/coloring_books" / \
        f"{args.show.lower()}_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.png"

    generate_coloring_image(args.prompt, out_path, steps=args.steps, guidance=args.guidance,
                            width=args.width, height=args.height,
                            negative_prompt=args.negative,
                            base_model=args.base_model, lora=args.lora, lora_scale=args.lora_scale)

    print(f"[OUTPUT_IMAGE]{out_path}")
    pdf_path = out_dir.parent / f"{args.show.lower()}.pdf"
    print(f"[OUTPUT_PDF]{pdf_path}")

    if args.preview:
        # Properly detach the preview process to avoid blocking
        subprocess.Popen(
            ["xdg-open", str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )


if __name__ == "__main__":
    main()
