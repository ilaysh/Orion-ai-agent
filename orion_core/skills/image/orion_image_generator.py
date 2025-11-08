#!/usr/bin/env python3
"""
Orion Image Generator — Text2Img + Img2Img with automatic refinement
"""

import os, argparse, random, gc, shutil
from datetime import datetime
from typing import Optional
from PIL import Image, ImageOps
from orion_core.skills.image.vision_captioner import VisionCaptioner
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
)

REALVIS_XL   = "SG161222/RealVisXL_V4.0"
SDXL_BASE    = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
PICTURES_DIR = os.path.expanduser("~/Pictures")


# ─────────────────────────────── PROMPTS ───────────────────────────────
def build_prompts(subject: str, style: str = "realistic"):
    if style == "realistic":
        pos = (
            f"{subject}, ultra realistic portrait, cinematic lighting, 85mm lens, "
            "natural skin texture, realistic hair, detailed eyes, lifelike lighting, masterpiece"
        )
        neg = (
            "anime, cartoon, cgi, plastic, overexposed, lowres, deformed, doll, extra fingers, "
            "watermark, text, logo, nsfw"
        )
    elif style == "semi":
        pos = (
            f"{subject}, semi-realistic anime-inspired style, expressive face, soft lighting, "
            "vivid colors, detailed hair and eyes, natural proportions, high quality"
        )
        neg = "lowres, bad anatomy, blurry, distorted, dull colors, watermark, text, logo, overexposed"
    else:
        pos, neg = subject, ""
    return pos, neg


# ─────────────────────────────── UTILITIES ───────────────────────────────
def _enable_memory_savers(pipe):
    try:
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    except Exception:
        pass


def _prep_device_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, dtype


def _resize_to(image: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), (width, height), Image.LANCZOS)


# ─────────────────────────────── GENERATE ───────────────────────────────
def generate(
    subject: str,
    seed: Optional[int],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    use_realvis: bool,
    use_refiner: bool,
    from_img: Optional[str],
    strength: float,
    style: str = "realistic",
) -> Image.Image:
    device, dtype = _prep_device_dtype()
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    g = torch.Generator(device=device).manual_seed(seed)
    pos, neg = build_prompts(subject, style=style)

    # ──────────────── IMG2IMG ────────────────
    if from_img is not None:
        if not os.path.isabs(from_img):
            from_img = os.path.join(PICTURES_DIR, from_img)
        if not os.path.exists(from_img):
            raise FileNotFoundError(f"Reference image not found: {from_img}")
        init = Image.open(from_img)
        init = _resize_to(init, width, height)
        model_id = REALVIS_XL if use_realvis else SDXL_BASE

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)
        _enable_memory_savers(pipe)

        image = pipe(
            prompt=pos, negative_prompt=neg,
            image=init, strength=strength,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=g,
        ).images[0]

        if use_refiner:
            refiner = AutoPipelineForImage2Image.from_pretrained(
                SDXL_REFINER, torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
            ).to(device)
            _enable_memory_savers(refiner)
            image = refiner(
                prompt=pos, negative_prompt=neg,
                image=image,
                num_inference_steps=max(15, int(steps * 0.25)),
                guidance_scale=max(5.5, guidance - 0.5),
                generator=g,
            ).images[0]

        print(f"[Gen:img2img] Seed {seed} | Steps {steps} | CFG {guidance} | Strength {strength}")
        gc.collect(); torch.cuda.empty_cache()
        return ImageOps.exif_transpose(image).convert("RGB")

    # ──────────────── TEXT2IMG ────────────────
    model_id = REALVIS_XL if use_realvis else SDXL_BASE
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device); _enable_memory_savers(pipe)
    image = pipe(
        prompt=pos, negative_prompt=neg,
        num_inference_steps=steps, guidance_scale=guidance,
        width=width, height=height,
        generator=g,
    ).images[0]

    if use_refiner:
        refiner = AutoPipelineForImage2Image.from_pretrained(
            SDXL_REFINER, torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)
        _enable_memory_savers(refiner)
        image = refiner(
            prompt=pos, negative_prompt=neg,
            image=image,
            num_inference_steps=max(15, int(steps * 0.25)),
            guidance_scale=max(5.5, guidance - 0.5),
            generator=g,
        ).images[0]

    print(f"[Gen:txt2img] Seed {seed} | Steps {steps} | CFG {guidance}")
    gc.collect(); torch.cuda.empty_cache()
    return ImageOps.exif_transpose(image).convert("RGB")


# ─────────────────────────────── MAIN ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Orion Image Generator — text2img + img2img")
    ap.add_argument("subject", help="Prompt text")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--model", choices=["realvis", "sdxl"], default="realvis")
    ap.add_argument("--use-refiner", action="store_true")
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip automatic refinement pass")
    ap.add_argument("--from-img", type=str, default=None,
                    help="Reference image (filename resolves in ~/Pictures)")
    ap.add_argument("--save-dir", default=os.path.join(PICTURES_DIR, "orion_outputs"),
                    help="Directory to save generated images")
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--refine-strength", type=float, default=0.45)
    ap.add_argument("--refine-steps", type=int, default=40)
    ap.add_argument("--style", choices=["realistic", "semi"], default="realistic")
    ap.add_argument("--make-both", action="store_true",
                    help="Generate both realistic and semi-realistic versions")
    ap.add_argument("--no-auto-prompt", action="store_true",
                    help="Disable automatic prompt generation from image reference.")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    beta_dir = os.path.join(args.save_dir, "beta")
    os.makedirs(beta_dir, exist_ok=True)

    def do_generate(style: str):
        print(f"\n[Generate] Style = {style}")
        
        if args.from_img and not args.no_auto_prompt:
            try:
                cap = VisionCaptioner()
                desc = cap.describe(args.from_img)
                print(f"[AutoPrompt] Extracted description: {desc}")
                subject = f"{desc}, {args.subject}"
            except Exception as e:
                print(f"[AutoPrompt] Failed to generate image description: {e}")
                subject = args.subject
        else:
            subject = args.subject
        
        
        image = generate(
            subject=subject,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
            use_realvis=(args.model == "realvis"),
            use_refiner=args.use_refiner,
            from_img=args.from_img,
            strength=args.strength,
            style=style,
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "_".join(args.subject.lower().split())[:60]
        out_name = f"{stamp}_{safe}_{style}.png" if args.make_both else f"{stamp}_{safe}.png"
        out_path = os.path.join(args.save_dir, out_name)
        image.save(out_path)
        print(f"[Save] → {out_path}")

        if not args.no_refine:
            # auto-refine second pass
            print("[Refine] Running automatic detail pass...")
            refined = generate(
                subject=args.subject,
                seed=args.seed,
                width=args.width,
                height=args.height,
                steps=args.refine_steps,
                guidance=max(5.5, args.guidance - 0.5),
                use_realvis=(args.model == "realvis"),
                use_refiner=True,
                from_img=out_path,
                strength=args.refine_strength,
                style=style,
            )
            # move original
            beta_copy = os.path.join(beta_dir, os.path.basename(out_path))
            shutil.move(out_path, beta_copy)
            refined.save(out_path)
            print(f"[Refined] Overwrote output, base moved to {beta_copy}")
        return out_path

    if args.make_both:
        for style in ["realistic", "semi"]:
            do_generate(style)
    else:
        do_generate(args.style)


if __name__ == "__main__":
    main()
