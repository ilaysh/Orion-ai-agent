#!/usr/bin/env python3
"""
Identity-preserving generation using IP-Adapter (SDXL/RealVisXL).
Simplified: minimal arguments, smart defaults.
"""

import os
import argparse
from typing import Optional
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

try:
    from ip_adapter import IPAdapterXL
except Exception as e:
    raise RuntimeError(
        "❌ IP-Adapter package missing. Install:\n"
        "  pip install git+https://github.com/tencent-ailab/IP-Adapter.git"
    ) from e


# ─────────────────────────── CONFIG DEFAULTS ───────────────────────────
DEFAULT_MODEL = "SG161222/RealVisXL_V4.0"     # SDXL-based RealVis
DEFAULT_WEIGHTS = os.path.expanduser(
    "~/projects/orion-v2/models/ip_adapter/ip-adapter_sdxl.bin"
)
PICTURES_DIR = os.path.expanduser("~/Pictures")


def _enable_memory_savers(pipe):
    try:
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    except Exception:
        pass


def _resolve_in_pictures(name: str) -> str:
    path = os.path.expanduser(name)
    if not os.path.isabs(path):
        path = os.path.join(PICTURES_DIR, name)
    return path


# ─────────────────────────── CORE FUNCTION ───────────────────────────
def generate_with_ip_adapter(
    prompt: str,
    ref_image: str,
    weights_path: str = DEFAULT_WEIGHTS,
    model_id: str = DEFAULT_MODEL,
    strength: float = 0.4,
    guidance: float = 7.5,
    steps: int = 48,
    width: int = 896,
    height: int = 1152,
    out_path: Optional[str] = None,
):
    """Generate a realistic/identity-preserving image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    ref_path = _resolve_in_pictures(ref_image)
    weights_path = os.path.expanduser(weights_path)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"IP-Adapter weights not found: {weights_path}")

    print(f"[Load] Model: {model_id}")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    _enable_memory_savers(pipe)

    print("[Load] IP-Adapter")
# The IP-Adapter SDXL expects an image encoder checkpoint + adapter weights
    image_encoder = os.path.expanduser(
        "~/projects/orion-v2/models/ip_adapter/image_encoder"
    )
    ip_model = IPAdapterXL(
        pipe,
        image_encoder,     # path to the image encoder directory
        weights_path,      # ip-adapter_sdxl.bin
        device,
        num_tokens=4
    )
    ref = Image.open(ref_path).convert("RGB")

    print(f"[Generate] {prompt}")
    image = ip_model.generate(
        prompt=prompt,
        image=ref,
        scale=1.0,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
    )

    if out_path is None:
        out_dir = os.path.join(PICTURES_DIR, "orion_outputs", "ip_adapter")
        os.makedirs(out_dir, exist_ok=True)
        safe_name = "_".join(prompt.lower().split())[:60]
        out_path = os.path.join(out_dir, f"{safe_name}.png")

    image.save(out_path)
    print(f"✅ Saved → {out_path}")
    return image


# ─────────────────────────── CLI ENTRY ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Identity-preserving generator (IP-Adapter).")
    ap.add_argument("prompt", help="Prompt text for pose / scene.")
    ap.add_argument("--ref-image", required=True, help="Reference image filename or path.")
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    generate_with_ip_adapter(
        prompt=args.prompt,
        ref_image=args.ref_image,
        strength=args.strength,
        guidance=args.guidance,
        steps=args.steps,
        width=args.width,
        height=args.height,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
