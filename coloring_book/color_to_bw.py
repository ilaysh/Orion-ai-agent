#!/usr/bin/env python3
"""
ORION COLOR TO B&W  |  Paw Patrol Coloring Page Generator
High-quality image-to-line-art conversion for printable coloring pages.
Uses ControlNet-Aux detectors with SDXL optional refinement.
"""

from controlnet_aux import HEDdetector
import sys
import os
import time
import traceback
import platform
import torch
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from diffusers.utils import load_image
from controlnet_aux import LineartDetector, PidiNetDetector
from controlnet_aux import HEDdetector

# ----------------------------------------------------


def print_header():
    print("===============================================")
    print("🎨 ORION COLOR TO B&W - Coloring Page Generator")
    print("===============================================")


def print_versions():
    import diffusers
    import transformers
    print("-----------------------------------------------")
    print(f"🐍 Python:       {platform.python_version()}")
    print(f"🧩 Torch:        {torch.__version__}")
    print(f"⚙️ CUDA:         {torch.version.cuda}")
    print(f"💻 Device:       {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"🧠 Diffusers:    {diffusers.__version__}")
    print(f"🗣 Transformers: {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"🎮 GPU:          {torch.cuda.get_device_name(0)}")
    print("-----------------------------------------------")


# ----------------------------------------------------


def extract_edges_high_quality(input_path, output_path):
    print("🧠 Running high-quality HEDdetector (cartoon-safe)...")
    image = Image.open(input_path).convert("RGB")

    try:
        detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        edges = detector(
            image,
            detect_resolution=2048,
            image_resolution=2048
        )
        edges.save(output_path)
        print(f"✅ HED lineart saved to: {output_path}")
        return True
    except Exception as e:
        print("⚠️ HEDdetector failed:", e)
        return False

# ----------------------------------------------------


def extract_edges_fallback_pidinet(input_path, output_path):
    print("🧩 Falling back to PidiNetDetector (precise outlines)...")
    image = Image.open(input_path).convert("RGB")
    try:
        pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        edges = pidinet(
            image,
            safe=True,
            detect_resolution=2048,
            image_resolution=2048
        )
        edges.save(output_path)
        print(f"✅ PidiNet fallback saved to: {output_path}")
        return True
    except Exception as e:
        print("❌ PidiNet also failed:", e)
        return False

# ----------------------------------------------------


def refine_with_sdxl(input_path, output_path, device, torch_dtype):
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_model = "ShermanG/ControlNet-Standard-Lineart-for-SDXL"
    sd15_model = "lllyasviel/control_v11p_sd15_lineart"

    print("🧠 Loading ControlNet + base model for refinement...")
    try:
        controlnet = ControlNetModel.from_pretrained(
            sdxl_model, torch_dtype=torch_dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
        ).to(device)
        print("✅ SDXL ControlNet ready.")
    except Exception as e:
        print("⚠️ SDXL load failed, falling back to SD1.5:", e)
        controlnet = ControlNetModel.from_pretrained(
            sd15_model, torch_dtype=torch_dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch_dtype,
        ).to(device)

    control_image = load_image(input_path)
    result = pipe(
        prompt="clean black and white line art coloring page, simple outlines, no shading, no background",
        image=control_image,
        control_image=control_image,
        num_inference_steps=30,
        controlnet_conditioning_scale=1.0,
        guidance_scale=8.0,
    )
    result.images[0].save(output_path)
    print(f"✅ SDXL refined lineart saved to: {output_path}")
    return True

# ----------------------------------------------------


def thicken_lines(path):
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thick = cv2.dilate(img_gray, kernel, iterations=1)
    thick_path = os.path.splitext(path)[0] + "_thick.png"
    cv2.imwrite(thick_path, thick)
    print(f"🖋️ Thickened version saved to: {thick_path}")
    return thick_path

# ----------------------------------------------------


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python color_to_bw.py <input_image> <output_image> [--refine] [--show]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    refine = "--refine" in sys.argv
    show = "--show" in sys.argv
    start_time = time.time()

    print_header()
    print(f"📂 Input:  {input_path}")
    print(f"💾 Output: {output_path}")
    print_versions()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    success = extract_edges_high_quality(input_path, output_path)
    if not success:
        success = extract_edges_fallback_pidinet(input_path, output_path)

    if refine:
        print("✨ Refinement mode ON (SDXL ControlNet)...")
        try:
            refine_with_sdxl(input_path, output_path, device, torch_dtype)
        except Exception as e:
            print("⚠️ Refinement failed:", e)

    if success:
        try:
            thick_path = thicken_lines(output_path)
            clean_path = clean_to_coloring_style(
                thick_path, thick_path.replace("_thick", "_clean"))

        except Exception as e:
            print("⚠️ Thickening failed:", e)

    print("-----------------------------------------------")
    print(f"🕓 Total runtime: {time.time()-start_time:.2f} seconds")
    print("✅ Done.")
    print("===============================================")

    if show:
        Image.open(output_path).show()


def clean_to_coloring_style(input_path, output_path):
    """Turn grayscale edge map into pure black/white printable style."""
    import cv2
    import numpy as np

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # --- Invert so lines are dark on white ---
    img_inv = cv2.bitwise_not(img)

    # --- Normalize and auto-contrast ---
    img_eq = cv2.equalizeHist(img_inv)

    # --- Adaptive threshold to remove gray noise ---
    bw = cv2.adaptiveThreshold(
        img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 10
    )

    # --- Morphological cleanups (remove dots / fill gaps) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- Ensure white background ---
    white_bg = np.full_like(bw, 255)
    white_bg[bw == 0] = 0

    cv2.imwrite(output_path, white_bg)
    print(f"🧼 Cleaned coloring-book version saved to: {output_path}")
    return output_path


# ----------------------------------------------------
if __name__ == "__main__":
    main()
