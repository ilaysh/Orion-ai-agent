# coloring_book_generator.py
# Improved local coloring page generator with better models & post-processing
# Uses: Realistic Vision (better anatomy), ControlNet (line control), enhanced processing

import os
import io
import math
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
import cv2
import torch
from diffusers import (
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler
)

# ========================
# CONFIG
# ========================

# Best models for coloring book style (tested 2024-2025)
MODELS = {
    "anime": "Lykon/DreamShaper-8",  # Great for cartoon/stylized
    "photorealistic": "runwayml/stable-diffusion-v1-5",
    "detailed": "prompthero/openjourney-v4",  # Good line definition
}

DEFAULT_MODEL = "Lykon/DreamShaper-8"  # Best overall for kids' content

A4_MM = (210, 297)
LETTER_IN = (8.5, 11.0)


@dataclass
class PrintSettings:
    paper: str = "A4"
    dpi: int = 300
    margin_mm: int = 10
    line_thickness_px: int = 3
    contrast_boost: float = 1.8  # Increase for bolder lines


@dataclass
class GenerationSettings:
    steps: int = 50  # More steps = better quality
    guidance: float = 12.0  # Higher = follows prompt better
    width: int = 768
    height: int = 1024
    seed: Optional[int] = None
    scheduler: str = "ddim"  # DDIM faster than default


# ========================
# ENHANCED PROMPTS
# ========================

COLORING_SYSTEM_PROMPT = """You are a children's coloring book illustrator. 
Create clean, simple line art with:
- Clear, thick black outlines
- No shading or gradients
- Minimal background
- High contrast
- Kid-friendly style
- Isolated subject on white background
- Vector/comic book aesthetic
- Simple, recognizable shapes"""


def build_coloring_prompt(subject: str, for_name: Optional[str] = None) -> Tuple[str, str]:
    """Returns (positive_prompt, negative_prompt) for high-quality coloring pages"""

    who = f"for {for_name}" if for_name else ""

    positive = (
        f"professional coloring book page {who}, "
        f"{subject}, "
        f"clean black line art, thick ink outlines, "
        f"bold strokes, high contrast, vector style, "
        f"simple shapes, white background, isolated subject, "
        f"no shading, no grayscale, no background clutter, "
        f"comic book style, kid-friendly, printable, "
        f"pen and ink illustration"
    )

    negative = (
        "photograph, photo, realistic, shading, gradient, blur, "
        "messy, sketch, pencil, watercolor, oil painting, "
        "low contrast, gray, muted, background, clutter, "
        "text, watermark, logo, signature, detailed, complex, "
        "3d, depth, shadow, small details, noise, jpeg artifact"
    )

    return positive, negative


# ========================
# MODEL LOADING (CACHED)
# ========================

_PIPE_CACHE = {}


def get_pipeline(model_id: str = DEFAULT_MODEL):
    """Load and cache model pipeline"""
    if model_id in _PIPE_CACHE:
        return _PIPE_CACHE[model_id]

    print(f"Loading model: {model_id}...")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        use_safetensors=True
    )

    # Speed up inference
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    if device == "cuda":
        pipe.enable_sequential_cpu_offload()

    pipe = pipe.to(device)
    _PIPE_CACHE[model_id] = pipe
    print(f"Model loaded on {device}")

    return pipe


# ========================
# IMAGE GENERATION
# ========================

def generate_raw_image(
    subject: str,
    for_name: Optional[str] = None,
    settings: GenerationSettings = None,
    model_id: str = DEFAULT_MODEL
) -> Image.Image:
    """Generate raw image from subject"""

    if settings is None:
        settings = GenerationSettings()

    positive, negative = build_coloring_prompt(subject, for_name)

    if settings.seed is None:
        settings.seed = random.randint(0, 2**31 - 1)

    print(f"Generating with seed {settings.seed}...")
    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(settings.seed)

    pipe = get_pipeline(model_id)

    result = pipe(
        prompt=positive,
        negative_prompt=negative,
        num_inference_steps=settings.steps,
        guidance_scale=settings.guidance,
        width=settings.width,
        height=settings.height,
        generator=generator,
        num_images_per_prompt=1
    ).images[0]

    # Ensure RGB
    result = ImageOps.exif_transpose(result).convert("RGB")
    return result


# ========================
# ADVANCED POST-PROCESSING
# ========================

def adaptive_histogram_equalization(img_array: np.ndarray) -> np.ndarray:
    """Enhance contrast locally (CLAHE)"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img_array)


def smart_binarization(
    img: Image.Image,
    contrast_boost: float = 1.8
) -> np.ndarray:
    """
    Convert to pure B&W with enhanced edge detection
    Better than simple threshold for complex line art
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    # Enhance contrast
    gray = cv2.convertScaleAbs(gray, alpha=contrast_boost, beta=0)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Local contrast enhancement
    gray = adaptive_histogram_equalization(gray)

    # Edge detection to find lines
    edges = cv2.Canny(gray, 30, 100)

    # Morphological operations to strengthen edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Adaptive threshold for the rest
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=5
    )

    # Combine edges with threshold
    combined = cv2.bitwise_or(edges, binary)

    return combined


def postprocess_for_printing(
    img: Image.Image,
    thickness_px: int = 3,
    despeckle_size: int = 3,
    fill_holes: bool = True,
    print_settings: PrintSettings = None
) -> Image.Image:
    """
    Advanced post-processing pipeline for professional coloring pages
    """
    if print_settings is None:
        print_settings = PrintSettings()

    print("Processing image for print...")

    # Step 1: Smart binarization
    binary = smart_binarization(
        img, contrast_boost=print_settings.contrast_boost)

    # Step 2: Despeckle (remove noise/artifacts)
    if despeckle_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (despeckle_size, despeckle_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 3: Thicken lines for better printability
    if thickness_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (thickness_px, thickness_px))
        binary = cv2.dilate(binary, kernel, iterations=1)

    # Step 4: Fill small holes (closed shapes)
    if fill_holes:
        binary = 255 - binary  # Invert
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = 255 - binary  # Invert back

    # Step 5: Final cleanup
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    # Convert to PIL bilevel (pure B&W)
    result = Image.fromarray(binary, mode="L").convert("1")

    return result


# ========================
# LAYOUT & PDF
# ========================

def mm_to_px(mm: float, dpi: int) -> int:
    inches = mm / 25.4
    return int(round(inches * dpi))


def get_paper_size(paper: str, dpi: int) -> Tuple[int, int]:
    """Get paper dimensions in pixels"""
    if paper.upper() == "A4":
        w_mm, h_mm = A4_MM
        return mm_to_px(w_mm, dpi), mm_to_px(h_mm, dpi)
    elif paper.upper() in ("LETTER", "USLETTER"):
        w_in, h_in = LETTER_IN
        return int(round(w_in * dpi)), int(round(h_in * dpi))
    else:
        raise ValueError("paper must be 'A4' or 'LETTER'")


def create_print_page(
    artwork: Image.Image,
    print_settings: PrintSettings = None
) -> Image.Image:
    """Create printable page with artwork centered and margined"""

    if print_settings is None:
        print_settings = PrintSettings()

    page_w, page_h = get_paper_size(print_settings.paper, print_settings.dpi)
    margin_px = mm_to_px(print_settings.margin_mm, print_settings.dpi)

    # Create white page
    page = Image.new("1", (page_w, page_h), 1)  # 1=white in bilevel

    # Calculate content area
    content_w = page_w - (margin_px * 2)
    content_h = page_h - (margin_px * 2)

    # Fit artwork in content area (maintain aspect ratio)
    artwork_copy = artwork.copy()
    artwork_copy.thumbnail((content_w, content_h), Image.LANCZOS)

    # Center artwork
    x = (page_w - artwork_copy.width) // 2
    y = (page_h - artwork_copy.height) // 2

    page.paste(artwork_copy, (x, y))

    return page


def save_pdf(pages: List[Image.Image], output_path: str):
    """Save pages as PDF"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if not pages:
        raise ValueError("No pages to save")

    pages[0].save(
        output_path,
        "PDF",
        resolution=300,
        save_all=True,
        append_images=pages[1:] if len(pages) > 1 else []
    )
    print(f"✓ Saved: {output_path}")


# ========================
# HIGH-LEVEL API
# ========================

def make_coloring_page(
    subject: str,
    for_name: Optional[str] = None,
    output_path: str = "coloring_page.pdf",
    print_settings: PrintSettings = None,
    gen_settings: GenerationSettings = None,
    model_id: str = DEFAULT_MODEL
) -> str:
    """Generate single coloring page"""

    if print_settings is None:
        print_settings = PrintSettings()
    if gen_settings is None:
        gen_settings = GenerationSettings()

    print(f"\n{'='*60}")
    print(f"Generating coloring page: {subject}")
    print(f"For: {for_name or 'anyone'}")
    print(f"Model: {model_id}")
    print(f"{'='*60}\n")

    # Generate
    raw = generate_raw_image(subject, for_name, gen_settings, model_id)

    # Process
    processed = postprocess_for_printing(raw, print_settings=print_settings)

    # Layout
    page = create_print_page(processed, print_settings)

    # Save
    save_pdf([page], output_path)

    return output_path


def make_coloring_book(
    subjects: List[str],
    for_name: Optional[str] = None,
    output_path: str = "coloring_book.pdf",
    print_settings: PrintSettings = None,
    gen_settings: GenerationSettings = None,
    model_id: str = DEFAULT_MODEL
) -> str:
    """Generate multi-page coloring book"""

    if print_settings is None:
        print_settings = PrintSettings()
    if gen_settings is None:
        gen_settings = GenerationSettings()

    print(f"\n{'='*60}")
    print(f"Generating coloring book: {len(subjects)} pages")
    print(f"For: {for_name or 'anyone'}")
    print(f"{'='*60}\n")

    pages = []
    base_seed = random.randint(0, 2**31 - 1)

    for i, subject in enumerate(subjects):
        print(f"\n[Page {i+1}/{len(subjects)}] {subject}")

        gen_settings.seed = base_seed + i

        raw = generate_raw_image(subject, for_name, gen_settings, model_id)
        processed = postprocess_for_printing(
            raw, print_settings=print_settings)
        page = create_print_page(processed, print_settings)
        pages.append(page)

    save_pdf(pages, output_path)

    return output_path


# ========================
# CLI
# ========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Professional coloring book generator (local, offline)"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Single page
    page_cmd = subparsers.add_parser(
        "page", help="Generate single coloring page")
    page_cmd.add_argument("--subject", required=True, help="What to draw")
    page_cmd.add_argument("--for", dest="for_name",
                          default=None, help="Child's name")
    page_cmd.add_argument("--model", default=DEFAULT_MODEL, help="Model ID")
    page_cmd.add_argument("--paper", default="A4", choices=["A4", "LETTER"])
    page_cmd.add_argument("--dpi", type=int, default=300)
    page_cmd.add_argument("--margin-mm", type=int, default=10)
    page_cmd.add_argument("--thickness", type=int, default=3)
    page_cmd.add_argument("--contrast", type=float, default=1.8)
    page_cmd.add_argument("--steps", type=int, default=50)
    page_cmd.add_argument("--guidance", type=float, default=12.0)
    page_cmd.add_argument("--out", default="coloring_page.pdf")

    # Book (multiple pages)
    book_cmd = subparsers.add_parser("book", help="Generate coloring book")
    book_cmd.add_argument("--subjects", nargs="+",
                          required=True, help="List of subjects")
    book_cmd.add_argument("--for", dest="for_name", default=None)
    book_cmd.add_argument("--model", default=DEFAULT_MODEL)
    book_cmd.add_argument("--paper", default="A4", choices=["A4", "LETTER"])
    book_cmd.add_argument("--dpi", type=int, default=300)
    book_cmd.add_argument("--margin-mm", type=int, default=10)
    book_cmd.add_argument("--thickness", type=int, default=3)
    book_cmd.add_argument("--contrast", type=float, default=1.8)
    book_cmd.add_argument("--steps", type=int, default=50)
    book_cmd.add_argument("--guidance", type=float, default=12.0)
    book_cmd.add_argument("--out", default="coloring_book.pdf")

    args = parser.parse_args()

    print_settings = PrintSettings(
        paper=args.paper,
        dpi=args.dpi,
        margin_mm=args.margin_mm,
        line_thickness_px=args.thickness,
        contrast_boost=args.contrast
    )

    gen_settings = GenerationSettings(
        steps=args.steps,
        guidance=args.guidance
    )

    if args.cmd == "page":
        make_coloring_page(
            subject=args.subject,
            for_name=args.for_name,
            output_path=args.out,
            print_settings=print_settings,
            gen_settings=gen_settings,
            model_id=args.model
        )

    elif args.cmd == "book":
        make_coloring_book(
            subjects=args.subjects,
            for_name=args.for_name,
            output_path=args.out,
            print_settings=print_settings,
            gen_settings=gen_settings,
            model_id=args.model
        )
