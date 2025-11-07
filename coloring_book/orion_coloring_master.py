#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐾 Orion Coloring Master — Paw Patrol Edition
Full-color SDXL generation with LoRA auto-load and single-subject prompt control.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from PIL import Image


ROOT = Path(__file__).resolve().parent
COLORING_SCRIPT = ROOT / "coloring_book_generator.py"
KNOWLEDGE_DIR = ROOT / "coloring_knowledge"
MODELS_DIR = ROOT / "models/lora"
PICTURES_DIR = Path.home() / "Pictures/coloring_books"
PICTURES_DIR.mkdir(parents=True, exist_ok=True)


def load_character(character: str, show: str):
    """Return dict with {description, lora_id} using new characters[] format."""
    json_path = KNOWLEDGE_DIR / f"{show.lower()}.json"
    if not json_path.exists():
        return {"description": f"{character} from {show}", "lora_id": None, "base_model": "stabilityai/sdxl-turbo"}

    data = json.loads(json_path.read_text())

    # pull top-level settings
    base_model = data.get("base_model", "stabilityai/sdxl-turbo")
    lora_scale = data.get("lora_scale", 1.0)

    for entry in data.get("characters", []):
        if entry["name"].lower() == character.lower():
            return {
                "description": entry.get("description", f"{character} from {show}"),
                "lora_id": entry.get("lora_id"),
                "base_model": base_model,
                "lora_scale": lora_scale,
            }

    return {"description": f"{character} from {show}", "lora_id": None, "base_model": base_model, "lora_scale": lora_scale}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("character")
    parser.add_argument("--show", default="PawPatrol")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    char = args.character.strip()
    show = args.show.strip()

    print("=" * 70)
    print("🐾 ORION COLORING MASTER - Full Color + Single Character Mode")
    print("=" * 70)
    print(f"Character: {char}")
    print(f"Show: {show}")
    print("=" * 70)

    # Step 1️⃣ — Load metadata
    info = load_character(char, show)
    desc = info["description"]
    lora_id = info["lora_id"]
    base_model = info["base_model"]
    lora_scale = str(info.get("lora_scale", 0.9))

    # Step 2️⃣ — Build improved prompt (simpler, more direct)
 # 🎨 Step 2️⃣ — Build single-character prompt
    full_prompt = (
        f"{char} from {show}, solo character portrait, full body, "
        f"standing upright, centered, high-quality 3D render, "
        f"bright colorful background, Pixar-style lighting, "
        f"official show design, kid-friendly cartoon, consistent style"
    )
    negative = (
        "multiple characters, two dogs, duplicate, double, twins, "
        "group, crowd, extra limbs, deformed, merged bodies, "
        "collage, split image, text, watermark, background objects"
    )

    print(f"📝 Prompt: {full_prompt}")

    # Detect LoRA file if available
    lora_path = None
    if lora_id:
        if str(lora_id).startswith("civitai:"):
            lora_path = lora_id  # remote reference (for completeness)
            print(f"🌐 Using online LoRA reference: {lora_id}")
        else:
            candidate = MODELS_DIR / lora_id
            if candidate.exists():
                lora_path = candidate
                print(f"🎨 Found local LoRA: {candidate.name}")
            else:
                print(f"[Warn] LoRA listed but not found: {candidate.name}")

    # Step 3️⃣ — Build command (FIXED: prompt as single argument)
    cmd = [
        "python", str(COLORING_SCRIPT),
        full_prompt,  # This is now properly passed as a single argument
        "--show", show,
        "--steps", "60",
        "--guidance", "7.5",
        "--width", "1024",
        "--height", "1024",
        "--negative", negative,
        "--base-model", base_model,
    ]

    if lora_path:
        cmd += ["--lora", str(lora_path), "--lora-scale", lora_scale]

    # Don't pass --preview to subprocess, we'll handle it ourselves
    # if args.preview:
    #     cmd.append("--preview")

    print(f"[Run] Executing coloring_book_generator.py...")
    print()

    # Stream output in real-time instead of buffering
    result = subprocess.run(cmd, text=True)
    print()

    if result.returncode != 0:
        print(f"❌ Generation failed with return code {result.returncode}")
        return

    # Step 4️⃣ — Detect output image (read from expected location)
    out_dir = PICTURES_DIR / f"{show.lower()}_pages"
    latest_result = out_dir / "result.png"

    if not latest_result.exists():
        print("❌ No output image detected.")
        print(f"Expected at: {latest_result}")
        print(f"Return code: {result.returncode}")
        return

    print(f"✅ Using colored render: {latest_result}")

    # Step 5️⃣ — Append to PDF
    pdf_path = PICTURES_DIR / f"{show.lower()}_coloring_book.pdf"
    img = Image.open(latest_result).convert("RGB")

    if pdf_path.exists():
        from PyPDF2 import PdfMerger
        temp_pdf = latest_result.with_suffix(".pdf")
        img.save(temp_pdf, "PDF")
        merger = PdfMerger()
        merger.append(str(pdf_path))
        merger.append(str(temp_pdf))
        merger.write(str(pdf_path))
        merger.close()
        temp_pdf.unlink(missing_ok=True)
    else:
        img.save(pdf_path, "PDF")

    # Step 6️⃣ — Preview
    if args.preview:
        try:
            subprocess.Popen(["xdg-open", str(latest_result)])
        except Exception:
            pass

    print("\n✅ Done! 🎨 Full-color single-character page generated.")
    print(f"🖼  Image: {latest_result}")
    print(f"📄  PDF:   {pdf_path}")


if __name__ == "__main__":
    main()
