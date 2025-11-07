#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont

CARDS_DIR = os.path.expanduser("~/Pictures/coloring_books/cards")
OUT_SHEET = os.path.join(CARDS_DIR, "character_cards_sheet.png")


def make_card_sheet(columns=4, margin=20, bg_color="white"):
    imgs = []
    for f in sorted(os.listdir(CARDS_DIR)):
        if f.lower().endswith((".png", ".jpg")) and not f.startswith("sheet"):
            path = os.path.join(CARDS_DIR, f)
            imgs.append((Image.open(path).convert(
                "RGB"), os.path.splitext(f)[0]))
    if not imgs:
        print("❌ No images found in cards folder.")
        return

    w, h = imgs[0][0].size
    rows = (len(imgs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * w + (columns+1)*margin,
                              rows * h + (rows+1)*margin), bg_color)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
    except:
        font = None

    for i, (img, name) in enumerate(imgs):
        x = margin + (i % columns) * (w + margin)
        y = margin + (i // columns) * (h + margin)
        sheet.paste(img, (x, y))
        d = ImageDraw.Draw(sheet)
        label = name.replace("_", " ").title()
        d.text((x + 20, y + 20), label, fill="black", font=font)

    sheet.save(OUT_SHEET)
    print(f"✅ Saved card sheet: {OUT_SHEET}")


if __name__ == "__main__":
    make_card_sheet(columns=4)
