# orion_core/skills/vision_captioner.py
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# Try using Florence-2 (more recent) if available; fallback to BLIP
USE_BLIP = True
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    _ = AutoProcessor.from_pretrained("microsoft/Florence-2-large")
    USE_BLIP = False
except Exception:
    pass

class VisionCaptioner:
    def __init__(self):
        if USE_BLIP:
            print("[VisionCaptioner] Using BLIP caption model.")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("[VisionCaptioner] Using Florence-2 model.")
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to("cuda" if torch.cuda.is_available() else "cpu")

    def describe(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")


        image_path = os.path.expanduser(image_path)
        if not os.path.isabs(image_path):
            default_dir = os.path.expanduser("~/Pictures")
            image_path = os.path.join(default_dir, image_path)
            
        image = Image.open(image_path).convert("RGB")

        if USE_BLIP:
            inputs = self.processor(image, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=100)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
        else:
            prompt = "<CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
            generated = self.model.generate(**inputs, max_new_tokens=100)
            caption = self.processor.batch_decode(generated, skip_special_tokens=True)[0]

        return caption.strip()


# Quick CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vision_captioner.py <image_path>")
        exit(0)
    cap = VisionCaptioner()
    desc = cap.describe(sys.argv[1])
    print(f"\n🧠 Image description:\n{desc}\n")
