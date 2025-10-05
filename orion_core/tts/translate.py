from transformers import MarianMTModel, MarianTokenizer
import torch

def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

def translate(text, direction="he-en"):
    # If text is English already → skip translation
    if direction == "he-en" and all(not ("\u0590" <= ch <= "\u05EA") for ch in text):
        return text, "en (skipped)"

    model_name = "Helsinki-NLP/opus-mt-he-en"
    tokenizer, model, device = load_model(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)

    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return result, direction

if __name__ == "__main__":
    while True:
        text = input("Enter text (or 'quit' to exit): ")
        if text.lower() == "quit":
            break
        translated, direction = translate(text, "he-en")
        print(f"[{direction}] {translated}")
