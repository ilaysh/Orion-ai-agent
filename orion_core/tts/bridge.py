# orion_core/tts/bridge.py
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/m2m100_418M"

_tok = None
_model = None

def _load():
    global _tok, _model
    if _model is not None:
        return _tok, _model
    _tok = M2M100Tokenizer.from_pretrained(MODEL_ID)
    dtype = torch.float16 if DEVICE == "cuda" else None
    _model = M2M100ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype).to(DEVICE)
    _model.eval()
    return _tok, _model

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    tok, model = _load()
    tok.src_lang = src_lang
    enc = tok(text, return_tensors="pt").to(DEVICE)
    gen = model.generate(
        **enc,
        forced_bos_token_id=tok.get_lang_id(tgt_lang),
        max_new_tokens=256,
        num_beams=4,
    )
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def _looks_hebrew(s: str) -> bool:
    return any('\u0590' <= ch <= '\u05FF' for ch in s)

def to_model_input(user_text: str):
    """Return (english_text_for_model, user_lang_hint)."""
    if _looks_hebrew(user_text):
        return translate(user_text, "he", "en"), "he"
    return user_text, "en"

def from_model_output(model_reply_en: str, user_lang_hint: str):
    """Translate back only if user started in Hebrew."""
    if user_lang_hint == "he":
        return translate(model_reply_en, "en", "he")
    return model_reply_en
