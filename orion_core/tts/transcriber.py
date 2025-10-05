# transcriber.py
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class Transcriber:
    def __init__(self, model_size="medium"):
        """
        Uses OpenAI Whisper via Hugging Face transformers.
        model_size: "tiny", "base", "small", "medium", "large-v2"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = f"openai/whisper-{model_size}"

        # Load processor + model
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)

        # Whisper expects 16 kHz mono
        self.sample_rate = 16000

    def transcribe(self, wav_input, language='en'):
        """
        Accepts a NumPy float32 waveform in [-1, 1] or a path to a wav file.
        Returns transcribed text.
        
        Args:
            wav_input: NumPy array or path to wav file
            language: Force specific language (e.g., 'en' for English)
        """
        # Load waveform
        if isinstance(wav_input, str):
            import soundfile as sf
            wav, sr = sf.read(wav_input)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)  # mix stereo → mono
            wav = wav.astype(np.float32)
            if sr != self.sample_rate:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        elif isinstance(wav_input, np.ndarray):
            wav = wav_input.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(wav_input)}")

        # Normalize
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

        # Tokenize
        inputs = self.processor(
            wav,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)

        # Force language token if specified
        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, 
                task="transcribe"
            )

        # Forward + generate
        with torch.no_grad():
            predicted_ids = self.model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids
            )

        # Decode
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()