import torch, numpy as np

class SileroVAD:
    """Wrapper for Silero VAD (torch.hub)."""
    def __init__(self, sr=16000, min_silence_ms=400):
        self.sr = sr
        self.model, utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', force_reload=False
        )
        (self.get_speech_ts, _, _, _, _) = utils
        self.model.eval()
        print("[VAD] Ready.")

    def segments(self, wav: np.ndarray):
        """Return speech segment list [{start,end}]"""
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return self.get_speech_ts(wav.squeeze(0), self.model, sampling_rate=self.sr)
