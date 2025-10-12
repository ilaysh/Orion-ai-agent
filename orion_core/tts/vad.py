import torch, numpy as np

class SileroVAD:
    def __init__(self, sr=16000,
                 threshold=0.28, 
                 min_speech_ms=120,
                 min_silence_ms=750):
        self.sr = sr
        self.model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
        (self.get_speech_ts, _, _, _, _) = utils
        self.model.eval()
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

    def is_speech(self, wav: np.ndarray) -> bool:
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        tensor = torch.from_numpy(np.copy(wav)).float()  # writable
        segs = self.get_speech_ts(
            tensor, self.model, sampling_rate=self.sr,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms
        )
        return len(segs) > 0
