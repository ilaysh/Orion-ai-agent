# orion_core/wakeword.py
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import torch
import torchaudio


SAMPLE_RATE = 16_000
# 30 ms chunk keeps latency low but is large enough for stable RMS
CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
# About ~1.0s rolling window → ~50 time frames with 20 ms hop
ROLL_SECONDS = 1.0
ROLL_SAMPLES = int(SAMPLE_RATE * ROLL_SECONDS)

# MFCC transform to match training (n_mfcc=40, hop ~10 ms – 160 samples)
_MFCC = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=40,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
)


def _mfcc_1x1x40x48(int16_wav: np.ndarray) -> np.ndarray:
    """
    Convert mono int16 PCM -> float32 MFCC (1,1,40,48) with center-crop/pad to T=48.
    """
    wav = torch.from_numpy(int16_wav.astype(np.float32) / 32768.0).unsqueeze(0)  # [1, T]
    mfcc = _MFCC(wav)  # [1, 40, T']
    mfcc = torch.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)

    T = mfcc.shape[-1]
    target_T = 48
    if T < target_T:
        pad = target_T - T
        left = pad // 2
        right = pad - left
        mfcc = torch.nn.functional.pad(mfcc, (left, right))
    elif T > target_T:
        # center-crop
        start = (T - target_T) // 2
        mfcc = mfcc[:, :, start:start + target_T]

    # to (1,1,40,48)
    mfcc = mfcc.unsqueeze(1).contiguous().numpy().astype(np.float32)
    return mfcc


class WakeMonitor(threading.Thread):
    """
    Background mic listener that continuously evaluates the wake-word ONNX model
    on a rolling 1s buffer. Calls the given callback once confidence crosses
    `threshold` for `consecutive` frames. Runs entirely outside the VAD/STT path.
    """

    def __init__(
        self,
        model_path: Path | str,
        on_detect: Callable[[], None],
        threshold: float = 0.75,
        consecutive: int = 2,
        provider: Optional[str] = None,  # e.g. "CUDAExecutionProvider"
        device_index: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(daemon=True)
        self.on_detect = on_detect
        self.threshold = threshold
        self.consecutive = consecutive
        self.verbose = verbose
        self._stop = threading.Event()

        opts = ort.SessionOptions()
        providers = [provider] if provider else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        print(f"✅ Wake word model loaded from {model_path}")

        # rolling Int16 buffer
        self._buf = deque(maxlen=ROLL_SAMPLES)
        self._hits = 0

        # mic stream (16 kHz mono)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            device=device_index,
            callback=self._on_audio,
        )

    # --------------- threading -------------

    def stop(self):
        self._stop.set()

    def run(self):
        self._stream.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.08)  # ~12.5 Hz inference
                self._infer_once()
        finally:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    # --------------- audio + inference -------------

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            # status contains overrun/underrun hints; not fatal
            pass
        self._buf.extend(indata[:, 0].copy())

    def _infer_once(self):
        if len(self._buf) < ROLL_SAMPLES // 2:
            return
        arr = np.frombuffer(np.array(self._buf, dtype=np.int16).tobytes(), dtype=np.int16)

        # shape (1,1,40,48)
        mfcc = _mfcc_1x1x40x48(arr)
        pred = self.session.run(None, {"input": mfcc})[0]  # [1,1]
        conf = float(pred.reshape(-1)[0])

        if self.verbose:
            print(f"[Wake] conf={conf:.3f}")

        if conf >= self.threshold:
            self._hits += 1
            if self._hits >= self.consecutive:
                self._hits = 0
                if self.verbose:
                    print("✅ Wake word fired!")
                self.on_detect()
        else:
            self._hits = 0
