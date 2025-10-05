import torch
import numpy as np
import collections


class Listener:
    """
    Simplified approach: Use energy + variance for silence detection, VAD only for validation.
    """

    def __init__(self, vad_model=None, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Load Silero VAD (optional - we'll use it lightly)
        if vad_model is None:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
            self.get_speech_timestamps = utils[0]
            self.vad_model = model
        else:
            self.vad_model = vad_model
            from silero_vad import get_speech_timestamps
            self.get_speech_timestamps = get_speech_timestamps

        # Buffers
        self.buffer = bytearray()
        self.max_buffer_seconds = 30
        self.max_buffer_bytes = sample_rate * 2 * self.max_buffer_seconds
        
        # Adaptive thresholds based on observed audio (removed - using fixed thresholds)
        self.recent_energies = collections.deque(maxlen=20)  # Keep for monitoring only
        
        # State
        self.in_speech = False
        self.consecutive_silence = 0
        self.consecutive_speech = 0
        self.min_utterance_samples = int(1.0 * sample_rate)  # Reduced to 1.0 second


    def reset_feed(self):
        """Clear all buffers and reset state."""
        self.buffer.clear()
        self.in_speech = False
        self.consecutive_silence = 0
        self.consecutive_speech = 0

    def _analyze_window(self, wav: np.ndarray):
        """
        Analyze audio window and return metrics.
        Returns: (energy, variance, is_speech_like)
        """
        energy = float(np.sqrt(np.mean(wav**2)))
        variance = float(np.var(wav))
        
        # Track energy history for monitoring only (not used for thresholds)
        self.recent_energies.append(energy)
        
        # Use FIXED thresholds - more predictable than adaptive
        # These work well across different speaking volumes
        SPEECH_ENERGY_MIN = 0.10   # Increased - more aggressive
        SPEECH_VARIANCE_MIN = 0.008  # Increased - more aggressive
        
        is_high_energy = energy > SPEECH_ENERGY_MIN
        is_high_variance = variance > SPEECH_VARIANCE_MIN
        
        # Both conditions must be true
        is_speech_like = is_high_energy and is_high_variance
        
        return energy, variance, is_speech_like

    def calibrate_ambient(self, duration=1.0):
            """
            Calibrate by listening to ambient noise for 'duration' seconds.
            Assumes called at startup when user is silent.
            """
            print("[VAD] Calibrating ambient noise...")
            
            # Use PyAudio or similar to capture ambient audio (since no stream param)
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=1024)
            
            ambient_data = bytearray()
            for _ in range(int(self.sample_rate / 1024 * duration)):
                chunk = stream.read(1024)
                ambient_data.extend(chunk)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            wav_ambient = np.frombuffer(ambient_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Compute ambient metrics
            ambient_energy = float(np.sqrt(np.mean(wav_ambient**2)))
            ambient_variance = float(np.var(wav_ambient))
            
            # Set thresholds as multipliers (tune these factors based on tests)
            self.SPEECH_ENERGY_MIN = max(0.05, ambient_energy * 3.0)  # e.g., 3x ambient
            self.SPEECH_VARIANCE_MIN = max(0.003, ambient_variance * 4.0)  # e.g., 4x ambient
            
            print(f"[VAD] Calibration complete: Energy min={self.SPEECH_ENERGY_MIN:.4f}, Variance min={self.SPEECH_VARIANCE_MIN:.6f}")
            
    def feed(self, audio_chunk: bytes):
        """
        Feed audio chunks. Uses simple energy/variance detection.
        """
        self.buffer.extend(audio_chunk)
        
        # Prevent buffer overflow
        if len(self.buffer) > self.max_buffer_bytes:
            print("[VAD] Buffer overflow - forcing return")
            result = bytes(self.buffer)
            self.reset_feed()
            return result
        
        # Need minimum buffer to analyze
        min_samples = self.sample_rate // 2
        if len(self.buffer) < min_samples * 2:
            return None
        
        # Analyze only recent 0.5 second window
        window_bytes = self.sample_rate * 2 * 1  # 0.5 seconds
        if len(self.buffer) > window_bytes:
            analysis_bytes = self.buffer[-window_bytes:]
        else:
            analysis_bytes = bytes(self.buffer)
        
        wav_window = np.frombuffer(analysis_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Get metrics
        energy, variance, is_speech_like = self._analyze_window(wav_window)
        
        print(f"[SIMPLE] Buffer: {len(self.buffer):6d} bytes | "
              f"Energy: {energy:.4f} | Variance: {variance:.6f} | "
              f"Speech-like: {is_speech_like}")
        
        # State machine based on energy/variance
        if is_speech_like:
            # Looks like speech
            
            # CRITICAL: Check BEFORE resetting consecutive_silence
            # If we had 2+ silence chunks and now speech resumes, return previous utterance
            if self.in_speech and self.consecutive_silence >= 2:
                total_samples = len(self.buffer) // 2
                if total_samples >= self.min_utterance_samples:
                    print(f"[SIMPLE] ✓ Speech resumed after silence, returning previous utterance ({total_samples} samples)")
                    result = bytes(self.buffer)
                    self.reset_feed()
                    # Don't re-add chunk - let the next feed() call handle it
                    return result
            
            # Now reset silence counter and continue
            self.consecutive_silence = 0
            self.consecutive_speech += 1
            
            if not self.in_speech and self.consecutive_speech >= 3:
                # Start of utterance (need 3 consecutive speech chunks)
                self.in_speech = True
                print("[SIMPLE] ✓ Speech started")
            
            return None
        
        else:
            # Looks like silence/noise
            self.consecutive_speech = 0
            
            if self.in_speech:
                # We were in speech, now silence
                self.consecutive_silence += 1
                print(f"[SIMPLE] Silence detected ({self.consecutive_silence}/3)")
                
                if self.consecutive_silence >= 2:
                    # End of utterance after 2 silence chunks (~1 second)
                    total_samples = len(self.buffer) // 2
                    
                    if total_samples >= self.min_utterance_samples:
                        print(f"[SIMPLE] ✓ Utterance complete ({total_samples} samples)")
                        result = bytes(self.buffer)
                        self.reset_feed()
                        return result
                    else:
                        print(f"[SIMPLE] ✗ Too short, discarding")
                        self.reset_feed()
                        return None
            else:
                # No speech detected, trim old silence
                if len(self.buffer) > self.sample_rate * 4:
                    self.buffer = self.buffer[-(self.sample_rate * 2):]
            
            return None