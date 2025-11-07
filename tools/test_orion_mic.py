"""
Standalone wake-word test using OrionCore logic
NO JS, NO UI, JUST MIC → handle_user_audio()

Goal:
- Confirm whether wakeword detection works correctly when fed
  real microphone audio directly into handle_user_audio().
"""

import asyncio
import numpy as np
import sounddevice as sd
import time
from orion_core.core import OrionCore


SAMPLE_RATE = 16000
CHUNK_SEC = 1.0                      # 1 second chunks (same as working tester)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)


async def main():
    core = OrionCore()
    print("[Test] Initializing OrionCore...")
    await core._on_init()

    print("[Test] OrionCore initialized. Starting speech engine...")
    await core.engine.start()

    print("[Test] ✅ Core ready. Starting mic. Say 'Orion' when ready.")
    print("----------------------------------------------------------")

    # Open a blocking microphone stream
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="float32",
                        blocksize=CHUNK_SAMPLES) as stream:

        while True:
            # Read exactly 1 second of audio
            data, _ = stream.read(CHUNK_SAMPLES)
            chunk = np.asarray(data, dtype=np.float32).squeeze()
            print("handle_user_audio")
            # IMPORTANT: feed raw float32 to core.handle_user_audio
            await core.handle_user_audio(chunk)

            # If wakeword fired, state becomes LISTEN
            if core.state.name == "LISTEN":
                print("\n✅ WAKEWORD DETECTED! You can speak now.\n")
                # For testing: stop after one detection
                break

            # Optional: tiny pause (non-blocking)
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Test] Exiting.")
