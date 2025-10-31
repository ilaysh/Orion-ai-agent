import asyncio
from orion_core.tts.engine import SpeechEngine


async def main():
    se = SpeechEngine()
    await se.init()  # sets up loop, VAD, transcriber, mic
    await se.start()
    print("🎙️ Speak a full sentence, then stay silent for a second.\n"
          "Orion will transcribe after silence is detected.\n"
          "Press Ctrl+C to stop.\n")

    try:

        text = await se.listen_and_transcribe()
        print(f"[STT] → {text}")

    except KeyboardInterrupt:
        print("\n🛑 Test stopped.")

asyncio.run(main())
