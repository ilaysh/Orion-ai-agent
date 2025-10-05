import asyncio
import edge_tts

VOICE = "en-US-AriaNeural"

async def _speak_bytes_async(
    text: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%"
):
    communicate = edge_tts.Communicate(
        text,
        voice=VOICE,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    audio = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio

async def tts_bytes(
    text: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%"
):
    return await _speak_bytes_async(text, rate, pitch, volume)

def tts_bytes_sync(
    text: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%"
):
    return asyncio.run(_speak_bytes_async(text, rate, pitch, volume))
