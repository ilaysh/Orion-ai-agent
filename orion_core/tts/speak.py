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
    """Async TTS helper."""
    return await _speak_bytes_async(text, rate, pitch, volume)


def tts_bytes_sync(
    text: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%"
):
    """
    Works both inside and outside an async event loop.
    Inside FastAPI (running event loop) → uses `asyncio.create_task()`
    Outside (CLI/test) → uses `asyncio.run()`.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop → standalone use
        return asyncio.run(_speak_bytes_async(text, rate, pitch, volume))
    else:
        # Already in async context — schedule and wait safely
        task = loop.create_task(_speak_bytes_async(text, rate, pitch, volume))
        # Use `asyncio.run_coroutine_threadsafe` if running in a different thread
        # but here, since we're in same loop, just gather:
        future = asyncio.ensure_future(task)
        return loop.run_until_complete(future)
