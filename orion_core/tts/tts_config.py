import pyttsx3

def speak(text: str, rate: int = 150, volume: float = 1.0, voice_id: str = None):
    engine = pyttsx3.init()

    # set playback speed (default ~200 wpm, lower = slower)
    engine.setProperty("rate", rate)

    # set volume (0.0 to 1.0)
    engine.setProperty("volume", volume)

    # choose voice (if provided)
    if voice_id:
        engine.setProperty("voice", voice_id)

    engine.say(text)
    engine.runAndWait()

def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    for i, v in enumerate(voices):
        print(f"{i}: {v.id} | {v.name} | {v.languages}")

if __name__ == "__main__":
    print("Available voices:")
    list_voices()

    print("\nSpeaking demo...")
    speak("שלום מה שלומך היום? אני מדבר עם API", rate=160,voice_id="he")
