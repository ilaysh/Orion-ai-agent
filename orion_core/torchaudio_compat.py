def fix_torchaudio_backend():
    import torchaudio

    # אם הפונקציה כבר קיימת - לא לעשות כלום
    if hasattr(torchaudio, "list_audio_backends"):
        return

    # אחרת נגדיר פונקציה מדומה
    def _list_audio_backends():
        # torchaudio>=2.8 לא צריך backend list, נחזיר אפשרויות נפוצות
        return ["sox_io", "soundfile", "ffmpeg"]

    # נרשום אותה
    torchaudio.list_audio_backends = _list_audio_backends

    # ננסה לבחור backend יציב
    for backend in ["sox_io", "soundfile", None]:
        try:
            torchaudio.set_audio_backend(backend)
            break
        except Exception:
            pass
