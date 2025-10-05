import speech_recognition as sr

class GoogleTranscriber:
    def __init__(self, language="en-US"):
        self.language = language
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def transcribe_once(self,wav):
        with self.mic as source:
            print("🎙️ Adjusting for background noise...")
            self.recognizer.adjust_for_ambient_noise(source)

            print("🎙️ Speak now...")
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"Google API error: {e}")
            return ""
