# from orion_core.tts.engine import SpeechEngine
# from orion_core.skills import Skills

# WAKE_WORD = "orion"

# def main():
#     tts = SpeechEngine()
#     skills = Skills()

#     print("Orion v2 ready. Say 'Orion' to wake me up.")

#     while True:
#         text = tts.listen_and_transcribe()
#         if not text:
#             continue

#         print("you ▶", text)

#         # Wake word detection
#         if not text.lower().startswith(WAKE_WORD):
#             continue  # ignore anything not addressed to Orion

#         # Remove wake word
#         command_text = text[len(WAKE_WORD):].strip()

#         # Orion thinks...
#         reply = skills.handle(command_text)
#         print("orion ▶", reply)

#         # Exit gracefully if reply signals stop
#         if "Exiting Orion" in reply:
#             tts.speak(reply)
#             break

#         tts.speak(reply)

# if __name__ == "__main__":
#     main()
