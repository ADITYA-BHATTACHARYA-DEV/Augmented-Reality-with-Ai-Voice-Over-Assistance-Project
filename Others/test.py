import pyttsx3
import time

def test_speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

test_speak("Testing one.")
time.sleep(1)
test_speak("Testing two.")