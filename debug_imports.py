import speech_recognition as sr

print("speech_recognition imported successfully.")

try:
    r = sr.Recognizer()
    print("Recognizer instance created.")
    with sr.Microphone() as source:
        print("Microphone accessed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
