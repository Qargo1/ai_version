import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Говорите что-нибудь...")
    audio = recognizer.listen(source, timeout=5)
    print("Аудио записано!")

try:
    text = recognizer.recognize_google(audio, language="ru-RU")
    print(f"Распознано: {text}")
except sr.UnknownValueError:
    print("Не удалось распознать речь")
except sr.RequestError:
    print("Ошибка запроса к сервису Google STT")