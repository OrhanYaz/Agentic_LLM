from openai import AzureOpenAI
import speech_recognition as sr
import os

client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"), 
                     api_version="2024-06-01", 
                     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))


def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    # Save audio file
    audio_path = "speech.wav"
    with open(audio_path, "wb") as f:
        f.write(audio.get_wav_data())

    # Use OpenAI's updated API call
    transcript = client.audio.transcriptions.create(
    file=(open(audio_path, "rb")),
    model="whisper"
    )

    return transcript.text