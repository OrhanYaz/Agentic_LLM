import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason, CancellationReason, audio
import os


# Set Azure Speech Services Credentials
SPEECH_KEY = os.getenv("AZURE_OPENAI_KEY")
SPEECH_REGION = os.getenv("REGION")
SPEECH_ENDPOINT = os.getenv("AZURE_OPENAI_TTS_ENDPOINT")

def speak_text(text):
    speech_config = SpeechConfig(subscription=SPEECH_KEY, endpoint=SPEECH_ENDPOINT) #you can either use region or endpoint
    speech_config.speech_synthesis_language = "en-NZ"
    speech_config.speech_synthesis_voice_name = "en-NZ-MollyNeural"
    audio_config = audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    print("Speaking response...")
    result = synthesizer.speak_text_async(text).get()

    if result.reason == ResultReason.SynthesizingAudioCompleted:
            print("Speech Synthesis Succeeded!")
    elif result.reason == ResultReason.Canceled:
        cancellation = result.cancellation_details
        print(f"peech Synthesis Canceled: {cancellation.reason}")
        if cancellation.reason == CancellationReason.Error:
            print(f"Error details: {cancellation.error_details}")

