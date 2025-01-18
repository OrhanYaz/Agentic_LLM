import torch
from TTS.api import TTS
import simpleaudio as sa
import numpy as np

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
# tts = TTS("tts_models/fr/css10/vits").to(device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
def speak_text(text):
    output_path = "output.wav"
    tts.tts_to_file(text=text, speaker_wav="speech.wav", file_path=output_path, language='fr')

    wave_obj = sa.WaveObject.from_wave_file("output.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()