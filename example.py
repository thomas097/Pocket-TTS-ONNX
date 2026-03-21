import sounddevice as sd
import soundfile as sf
from pocket_tts import PocketTTS

model = PocketTTS(
    models_dir="checkpoints/pocket-tts",
    voice="reference_sample.wav"
    )

for chunk in model.stream("Hello how are you?"):
    print(chunk)
