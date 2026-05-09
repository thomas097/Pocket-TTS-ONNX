import numpy as np
import sounddevice as sd
from pathlib import Path

from src.pocket_tts import PocketTTS

model = PocketTTS(
    model_dir=Path("./models/pocket-tts"),
    voice_ref=Path("./samples/jean.wav")
    )

print("Type something (Enter to quit)")
while True:
    text = input(">> ")
    if not text:
        break

    chunks = list(model.stream_fast(text))
    audio = np.concatenate(chunks, axis=0)

    sd.play(audio, samplerate=24_000, blocking=True) 