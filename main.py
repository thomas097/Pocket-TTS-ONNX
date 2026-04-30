import numpy as np
import matplotlib.pyplot as plt

from src.pocket_tts.pocket_tts import PocketTTS

model = PocketTTS(
    models_dir="models/pocket-tts/english",
    voice="samples/reference_sample.wav"
    )

text = "Hello how are you?"

chunks = np.concatenate(list(model.stream(text)))

plt.plot(chunks)
plt.show()