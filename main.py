import numpy as np
import matplotlib.pyplot as plt

from src.pocket_tts import PocketTTS

model = PocketTTS(
    model_dir="models/pocket-tts",
    voice="samples/reference_sample.wav"
    )

text = "Hello how are you?"

chunks = np.concatenate(list(model.stream(text)))

plt.plot(chunks)
plt.show()