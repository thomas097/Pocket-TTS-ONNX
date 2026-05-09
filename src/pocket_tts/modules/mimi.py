import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Literal
from scipy.signal import resample_poly

from ._onnx_base_model import OnnxBaseModel


class MimiConfig:
    sample_rate: int = 24_000


class MimiEncoder(OnnxBaseModel):
    def __init__(
            self, 
            model_path: Path, 
            config: MimiConfig = MimiConfig(),
            device: Literal['cpu', 'cuda'] = 'cpu'
            ):
        self._config = config
        super().__init__(
            model_path=model_path, 
            device=device
            )
        
    def _resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        gcd = np.gcd(orig_sr, self._config.sample_rate)
        audio = resample_poly(
            x=audio, 
            up=self._config.sample_rate // gcd, 
            down=orig_sr // gcd
            )
        return audio
        
    def _load_audio_from_file(self, path: Path) -> np.ndarray:
        """
        Loads a PCM waveform from an audio file and prepares it for encoding.
        """
        audio, orig_sr = sf.read(path, dtype='float32')

        # Convert to mono (single channel)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 24kHz if needed
        if orig_sr != self._config.sample_rate:
            audio = self._resample_audio(
                audio=audio, 
                orig_sr=orig_sr
                )

        # Normalize 
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        return audio.reshape(1, 1, -1)
        
    def encode(self, audio: np.ndarray | Path) -> np.ndarray:
        if isinstance(audio, Path):
            audio = self._load_audio_from_file(audio)
        return self._forward(audio=audio)[0]


class MimiDecoder(OnnxBaseModel):
    def __init__(
            self, 
            model_path: Path, 
            device: Literal['cpu', 'cuda'] = 'cpu',
            precision: str | None = None
            ):
        super().__init__(
            model_path=model_path, 
            device=device,
            precision=precision
            )
        
        # All outputs corresponding to the hidden state
        # nodes of the decoder model
        self._state_nodes: list = self.outputs[1:]

    def reset_state(self) -> dict:
        """
        Initialize state tensors for a stateful decoder model.
        """
        state = {}

        type_map = {
            "tensor(float)": np.float32,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }

        for inp in self._session.get_inputs():
            if inp.name.startswith("state_"):
                shape = [s if isinstance(s, int) else 0 for s in inp.shape]
                dtype = type_map.get(inp.type, np.float32)
                state[inp.name] = np.zeros(shape, dtype=dtype)

        return state
    
    def _update_state(self, state: dict[str, np.ndarray], new_state: list[np.ndarray]) -> None:
        for i, node in enumerate(self._state_nodes):
            if node.name.startswith("out_state_"):
                idx = int(node.name[10:])
                state[f"state_{idx}"] = new_state[i]
        
    def decode(
            self, 
            latent: np.ndarray, 
            state: dict[str, np.ndarray], 
            update_state: bool = True
            ) -> np.ndarray:
        audio_chunk, *new_state = self._forward(latent=latent, **state) 

        if update_state:
            self._update_state(state, new_state)

        return audio_chunk