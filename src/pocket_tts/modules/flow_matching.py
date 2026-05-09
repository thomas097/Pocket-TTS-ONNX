import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Literal
from scipy.signal import resample_poly

from ._onnx_base_model import OnnxBaseModel


class TextConditioner(OnnxBaseModel):
    def __init__(
            self, 
            model_path: Path, 
            device: Literal['cpu', 'cuda'] = 'cpu'
            ):
        super().__init__(
            model_path=model_path, 
            device=device
            )
        
    def embed_text(self, token_ids: np.ndarray) -> np.ndarray:
        return self._forward(token_ids=token_ids)[0]


class FlowLmConditioner(OnnxBaseModel):
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
        
    def reset_state(self) -> dict:
        """
        Initialize state tensors for a stateful flow model.
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
    
    def _update_state(self, state: dict[str, np.ndarray], result: list[np.ndarray]):
        """Update state dict from model outputs."""
        for i in range(2, len(self._session.get_outputs())):
            name = self._session.get_outputs()[i].name
            if name.startswith("out_state_"):
                idx = int(name.replace("out_state_", ""))
                state[f"state_{idx}"] = result[i]
    
    def condition_emb(
            self, 
            sequence: np.ndarray, 
            embeddings: np.ndarray, 
            state: dict[str, np.ndarray],
            update_state: bool = True
            ) -> np.ndarray:
        
        result = self._forward(
            sequence=sequence,
            text_embeddings=embeddings,
            **state
            )
        
        if update_state:
            self._update_state(state, result)

        return result
        
        
class FlowLmNetwork(OnnxBaseModel):
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
        
    def pred_flow(
            self, 
            conditioning: np.ndarray, 
            s_arr: np.ndarray,
            t_arr: np.ndarray,
            x: np.ndarray
            ) -> np.ndarray:
        return self._forward(c=conditioning, s=s_arr, t=t_arr, x=x)
        
    