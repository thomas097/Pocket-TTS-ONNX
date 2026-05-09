import os
import onnxruntime as ort
from pathlib import Path
from typing import Literal, Any

_ORT_PROVIDERS = {
    'cpu': "CPUExecutionProvider",
    'cuda': "CUDAExecutionProvider"
}

class OnnxBaseModel:
    """
    Base model to abtract away the boilerplate associated with the ONNX Runtime API.
    """
    def __init__(
            self, 
            model_path: Path, 
            device: Literal['cuda', 'cpu'],
            precision: str | None = None
            ):
        assert model_path.suffix == '.onnx', \
            f"Unable to load '{model_path}' as an ONNX model."
        
        # We assume the model's precision is denoted in 
        # the model's filename via a special suffix
        if precision is not None:
            model_path = model_path.with_name(model_path.stem + f"_{precision}.onnx")

        self._session = ort.InferenceSession(
                model_path,
                sess_options=self._get_session_options(), 
                providers=[self._get_provider(device)]
            )
        
    @staticmethod
    def _get_session_options() -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(os.cpu_count() or 4, 4)
        options.inter_op_num_threads = 1
        return options

    @staticmethod
    def _get_provider(device: Literal['cuda', 'cpu']) -> str:
        provider = _ORT_PROVIDERS.get(device, None)
        if provider is None:
            raise RuntimeError(f"device '{device}' not understood. Choose from {list(_ORT_PROVIDERS.keys())}")
        
        if provider not in ort.get_all_providers():
            raise RuntimeError(f"The ONNX runtime was not installed with the {provider}.")
        
        return provider
    
    @property
    def outputs(self) -> list[ort.NodeArg]:
        return self._session.get_outputs() # type:ignore
    
    @property
    def inputs(self) -> list[ort.NodeArg]:
        return self._session.get_inputs() # type:ignore
    
    def _forward(self, **kwargs) -> Any:
        return self._session.run(
            output_names=None,
            input_feed=kwargs
        )