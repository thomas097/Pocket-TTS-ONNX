import time
import numpy as np
from pathlib import Path
from typing import Literal, Generator

from .modules.tokenizer import SentencePieceTokenizer
from .modules.mimi import MimiEncoder, MimiDecoder
from .modules.flow_matching import TextConditioner, FlowLmConditioner, FlowLmNetwork

class PocketTTSConfig:
    sample_rate: int = 24_000
    samples_per_frame: int = 1920
    temperature: float = 0.1
    lsd_steps: int = 5
    max_frames: int = 512
    max_chunk_frames: int = 15
    frames_after_eos: int = 5

class PocketTTS:
    """
    The Pocket TTS text-to-speech model from Kyutai Labs based on:
    "Continuous Audio Language Models" by Rouard et al. (2025)
    """

    def __init__(
        self,
        model_dir: Path,
        voice_ref: Path,
        config: PocketTTSConfig = PocketTTSConfig(),
        precision: Literal['int8', 'fp32'] = 'int8',
        device: Literal['cpu', 'cuda'] = 'cpu'
        ) -> None:
        self._config = config

        self._mimi_encoder = MimiEncoder(
            model_path=model_dir / "mimi_encoder.onnx",
            device=device
        )

        self._text_conditioner = TextConditioner(
            model_path=model_dir / "text_conditioner.onnx",
            device=device
        )

        self._flow_lm_conditioner = FlowLmConditioner(
            model_path=model_dir / "flow_lm_main.onnx",
            device=device,
            precision=precision
        )

        self._flow_lm_network = FlowLmNetwork(
            model_path=model_dir / "flow_lm_flow.onnx",
            device=device,
            precision=precision
        )

        self._mimi_decoder = MimiDecoder(
            model_path=model_dir / "mimi_decoder.onnx",
            device=device,
            precision=precision
        )

        self._tokenizer = SentencePieceTokenizer(
            model_path=model_dir / "tokenizer.model"
        )

        self._flow_buffers = self._precompute_flow_buffers()
        self._voice_emb = self._mimi_encoder.encode(voice_ref)

    # ============
    #   Internal
    # ============

    def _precompute_flow_buffers(self) -> np.ndarray:
        """
        Pre-computes time step buffers for flow matching.

        Returns:
            numpy.ndarray: Pre-computed s/t flow buffers with
                shape `(config.lsd_steps, 2, 1, 1)`.
        """
        steps = self._config.lsd_steps
        times = np.linspace(0.0, 1.0, num=steps + 1, dtype=np.float32)
        st_buffers = np.stack((times[:-1], times[1:]), axis=1)
        return st_buffers[..., None, None]
    
    def _run_flow_lm(
        self,
        voice_emb: np.ndarray,
        token_ids: np.ndarray
        ) -> Generator[np.ndarray, None, None]:
        """
        Run flow LM autoregressive generation, yielding latents.
        Uses dual model architecture:
        - flow_lm_main: transformer/conditioner (produces conditioning vector)
        - flow_lm_flow: flow network (Euler integration for latent sampling)
        Yields individual latent frames as they're generated.
        """
        text_emb = self._text_conditioner.embed_text(token_ids=token_ids)

        flow_lm_state = self._flow_lm_conditioner.reset_state()

        empty_seq = np.zeros((1, 0, 32), dtype=np.float32)
        empty_text = np.zeros((1, 0, 1024), dtype=np.float32)

        # Voice conditioning pass
        self._flow_lm_conditioner.condition_emb(
            sequence=empty_seq,
            embeddings=voice_emb,
            state=flow_lm_state,
            update_state=True
        )

        # Text conditioning pass
        self._flow_lm_conditioner.condition_emb(
            sequence=empty_seq,
            embeddings=text_emb,
            state=flow_lm_state,
            update_state=True
        )

        # Autoregressive generation
        curr = np.full((1, 1, 32), np.nan, dtype=np.float32)
        dt = 1.0 / self._config.lsd_steps
        
        eos_step = None

        for step in range(self._config.max_frames):
            # Run main model to get conditioning and EOS
            res_step = self._flow_lm_conditioner.condition_emb(
                sequence=curr,
                embeddings=empty_text,
                state=flow_lm_state,
                update_state=True
            )

            conditioning = res_step[0]  # [1, 1, dim]
            eos_logit = res_step[1]     # [1, 1]

            # Check EOS - record when EOS is first detected
            if eos_logit[0][0] > -4.0 and eos_step is None:
                eos_step = step
            
            # Stop only after frames_after_eos additional frames
            if eos_step is not None and step >= eos_step + self._config.frames_after_eos:
                break

            # Flow matching with external loop (enables temperature control)
            # Initialize with noise scaled by temperature
            std = np.sqrt(self._config.temperature) if self._config.temperature > 0 else 0.0
            x = np.random.normal(0, std, (1, 32)).astype(np.float32) if std > 0 else np.zeros((1, 32), dtype=np.float32)

            # Euler integration over flow network
            for j in range(self._config.lsd_steps):
                s_arr, t_arr = self._flow_buffers[j]
                flow_out = self._flow_lm_network.pred_flow(
                    conditioning=conditioning,
                    s_arr=s_arr,
                    t_arr=t_arr,
                    x=x
                    )
                x = x + flow_out[0] * dt

            latent = x.reshape(1, 1, 32)
            yield latent
            curr = latent

    def _decode_latent_frames(
            self, 
            latent_frames: list[np.ndarray], 
            mimi_state: dict[str, np.ndarray]
            ) -> np.ndarray:
        return self._mimi_decoder.decode(
            latent=np.concatenate(latent_frames, axis=1),
            state=mimi_state
        ).squeeze()
    
    # ==============
    #   Public API
    # ==============

    def stream(self, text: str) -> Generator:
        text_ids = self._tokenizer.tokenize(text)

        frames = []
        n_decoded_frames = 0
        
        mimi_state = self._mimi_decoder.reset_state()
        
        for latent_frame in self._run_flow_lm(self._voice_emb, text_ids):
            frames.append(latent_frame)

            # Decode ASAP in large batches
            pending = len(frames) - n_decoded_frames
            if pending >= self._config.max_chunk_frames:

                audio_chunk = self._decode_latent_frames(
                    latent_frames=frames[n_decoded_frames: n_decoded_frames + self._config.max_chunk_frames],
                    mimi_state=mimi_state
                )

                n_decoded_frames += self._config.max_chunk_frames
                yield audio_chunk

        # Flush leftovers
        if n_decoded_frames < len(frames):
            audio_chunk = self._decode_latent_frames(
                latent_frames=frames[n_decoded_frames:],
                mimi_state=mimi_state
            )

            yield audio_chunk