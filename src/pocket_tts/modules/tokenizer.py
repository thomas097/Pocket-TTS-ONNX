import numpy as np
import sentencepiece as sp
from pathlib import Path

class SentencePieceTokenizer:
    def __init__(self, model_path: Path) -> None:
        self._tokenizer = sp.SentencePieceProcessor()
        self._tokenizer.Load(str(model_path))

    def tokenize(self, text: str) -> np.ndarray:
        """Tokenizes the input text.

        Example:
        ```
        "Hello world" → [154, 82, 991, ...]
        ```

        Args:
            text (str): Text to synthesize.

        Returns:
            np.ndarray[np.int64]: Array of token IDs with shape `(1, num_tokens)`.
        """
        text = text.strip()
        if not text:
            raise RuntimeError("Input text cannot be empty!")
        
        # Check proper punctuation
        if text[-1].isalnum():
            text += "."

        # Capitalize first letter
        if not text[0].isupper():
            text = text[0].upper() + text[1:]

        token_ids = self._tokenizer.Encode(text)
        return np.array(token_ids, dtype=np.int64).reshape(1, -1)