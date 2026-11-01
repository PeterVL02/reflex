"Prompt embeddings with Qwen/Qwen3-Embedding-0.6B, run locally."

from functools import cache

import numpy as np
import numpy.typing as npt
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_ID: str = "Qwen/Qwen3-Embedding-0.6B"


@cache
def load_embedder() -> SentenceTransformer:
    """Load the embedding model once (downloaded on first use, GPU if available)."""
    return SentenceTransformer(MODEL_ID)


def embed_prompts(
    prompts: list[str] | pd.Series[str], batch_size: int = 32
) -> npt.NDArray[np.float32]:
    """
    Embed prompts into L2-normalised vectors.

    Args:
        prompts (list[str] | pd.Series[str]): Texts to embed.
        batch_size (int): Prompts per forward pass. Lower it if the GPU runs out of memory.

    Returns:
        npt.NDArray[np.float32]: Array of shape (n_prompts, 1024).
    """
    return load_embedder().encode(
        list(prompts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
