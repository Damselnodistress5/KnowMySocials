"""
Embedding Generation Module
Generates sentence embeddings for preprocessed review text using Sentence-Transformers.
"""

from hashlib import sha256
from pathlib import Path
from typing import Any, cast, Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import]

from src.utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load a sentence-transformers model for text embedding generation."""
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
        return model
    except Exception as exc:
        logger.warning(
            "Could not load embedding model; falling back to a deterministic local embedding generator."
        )
        logger.warning(str(exc))
        raise


def fallback_sentence_embeddings(sentences: list[str], dim: int = 384) -> np.ndarray[Any, Any]:
    """Create deterministic fallback embeddings when external models are unavailable."""
    embeddings = np.zeros((len(sentences), dim), dtype=float)
    for idx, sentence in enumerate(sentences):
        digest = sha256(sentence.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.RandomState(seed)
        embeddings[idx] = rng.standard_normal(dim)
    return embeddings


def generate_sentence_embeddings(
    texts: Iterable[Optional[str]],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
    save_path: Optional[Path] = None
) -> np.ndarray[Any, Any]:
    """Generate sentence embeddings for a list of input documents.

    Args:
        texts: Iterable of preprocessed review text.
        model_name: Sentence-Transformers model name.
        batch_size: Batch size for encoding.
        show_progress: Whether to show a progress bar.
        save_path: Optional path to save embeddings as a numpy file.

    Returns:
        Embedding matrix with shape (n_samples, embedding_dim).
    """
    sentences = [item if item is not None else "" for item in texts]
    try:
        model = load_embedding_model(model_name)
    except Exception:
        logger.warning("Embedding model load failed; using deterministic fallback embeddings.")
        embeddings = fallback_sentence_embeddings(sentences, dim=384)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, embeddings)
            logger.info(f"Saved fallback embeddings to {save_path}")
        logger.info(f"Generated fallback embeddings with shape {embeddings.shape}")
        return embeddings

    logger.info(f"Generating embeddings for {len(sentences)} records")
    model_encode = getattr(model, "encode")  # type: ignore[assignment]
    raw_embeddings = model_encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True
    )  # type: ignore[call-arg]
    embeddings = cast(np.ndarray[Any, Any], np.asarray(raw_embeddings))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, embeddings)
        logger.info(f"Saved embeddings to {save_path}")

    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings
