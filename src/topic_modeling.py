"""
Topic Modeling Module
Uses BERTopic and precomputed embeddings to discover topics across Honda review text.
"""

from typing import Any, cast, List, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic  # type: ignore[import]
from hdbscan import HDBSCAN  # type: ignore[import]
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP  # type: ignore[import]

from src.utils import setup_logger

logger = setup_logger(__name__)


def _build_topic_model(
    n_neighbors: int = 15,
    n_components: int = 5,
    min_topic_size: int = 10,
    vectorizer_max_features: int = 15000,
    n_reviews: int = 5
) -> BERTopic:
    """Create a BERTopic instance configured for automotive review discovery."""
    # Adjust parameters for small datasets
    effective_n_neighbors = min(n_neighbors, max(2, n_reviews - 1)) if n_reviews > 1 else 2
    effective_n_components = min(n_components, max(2, n_reviews - 2)) if n_reviews > 2 else 2

    umap_model = UMAP(
        n_neighbors=effective_n_neighbors,
        n_components=effective_n_components,
        metric="cosine",
        random_state=42,
        min_dist=0.0,
        init="random"
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min(min_topic_size, max(2, n_reviews // 2)),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=vectorizer_max_features
    )

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        verbose=False,
        top_n_words=12
    )

    return topic_model


def discover_topics(
    review_ids: List[str],
    documents: List[str],
    embeddings: np.ndarray[Any, Any],
    min_topic_size: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Discover topics using BERTopic and return assignments plus topic summary."""
    if len(review_ids) != len(documents) or len(review_ids) != embeddings.shape[0]:
        raise ValueError("review_ids, documents and embeddings must be the same length")

    logger.info("Starting BERTopic topic discovery")
    topic_model = _build_topic_model(min_topic_size=min_topic_size, n_reviews=len(review_ids))
    topics, probabilities = cast(
        Tuple[List[int], np.ndarray[Any, Any] | None],
        topic_model.fit_transform(documents, embeddings)  # type: ignore[reportUnknownFunctionType]
    )  # type: ignore[assignment]

    logger.info("BERTopic fit complete")

    topic_labels: dict[int, str] = getattr(topic_model, "topic_labels_", {})  # type: ignore[assignment]
    topic_info = topic_model.get_topic_info()

    assignment_rows: List[dict[str, Any]] = []
    for idx, review_id in enumerate(review_ids):
        topic_id = int(topics[idx])
        probability = 0.0

        if probabilities is not None:
            row_prob = probabilities[idx]
            if isinstance(row_prob, np.ndarray) and 0 <= topic_id < row_prob.shape[0]:
                probability = float(row_prob[topic_id])
            elif isinstance(row_prob, list):
                row_prob = cast(List[float], row_prob)
                if 0 <= topic_id < len(row_prob):
                    probability = float(row_prob[topic_id])

        keywords = ""
        if topic_id >= 0:
            topic_terms = topic_model.get_topic(topic_id)
            if not isinstance(topic_terms, list):
                topic_terms = []
            topic_terms = cast(List[tuple[str, Any]], topic_terms)
            keywords = ", ".join(
                str(term) for term, _ in topic_terms[:8]
            )

        assignment_rows.append({
            "review_id": review_id,
            "topic_id": topic_id,
            "topic_probability": probability,
            "topic_keywords": keywords,
            "topic_label": topic_labels.get(topic_id, "") if topic_id >= 0 else "Outlier"
        })

    assignment_df = pd.DataFrame(assignment_rows)

    keyword_rows: List[dict[str, Any]] = []
    for row in topic_info.itertuples(index=False):
        topic_id = int(getattr(row, "Topic", -1))
        if topic_id == -1:
            continue

        topic_terms = topic_model.get_topic(topic_id)
        if not isinstance(topic_terms, list):
            topic_terms = []
        topic_terms = cast(List[tuple[str, Any]], topic_terms)

        keyword_rows.append({
            "topic_id": topic_id,
            "Name": str(getattr(row, "Name", "")),
            "keywords": ", ".join(
                str(term) for term, _ in topic_terms[:12]
            ),
            "Count": int(getattr(row, "Count", 0))
        })

    topic_summary_df = pd.DataFrame(keyword_rows)
    logger.info(f"Discovered {len(topic_summary_df)} meaningful topics")

    return assignment_df, topic_summary_df
