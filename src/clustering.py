"""
Clustering Module
Performs density-based clustering on review embeddings using HDBSCAN.
"""

from typing import Any, cast, List

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN  # type: ignore[import]

from src.utils import setup_logger

logger = setup_logger(__name__)


def perform_hdbscan_clustering(
    review_ids: List[str],
    embeddings: np.ndarray[Any, Any],
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    metric: str = "euclidean"
) -> pd.DataFrame:
    """Run HDBSCAN clustering and assign cluster labels for each review."""
    if len(review_ids) != embeddings.shape[0]:
        raise ValueError("review_ids and embeddings must have the same length")

    logger.info("Running HDBSCAN clustering on review embeddings")
    embeddings = np.asarray(embeddings)
    n_samples = embeddings.shape[0]

    safe_min_cluster_size = min(min_cluster_size, max(2, n_samples // 2)) if n_samples > 2 else max(2, n_samples)
    safe_min_cluster_size = min(safe_min_cluster_size, n_samples)
    safe_min_samples = min_samples if min_samples is not None else min(safe_min_cluster_size, max(1, n_samples - 1))
    safe_min_samples = max(1, safe_min_samples)

    logger.info(
        f"HDBSCAN config: n_samples={n_samples}, min_cluster_size={safe_min_cluster_size}, min_samples={safe_min_samples}"
    )

    clusterer = HDBSCAN(
        min_cluster_size=safe_min_cluster_size,
        min_samples=safe_min_samples,
        metric=metric,
        cluster_selection_method="eom",
        prediction_data=True
    )

    labels = cast(np.ndarray[Any, Any], np.asarray(clusterer.fit_predict(embeddings)))  # type: ignore[assignment]
    if hasattr(clusterer, "probabilities_"):
        probabilities = cast(
            np.ndarray[Any, Any],
            np.asarray(getattr(clusterer, "probabilities_") )  # type: ignore[assignment]
        ).astype(float)
    else:
        probabilities = np.zeros(len(labels), dtype=float)

    assignment_rows: List[dict[str, Any]] = []
    for review_id, label, prob in zip(review_ids, labels, probabilities):
        assignment_rows.append({
            "review_id": review_id,
            "cluster_label": int(label),
            "cluster_probability": float(prob),
            "is_noise": label == -1
        })

    cluster_df = pd.DataFrame(assignment_rows)
    logger.info(f"Clustering completed, found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    return cluster_df
