import os
import sys
import traceback

# Ensure project root is on sys.path so `src` imports work when run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Any, Dict, List, Iterable, cast  # noqa: E402

# Local package imports (allowed after sys.path manipulation)
from src.sentiment import (  # noqa: E402
    SentimentAnalyzer,
    analyze_sentiment_batch,
)
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

# For clearer typing of matplotlib bar objects
from matplotlib.patches import Rectangle  # noqa: E402

# Matplotlib headless backend
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Create dummy data: 200 reviews with ~50% negative content
negative_texts = [
    "This engine is terrible and I am very disappointed.",
    "Bad heat management and constant vibration issues.",
    "Complete waste of money worst purchase ever.",
    "Product is broken and unreliable, such a nightmare.",
    "Very disappointing and overpriced, creak and tear after a week.",
]
positive_texts = [
    "I love the smooth ride and excellent pickup.",
    "Great performance, very satisfied with the product.",
    "Amazing build quality and reliable service.",
    "Comfortable and fast, would recommend.",
]
neutral_texts = [
    "The product arrived on time.",
    "It works as described.",
    "Average experience, nothing special.",
]


def main():
    try:
        np.random.seed(42)

        rows: List[Dict[str, Any]] = []
        for i in range(200):
            if i % 2 == 0:
                text = np.random.choice(negative_texts)
            else:
                text = np.random.choice(positive_texts + neutral_texts)
            rows.append({
                "review_id": f"r{i}",
                "preprocessed_text": str(text),
                "tokens": [],
                "embedding": None,
            })

        df = pd.DataFrame(rows)

        analyzer = SentimentAnalyzer()
        result_df = analyze_sentiment_batch(df, analyzer=analyzer, use_parallel=False)

        # Aggregate
        def _extract_label(s: Any) -> str:
            if isinstance(s, dict):
                s_dict = cast(Dict[str, Any], s)
                lbl = s_dict.get("label")
                return lbl if isinstance(lbl, str) else "neutral"
            return "neutral"

        labels_series = result_df["sentiment"].apply(_extract_label)  # type: ignore[reportUnknownMemberType]  # noqa: E501
        counts = labels_series.value_counts()
        print("Sentiment distribution:\n")
        print(counts.to_string())

        # Ensure graphs dir
        out_dir = os.path.join(ROOT, "data/graphs")
        os.makedirs(out_dir, exist_ok=True)

        # Simple bar plot
        labels = counts.index.tolist()
        values = [int(counts.get(lbl, 0)) for lbl in labels]
        colors_map = {
            "positive": "#2ca02c",
            "negative": "#d62728",
            "neutral": "#7f7f7f",
        }
        colors = [colors_map.get(lbl, "#7f7f7f") for lbl in labels]

        plt.figure(figsize=(6, 4))  # type: ignore[reportUnknownMemberType]
        bars = cast(Iterable[Rectangle], plt.bar(labels, values, color=colors))  # type: ignore[reportUnknownMemberType]  # noqa: E501
        plt.title("Dummy Sentiment Distribution")  # type: ignore[reportUnknownMemberType]
        for bar, v in zip(bars, values):
            cast(Any, plt).text(
                bar.get_x() + bar.get_width() / 2,
                v + 1,
                str(v),
                ha="center",
                va="bottom",
            )  # type: ignore[reportUnknownMemberType]

        out_file = os.path.join(out_dir, "sentiment_distribution_test.png")  # noqa: E501
        plt.savefig(out_file, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]  # noqa: E501
        print("Wrote", out_file)

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
