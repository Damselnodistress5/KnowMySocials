from collections import Counter
from pathlib import Path
import ast
from typing import Any, Dict, List, cast

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("data/reviews_final_structured.csv")
OUT_DIR = Path("data/graphs")


def parse_sentiment_cell(value: Any) -> Any:
    """Safely parse a CSV cell that may contain a Python literal.

    If the cell is a string representing a dict/list, attempt literal_eval.
    Otherwise return the original value unchanged.
    """
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def build_sentiment_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Return a mapping of sentiment label -> count extracted from DataFrame.

    The `sentiment` column may contain dicts like {'label': 'negative'} or
    raw strings. This function normalizes entries to str labels.
    """
    graphs: List[str] = []
    # ensure the column exists
    if "sentiment" not in df.columns:
        return {}

    for item in df["sentiment"].tolist():
        if isinstance(item, dict):
            item_dict = cast(Dict[str, Any], item)
            lbl = item_dict.get("label")
            graphs.append(lbl if isinstance(lbl, str) else "neutral")
        else:
            graphs.append(str(item))

    return dict(Counter(graphs))


def main() -> None:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # Apply safe parsing to the sentiment column if present
    if "sentiment" in df.columns:
        # cast to pd.Series[Any] to help the type checker; pandas' apply is
        # a complex overload that can confuse Pylance so a narrow ignore is used.
        df["sentiment"] = cast(pd.Series, df["sentiment"]).apply(parse_sentiment_cell)  # type: ignore[reportUnknownMemberType]

    counts = build_sentiment_counts(df)
    print("counts", counts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plotting
    # Use casts to silence complex third-party signatures in Pylance for plotting
    fig, ax = cast(Any, plt).subplots(figsize=(4, 3))  # type: ignore[reportUnknownMemberType]
    labels = list(counts.keys())
    values = list(counts.values())
    cast(Any, ax).bar(labels, values)  # type: ignore[reportUnknownMemberType]
    out_file = OUT_DIR / "graph_test2.png"
    cast(Any, fig).savefig(out_file)  # type: ignore[reportUnknownMemberType]
    plt.close(fig)
    print("saved", out_file.exists())


if __name__ == "__main__":
    main()
