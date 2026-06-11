"""Dry-run pipeline for KnowMySocials.
Runs preprocessing, embedding, topic discovery, and clustering in-memory
without writing to the database.
"""

import config
import pandas as pd
from pprint import pprint
from typing import cast

# Imports
from src.preprocessing import preprocess_data
from src.embedding import generate_sentence_embeddings
from src.topic_modeling import discover_topics
from src.clustering import perform_hdbscan_clustering


def run_dry():
    print("🚀 Starting Dry Run Pipeline for KnowMySocials...\n")

    # 1. Load Data
    print(f"Loading CSV: {config.CSV_FILE_PATH}")
    try:
        # pandas stubs are noisy for read_csv; cast to DataFrame for the type-checker
        df = cast(pd.DataFrame, pd.read_csv(config.CSV_FILE_PATH))  # type: ignore[reportUnknownMemberType]
        print(f"✅ Loaded {len(df)} rows\n")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # 2. Preprocessing (Hybrid)
    print("🔄 Running Hybrid Preprocessing...")
    pre_df = preprocess_data(df)
    print(f"✅ Preprocessed {len(pre_df)} rows\n")

    # 3. Hybrid Routing Stats
    if "is_complex" in pre_df.columns:
        simple_count = int((pre_df["is_complex"] == False).sum())
        complex_count = int((pre_df["is_complex"] == True).sum())
    else:
        simple_count = 0
        complex_count = len(pre_df)

    print(f"📊 Hybrid Routing → Simple: {simple_count} | Complex: {complex_count}\n")

    # Sample of preprocessed data
    print("📋 Sample Preprocessed Data:")
    # cast the to_dict result to Any to avoid pandas overload typing noise
    records = pre_df.head(3).to_dict(orient="records")  # type: ignore[reportUnknownMemberType]
    pprint(records)
    print("-" * 80)

    # ================== AI PATH (Only Complex Reviews) ==================
    # Prefer explicit checks over DataFrame.get to avoid stub overloads
    if "is_complex" in pre_df.columns:
        mask_series = cast(pd.Series, pre_df["is_complex"])  # type: ignore[reportUnknownMemberType]
    else:
        mask_series = cast(pd.Series, pd.Series([True] * len(pre_df)))  # type: ignore[reportUnknownMemberType]
    complex_df = pre_df[mask_series == True]

    if len(complex_df) > 0:
        print(f"⚡ Running Full AI Pipeline on {len(complex_df)} Complex Reviews...")

        texts = complex_df["preprocessed_text"].astype(str).tolist()
        ids = complex_df["review_id"].astype(str).tolist()

        # Embeddings
        print("   → Generating Embeddings...")
        embeddings = generate_sentence_embeddings(texts, batch_size=16)

        # Topic Modeling
        print("   → Running BERTopic...")
        topic_assignments, topic_summary = discover_topics(
            review_ids=ids,
            documents=texts,
            embeddings=embeddings,
        )
        _ = topic_summary

        # Clustering
        print("   → Running HDBSCAN Clustering...")
        cluster_df = perform_hdbscan_clustering(ids, embeddings)

        print("\n✅ Topic Assignments (Sample):")
        ta = topic_assignments.head(5).to_dict(orient="records")  # type: ignore[reportUnknownMemberType]
        pprint(ta)

        print("\n✅ Cluster Assignments (Sample):")
        cd = cluster_df.head(5).to_dict(orient="records")  # type: ignore[reportUnknownMemberType]
        pprint(cd)

    else:
        print("ℹ️  No complex reviews to process with AI.")

    # ================== Simple Reviews ==================
    # Build simple mask similarly to the complex mask to avoid DataFrame.get overloads
    if "is_complex" in pre_df.columns:
        mask_simple = cast(pd.Series, pre_df["is_complex"])  # type: ignore[reportUnknownMemberType]
    else:
        mask_simple = cast(pd.Series, pd.Series([False] * len(pre_df)))  # type: ignore[reportUnknownMemberType]
    simple_df = pre_df[mask_simple == False]
    if len(simple_df) > 0:
        print(f"\n📌 Rule-based Results for {len(simple_df)} Simple Reviews:")
        cols = ["review_id", "rule_topics", "rule_tags", "rule_summary"]
        available = [c for c in cols if c in simple_df.columns]
        if available:
            simple_preview = simple_df[available].head(5).to_dict(orient="records")  # type: ignore[reportUnknownMemberType]
            pprint(simple_preview)

    print("\n🎉 Dry Run Completed Successfully!")


if __name__ == "__main__":
    run_dry()