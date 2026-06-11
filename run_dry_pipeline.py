"""Dry-run pipeline for KnowMySocials.
Runs preprocessing, embedding, topic discovery, and clustering in-memory
without writing to the database. Useful for testing the hybrid routing.
"""
from pprint import pprint

import config
import pandas as pd

from src.preprocessing import preprocess_data
from src.embedding import generate_sentence_embeddings
from src.topic_modeling import discover_topics
from src.clustering import perform_hdbscan_clustering


def run_dry():
    print("Loading sample CSV:", config.CSV_FILE_PATH)
    df = pd.read_csv(config.CSV_FILE_PATH)
    print(f"Loaded {len(df)} rows")

    pre_df = preprocess_data(df)
    print(f"Preprocessed rows: {len(pre_df)}")

    # Show hybrid routing counts
    if "is_complex" in pre_df.columns:
        simple = int((pre_df["is_complex"] == False).sum())
        complex_n = int((pre_df["is_complex"] == True).sum())
    else:
        simple = 0
        complex_n = len(pre_df)

    print(f"Hybrid routing: {simple} simple, {complex_n} complex")

    # Show a sample of preprocessed results
    print("\nSample preprocessed rows:")
    pprint(pre_df.head(5).to_dict(orient="records"))

    # AI path for complex reviews
    complex_df = pre_df[pre_df.get("is_complex", True) == True]
    if len(complex_df) > 0:
        texts = complex_df["preprocessed_text"].tolist()
        ids = complex_df["review_id"].tolist()
        print(f"\nGenerating embeddings for {len(ids)} complex reviews...")
        embeddings = generate_sentence_embeddings(texts, batch_size=8, show_progress=False)
        print("Embeddings shape:", embeddings.shape)

        print("\nRunning topic discovery (BERTopic) on complex reviews...")
        topic_assignments, topic_summary = discover_topics(review_ids=ids, documents=texts, embeddings=embeddings)
        print("Topic assignments:")
        pprint(topic_assignments.to_dict(orient="records"))
        print("\nTopic summary:")
        pprint(topic_summary.to_dict(orient="records"))

        print("\nRunning HDBSCAN clustering on complex embeddings...")
        cluster_df = perform_hdbscan_clustering(ids, embeddings)
        print("Cluster assignments:")
        pprint(cluster_df.to_dict(orient="records"))
    else:
        print("No complex reviews to run AI pipeline on.")

    # Rule-based outputs for simple reviews
    simple_df = pre_df[pre_df.get("is_complex", False) == False]
    if len(simple_df) > 0:
        print("\nRule-based summaries for simple reviews:")
        pprint(simple_df[["review_id", "rule_topics", "rule_tags", "rule_summary"]].to_dict(orient="records"))


if __name__ == "__main__":
    run_dry()
