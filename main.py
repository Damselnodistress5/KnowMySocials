"""
KnowMySocials NLP Pipeline - Main Orchestrator
Complete pipeline: Data Collection -> Ingestion -> Preprocessing
Run this file to execute the full pipeline.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.utils import setup_logger
from src.collection import collect_data
from src.ingestion import ingest_data, DatabaseIngester
from src.preprocessing import preprocess_data, extract_keywords
_ = extract_keywords  # ensure imported symbol is referenced for static checkers
from typing import Any, Dict, List, cast
import pandas as pd
import numpy as np
from src.embedding import generate_sentence_embeddings
from src.topic_modeling import discover_topics
from src.clustering import perform_hdbscan_clustering
# (config already imported above)


# Setup main logger
logger = setup_logger(__name__)


def print_banner(text: str):
    """Print decorative banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def main():
    """
    Main pipeline orchestrator.
    Executes: Collection -> Ingestion -> Preprocessing
    """
    try:
        print_banner("KNOWMYSOCIALS NLP PIPELINE - STARTING")
        
        # ============================================
        # PHASE 1: DATA COLLECTION
        # ============================================
        print_banner("PHASE 1: DATA COLLECTION")
        raw_data = collect_data(config.CSV_FILE_PATH)
        print(f"✓ Collected {len(raw_data)} reviews")
        mem_sum = float(cast(Any, raw_data.memory_usage(deep=True).sum()))  # type: ignore[reportUnknownMemberType]
        print(f"✓ Memory usage: {mem_sum / 1024**2:.2f} MB")
        
        # ============================================
        # PHASE 2: DATA INGESTION
        # ============================================
        print_banner("PHASE 2: DATA INGESTION - RAW DATA")
        ingest_data(raw_data)
        print(f"✓ Ingested raw reviews into PostgreSQL")
        
        ingester = DatabaseIngester()
        raw_count = ingester.get_record_count(config.RAW_DATA_TABLE)
        print(f"✓ Total records in {config.RAW_DATA_TABLE}: {raw_count}")
        
        # ============================================
        # PHASE 3: PREPROCESSING
        # ============================================
        print_banner("PHASE 3: PREPROCESSING")
        preprocessed_data = preprocess_data(raw_data)
        print(f"✓ Preprocessed {len(preprocessed_data)} reviews")

        # Count simple vs complex (hybrid) reviews
        simple_count = 0
        complex_count = 0
        if "is_complex" in preprocessed_data.columns:
            simple_count = int((preprocessed_data["is_complex"] == False).sum())
            complex_count = int((preprocessed_data["is_complex"] == True).sum())
        else:
            # If hybrid not in place, assume all complex
            complex_count = len(preprocessed_data)

        logger.info(f"Hybrid workflow: {simple_count} fast-path reviews, {complex_count} AI-path reviews")
        print(f"✓ Hybrid routing: {simple_count} fast-path, {complex_count} AI-path")
        
        # ============================================
        # PHASE 4: INGEST PREPROCESSED DATA
        # ============================================
        print_banner("PHASE 4: DATA INGESTION - PREPROCESSED DATA")
        
        # Ingest preprocessed reviews (includes hybrid markers)
        preprocessed_count = ingester.ingest_preprocessed_data(preprocessed_data)
        print(f"✓ Ingested {preprocessed_count} preprocessed reviews")
        
        # Extract metadata for ingestion
        metadata_df = preprocessed_data[[
            'review_id', 'text_length', 'token_count', 
            'unique_tokens', 'language_detected', 'hinglish_score'
        ]].copy()
        
        metadata_count = ingester.ingest_metadata(metadata_df)
        print(f"✓ Ingested metadata for {metadata_count} reviews")
        
        # ============================================
        # PHASE 5: EMBEDDING GENERATION
        # ============================================
        print_banner("PHASE 5: EMBEDDING GENERATION")
        # Embeddings & AI steps: only for complex reviews when hybrid is enabled
        complex_db = ingester.fetch_complex_preprocessed_reviews()
        if len(complex_db) > 0:
            complex_review_ids = complex_db["review_id"].astype(str).tolist()
            complex_texts = complex_db["preprocessed_text"].astype(str).tolist()
        else:
            complex_review_ids = []
            complex_texts = []
        if len(complex_review_ids) > 0:
            embeddings = generate_sentence_embeddings(
                complex_texts,
                batch_size=64,
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            # ensure numeric ndarray for downstream tools
            embeddings = np.asarray(embeddings, dtype=float)
            embedding_count = ingester.ingest_embeddings(complex_review_ids, embeddings)
            print(f"✓ Ingested {embedding_count} complex review embeddings")
        else:
            embeddings = []
            print("✓ No complex reviews to embed")

        # ============================================
        # PHASE 6: TOPIC DISCOVERY
        # ============================================
        print_banner("PHASE 6: TOPIC DISCOVERY")
        # Topic discovery: AI path for complex reviews
        topic_summary = []

        if len(complex_review_ids) > 0:
            complex_embeddings = np.asarray(embeddings, dtype=float)
            topic_assignments, topic_summary = discover_topics(
                review_ids=complex_review_ids,
                documents=complex_texts,
                embeddings=complex_embeddings
            )
            # topic_assignments contains only complex-topic rows
            topic_count = ingester.ingest_topic_assignments(topic_assignments)
            summary_count = ingester.ingest_topic_summary(topic_summary)
            print(f"✓ Ingested {topic_count} complex topic assignments")
            print(f"✓ Stored {summary_count} discovered topics")
        else:
            print("✓ No complex reviews for topic discovery")

        # Rule-based topic assignments for simple reviews
        simple_assignments: List[Dict[str, Any]] = []
        if "is_complex" in preprocessed_data.columns:
            simple_df = preprocessed_data[preprocessed_data["is_complex"] == False]
        else:
            simple_df = preprocessed_data.iloc[0:0]

        # Iterate over plain dict records to avoid pandas Series typing
        simple_records = cast(List[Dict[str, Any]], cast(Any, simple_df.to_dict(orient="records"))) if len(simple_df) > 0 else []  # type: ignore[reportUnknownMemberType]
        for row in simple_records:
            # Use rule_summary as topic_label with topic_id -1
            topic_row: Dict[str, Any] = {
                "review_id": row.get("review_id"),
                "topic_id": -1,
                "topic_probability": 1.0,
                "topic_label": row.get("rule_summary", "rule_based")
            }
            simple_assignments.append(topic_row)

        if simple_assignments:
            simple_df_topics = pd.DataFrame(simple_assignments)
            ingester.ingest_topic_assignments(simple_df_topics)
            print(f"✓ Ingested {len(simple_df_topics)} rule-based topic assignments")

        # ============================================
        # PHASE 7: CLUSTERING
        # ============================================
        print_banner("PHASE 7: CLUSTERING")
        if len(complex_review_ids) > 0:
            complex_embeddings = np.asarray(embeddings, dtype=float)
            cluster_df = perform_hdbscan_clustering(complex_review_ids, complex_embeddings)
            cluster_count = ingester.ingest_cluster_assignments(cluster_df)
            print(f"✓ Ingested {cluster_count} complex cluster assignments")
        else:
            # No complex reviews: mark simple reviews as noise clusters (-1)
            simple_cluster_rows: List[Dict[str, Any]] = []
            simple_records = cast(List[Dict[str, Any]], cast(Any, simple_df.to_dict(orient="records"))) if len(simple_df) > 0 else []  # type: ignore[reportUnknownMemberType]
            for row in simple_records:
                simple_cluster_rows.append({
                    "review_id": row.get("review_id"),
                    "cluster_label": -1,
                    "cluster_probability": 1.0,
                    "is_noise": True
                })
            if simple_cluster_rows:
                cluster_df = pd.DataFrame(simple_cluster_rows)
                cluster_count = ingester.ingest_cluster_assignments(cluster_df)
                print(f"✓ Ingested {cluster_count} rule-based cluster assignments")
            else:
                print("✓ No reviews to cluster")

        # ============================================
        # FINAL SUMMARY
        # ============================================
        print_banner("PIPELINE EXECUTION SUMMARY")
        
        final_raw_count = ingester.get_record_count(config.RAW_DATA_TABLE)
        final_preprocessed_count = ingester.get_record_count(config.PREPROCESSED_DATA_TABLE)
        final_metadata_count = ingester.get_record_count(config.METADATA_TABLE)
        final_embedding_count = ingester.get_record_count(config.EMBEDDINGS_TABLE)
        final_topic_count = ingester.get_record_count(config.TOPIC_ASSIGNMENTS_TABLE)
        final_cluster_count = ingester.get_record_count(config.CLUSTER_ASSIGNMENTS_TABLE)
        
        print(f"Raw Reviews Table: {final_raw_count} records")
        print(f"Preprocessed Reviews Table: {final_preprocessed_count} records")
        print(f"Metadata Table: {final_metadata_count} records")
        print(f"Embeddings Table: {final_embedding_count} records")
        print(f"Topic Assignments Table: {final_topic_count} records")
        print(f"Cluster Assignments Table: {final_cluster_count} records")
        
        # Print sample preprocessing results
        print("\n" + "=" * 70)
        print("SAMPLE PREPROCESSED REVIEW (First record)")
        print("=" * 70)
        
        if len(preprocessed_data) > 0:
            sample = cast(Dict[str, Any], cast(Any, preprocessed_data.iloc[0].to_dict()))  # type: ignore[reportUnknownMemberType]
            print(f"Review ID: {sample.get('review_id')}")
            print(f"Model: {sample.get('model_name')}")
            print(f"\nOriginal Text:")
            print(f"  {str(sample.get('original_text',''))[:150]}...")
            print(f"\nCleaned Text:")
            print(f"  {str(sample.get('cleaned_text',''))[:150]}...")
            print(f"\nTokens ({sample.get('token_count',0)} total, {sample.get('unique_tokens',0)} unique):")
            tokens_preview = str(sample.get('tokens', '')).split()[:20]
            print(f"  {' '.join(tokens_preview)}...")
            print(f"\nLemmas:")
            lemmas_preview = str(sample.get('lemmas', '')).split()[:20]
            print(f"  {' '.join(lemmas_preview)}...")
            print(f"\nMetadata:")
            print(f"  Text Length: {sample.get('text_length')} characters")
            print(f"  Hinglish Score: {sample.get('hinglish_score')}%")
        
        print_banner("PIPELINE COMPLETED SUCCESSFULLY ✓")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        print_banner("PIPELINE EXECUTION FAILED ✗")
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
