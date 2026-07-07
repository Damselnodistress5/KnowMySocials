"""
Summarization Module
Generates abstractive summaries using BART or T5 with zero-shot classification.
Includes batch processing and caching for efficiency.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, cast
from transformers import pipeline  # type: ignore[import-not-found]
from tqdm import tqdm
import hashlib

import config
from src.utils import setup_logger

logger = setup_logger(__name__)


class TextSummarizer:
    """Generates abstractive summaries using transformer models (BART/T5)."""

    def __init__(self, model_name: str = config.SUMMARIZATION_MODEL):
        """
        Initialize summarizer with specified model.
        
        Args:
            model_name: HuggingFace model name (e.g., "facebook/bart-large-cnn")
        """
        self.logger = setup_logger(__name__)
        self.model_name = model_name
        
        try:
            self.logger.info(f"Loading summarization model: {model_name}")
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if self._cuda_available() else -1  # type: ignore
            )
            self.logger.info("Summarization model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load summarization model: {e}")
            self.summarizer = None
        
        # Simple cache: review_text_hash -> summary
        self._summary_cache: Dict[str, str] = {}

    @staticmethod
    def _cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _hash_text(self, text: str) -> str:
        """Compute hash of text for caching."""
        return hashlib.md5(text.encode()).hexdigest()

    def summarize_single(
        self, text: str, use_cache: bool = config.CACHE_SUMMARIES
    ) -> Optional[str]:
        """
        Generate summary for a single review.
        
        Args:
            text: Review text to summarize
            use_cache: Use cached summary if available
            
        Returns:
            Summary string or None if summarization failed
        """
        if not text or not self.summarizer:
            return None
        
        # Check cache
        text_hash = self._hash_text(text)
        if use_cache and text_hash in self._summary_cache:
            return self._summary_cache[text_hash]
        
        try:
            # Truncate to reasonable length for transformer (max ~1024 tokens)
            text_truncated = text[:1500]
            
            result = self.summarizer(
                text_truncated,
                max_length=config.SUMMARY_MAX_LENGTH,
                min_length=config.SUMMARY_MIN_LENGTH,
                do_sample=False,
                truncation=True
            )
            
            summary = result[0]["summary_text"] if result else None
            
            # Cache result
            if summary and use_cache:
                self._summary_cache[text_hash] = summary
            
            return summary
        
        except Exception as e:
            self.logger.warning(f"Error summarizing text: {e}")
            return None

    def summarize_batch(self, texts: List[str], batch_size: int = config.SUMMARIZATION_BATCH_SIZE) -> List[Optional[str]]:
        """
        Generate summaries for multiple reviews in batches (more efficient).
        
        Args:
            texts: List of review texts
            batch_size: Batch size for processing
            
        Returns:
            List of summaries (None for failed summaries)
        """
        if not self.summarizer:
            return [None] * len(texts)
        
        summaries = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Summarizing batch"):
            batch = texts[i : i + batch_size]
            
            try:
                # Truncate texts
                batch_truncated = [t[:1500] for t in batch]
                
                results = self.summarizer(
                    batch_truncated,
                    max_length=config.SUMMARY_MAX_LENGTH,
                    min_length=config.SUMMARY_MIN_LENGTH,
                    do_sample=False,
                    truncation=True,
                    batch_size=batch_size
                )
                
                for result in results:
                    summary = cast(Any, result).get("summary_text")  # type: ignore[reportUnknownArgumentType]
                    summaries.append(summary)  # type: ignore[reportUnknownMemberType]
                    
            except Exception as e:
                self.logger.warning(f"Error in batch summarization: {e}")
                summaries.extend([None] * len(batch))  # type: ignore[reportUnknownMemberType]
        
        return summaries  # type: ignore[return-value]


class ZeroShotAspectClassifier:
    """Classify review aspects using zero-shot classification."""

    def __init__(self):
        """Initialize zero-shot classifier."""
        self.logger = setup_logger(__name__)
        
        try:
            self.logger.info("Loading zero-shot classification model")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self._cuda_available() else -1  # type: ignore
            )
            self.logger.info("Zero-shot classifier loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load zero-shot classifier: {e}")
            self.classifier = None

    @staticmethod
    def _cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def classify_aspect(self, text: str, aspect_labels: List[str] = config.ZERO_SHOT_LABELS) -> Dict[str, float]:
        """
        Classify which aspects are most relevant to a review using zero-shot classification.
        
        Args:
            text: Review text
            aspect_labels: List of possible aspect labels
            
        Returns:
            Dict mapping aspect label to confidence score
        """
        if not self.classifier or not text:
            return {}
        
        try:
            text_truncated = text[:512]
            result = self.classifier(  # type: ignore[reportUnknownMemberType]
                text_truncated,
                aspect_labels,
                multi_class=True
            )
            
            # Map scores
            scores: Dict[str, float] = {}
            for label, score in zip(result.get("labels", []), result.get("scores", [])):  # type: ignore[reportUnknownMemberType]
                scores[label] = float(score)  # type: ignore[reportUnknownArgumentType]
            
            return scores
        
        except Exception as e:
            self.logger.warning(f"Error in zero-shot classification: {e}")
            return {}


def generate_summaries_batch(
    df: pd.DataFrame,
    summarizer: Optional[TextSummarizer] = None,
    use_parallel: bool = False
) -> pd.DataFrame:
    """
    Generate summaries for a batch of reviews.
    
    Args:
        df: DataFrame with column [preprocessed_text]
        summarizer: TextSummarizer instance (created if None)
        use_parallel: Use parallel processing (note: threading + transformers may cause issues)
        
    Returns:
        DataFrame with columns [review_id, summary, summary_length]
    """
    if summarizer is None:
        summarizer = TextSummarizer()
    
    logger.info(f"Generating summaries for {len(df)} reviews...")
    
    results: Dict[str, Any] = {
        "review_id": [],
        "summary": [],
        "summary_length": []
    }
    
    records = cast(Any, df.to_dict(orient="records"))  # type: ignore[reportUnknownMemberType]
    
    # Batch processing is more efficient for transformers
    texts = [row.get("preprocessed_text", "") for row in records]
    review_ids = [row.get("review_id") for row in records]
    
    summaries = summarizer.summarize_batch(texts, batch_size=config.SUMMARIZATION_BATCH_SIZE)
    
    for review_id, summary in zip(review_ids, summaries):
        results["review_id"].append(review_id)  # type: ignore[reportUnknownMemberType]
        results["summary"].append(summary if summary else "")  # type: ignore[reportUnknownMemberType]
        results["summary_length"].append(len(summary.split()) if summary else 0)  # type: ignore[reportUnknownMemberType]
    
    result_df = pd.DataFrame(results)
    logger.info(f"Summary generation completed for {len(result_df)} reviews")
    
    return result_df


def classify_aspects_batch(
    df: pd.DataFrame,
    classifier: Optional[ZeroShotAspectClassifier] = None
) -> pd.DataFrame:
    """
    Classify relevant aspects for a batch of reviews using zero-shot classification.
    
    Args:
        df: DataFrame with column [preprocessed_text]
        classifier: ZeroShotAspectClassifier instance (created if None)
        
    Returns:
        DataFrame with columns [review_id, aspect_scores]
    """
    if classifier is None:
        classifier = ZeroShotAspectClassifier()
    
    logger.info(f"Classifying aspects for {len(df)} reviews...")
    
    results: Dict[str, Any] = {
        "review_id": [],
        "aspect_scores": []
    }
    
    records = cast(Any, df.to_dict(orient="records"))  # type: ignore[reportUnknownMemberType]
    
    for row in tqdm(records, total=len(records), desc="Classifying aspects"):
        review_id = row.get("review_id")
        text = row.get("preprocessed_text", "")
        
        try:
            scores = classifier.classify_aspect(text)
            results["review_id"].append(review_id)  # type: ignore[reportUnknownMemberType]
            results["aspect_scores"].append(scores)  # type: ignore[reportUnknownMemberType]
        except Exception as e:
            logger.warning(f"Error classifying aspects for {review_id}: {e}")
            results["review_id"].append(review_id)  # type: ignore[reportUnknownMemberType]
            results["aspect_scores"].append({})  # type: ignore[reportUnknownMemberType]
    
    result_df = pd.DataFrame(results)
    logger.info(f"Aspect classification completed for {len(result_df)} reviews")
    
    return result_df
