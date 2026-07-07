"""
Sentiment Analysis Module
Performs sentiment classification using VADER (fast-path) + embedding similarity.
Extracts aspect-based sentiment from reviews.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, cast
from numpy.typing import NDArray

try:
    import nltk  # type: ignore[import]
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore[import-not-found]
except Exception:
    nltk = None  # type: ignore[assignment]
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]

from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-not-found]
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from src.embedding import fallback_sentence_embeddings, generate_sentence_embeddings
from src.utils import setup_logger

logger = setup_logger(__name__)

NEGATIVE_SENTIMENT_KEYWORDS = [
    # Focused negative keywords (English + common Hindi/Hinglish romanized + Devanagari)
    "bekar", "kharab", "disappointed", "waste", "problem", "issue", "rust",
    "heat", "vibration", "loose", "hard", "expensive", "pathetic", "frustrating",
    "weak", "noise", "uncomfortable", "dim", "shame", "headache", "nightmare",
    "terrible", "worst", "bad", "poor", "cheap", "rude", "टूट", "टूटा", "बेकार",
    "खराब"
]

POSITIVE_SENTIMENT_KEYWORDS = [
    # Compact positive keywords
    "great", "excellent", "amazing", "love", "fantastic", "perfect", "good",
    "happy", "smooth", "reliable", "fast", "clean", "superb", "comfortable",
    "satisfied", "impressed", "wonderful", "best", "awesome", "accha", "achha",
]

# Boost and threshold tuning
NEGATIVE_BOOST_STEP = 0.18
NEGATIVE_BOOST_CAP = 0.60
POSITIVE_BOOST_STEP = 0.08
POSITIVE_BOOST_CAP = 0.25

# Binary decision threshold: combined score >= POSITIVE_THRESHOLD => Positive, else Negative
POSITIVE_THRESHOLD = 0.05
STRONG_NEGATIVE_COMPOUND = -0.05


class SentimentAnalyzer:
    """Performs hybrid sentiment analysis (VADER + embedding similarity)."""

    def __init__(self):
        """Initialize sentiment analyzer with VADER and reference embeddings."""
        self.logger = setup_logger(__name__)
        # VADER sentiment analyzer instance (if available)
        self.sia: Optional[Any] = None
        self._initialize_vader()

        # Build stable sentiment reference embeddings from anchor phrases.
        # This gives the embedding similarity stage a meaningful negative/positive axis.
        self.positive_embedding, self.negative_embedding = self._initialize_reference_embeddings()
        # typed embeddings
        # type: ignore[assignment]
        self.positive_embedding: NDArray[np.float32]
        self.negative_embedding: NDArray[np.float32]

    def _initialize_vader(self) -> None:
        """Attempt to initialize VADER and fallback to neutral if the resource is unavailable."""
        if nltk is None or SentimentIntensityAnalyzer is None:
            self.logger.warning("NLTK is not available; VADER sentiment will fall back to neutral.")
            self.sia = None
            return

        try:
            data_dir = str(config.NLTK_DATA_DIR)
            nltk_data = getattr(nltk, "data", None)
            vader_available = False

            if nltk_data is not None:
                data_path = getattr(nltk_data, "path", None)
                if isinstance(data_path, list) and data_dir not in data_path:
                    setattr(nltk_data, "path", [data_dir] + data_path)
                self.logger.info(f"NLTK data path: {data_dir}")

                find_fn = getattr(nltk_data, "find", None)
                if callable(find_fn):
                    try:
                        find_fn("sentiment/vader_lexicon")  # type: ignore[reportUnknownMemberType]
                        vader_available = True
                    except LookupError:
                        self.logger.warning("NLTK vader_lexicon not found; VADER sentiment will fall back to neutral.")

            if vader_available:
                self.sia = SentimentIntensityAnalyzer()
            else:
                self.logger.warning("VADER resource unavailable; falling back to neutral sentiment.")
                self.sia = None
        except Exception as exc:
            self.logger.warning("Could not initialize VADER sentiment analyzer; falling back to neutral sentiment.")
            self.logger.warning(str(exc))
            self.sia = None

    def _initialize_reference_embeddings(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Create positive and negative reference embeddings from sentiment anchor phrases."""
        positive_anchors = [
            "This product was excellent and I loved it.",
            "I am very satisfied with the service and experience.",
            "The performance was great and reliable.",
            "Everything worked perfectly and smoothly."
        ]
        negative_anchors = [
            "This product was terrible and disappointing.",
            "I experienced a serious issue and poor quality.",
            "The service was bad and unreliable.",
            "This was the worst experience and a major problem."
        ]

        text_vectors = positive_anchors + negative_anchors
        embeddings = None

        try:
            embeddings = generate_sentence_embeddings(text_vectors, batch_size=8, show_progress=False)
        except Exception as exc:
            self.logger.warning(
                "Failed to generate sentiment anchor embeddings with sentence-transformers; falling back to deterministic anchors."
            )
            self.logger.debug(str(exc))
            embeddings = fallback_sentence_embeddings(text_vectors, dim=384)

        pos_emb = np.asarray(embeddings[: len(positive_anchors)], dtype=float).mean(axis=0)
        neg_emb = np.asarray(embeddings[len(positive_anchors) :], dtype=float).mean(axis=0)
        pos_emb = pos_emb / (np.linalg.norm(pos_emb) + 1e-7)
        neg_emb = neg_emb / (np.linalg.norm(neg_emb) + 1e-7)

        return pos_emb.astype(np.float32), neg_emb.astype(np.float32)

    def _negative_keyword_boost(self, text: str) -> float:
        """Boost negative sentiment for explicit negative keywords in the review text."""
        if not text:
            return 0.0
        matches_list = self._negative_keyword_matches(text)
        matches = len(matches_list)

        if matches == 0:
            return 0.0

        boost = min(NEGATIVE_BOOST_CAP, matches * NEGATIVE_BOOST_STEP)
        self.logger.debug(
            "Negative keyword boost: %s matches (%s) -> %s",
            matches,
            ",".join(matches_list[:8]),
            round(boost, 3),
        )
        return boost

    def _negative_keyword_matches(self, text: str) -> List[str]:
        """Return list of negative keywords found in text (deduped, lowercased)."""
        if not text:
            return []
        text_lower = text.lower()
        found: List[str] = []
        for kw in NEGATIVE_SENTIMENT_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                found.append(kw)
        # deduplicate while preserving order
        seen: Set[str] = set()
        deduped: List[str] = []
        for w in found:
            if w not in seen:
                deduped.append(w)
                seen.add(w)
        return deduped

    def _positive_keyword_boost(self, text: str) -> float:
        """Boost positive sentiment for explicit positive keywords in the review text."""
        if not text:
            return 0.0

        text_lower = text.lower()
        matches = 0
        for kw in POSITIVE_SENTIMENT_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                matches += 1

        if matches == 0:
            return 0.0

        boost = min(POSITIVE_BOOST_CAP, matches * POSITIVE_BOOST_STEP)
        self.logger.debug(
            "Positive keyword boost: %s matches -> %s",
            matches,
            round(boost, 3),
        )
        return boost

    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Perform VADER sentiment analysis (fast-path, rule-based).
        
        Args:
            text: Review text to analyze
            
        Returns:
            Dict with {compound, positive, negative, neutral} scores
        """
        if self.sia is None:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0}

        try:
            scores = self.sia.polarity_scores(text)  # type: ignore[reportUnknownMemberType]
            # keep only pos, neg, compound
            return {"compound": float(scores.get("compound", 0.0)), "pos": float(scores.get("pos", 0.0)), "neg": float(scores.get("neg", 0.0))}
        except Exception as exc:
            self.logger.warning(f"VADER sentiment failed: {exc}")
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    def analyze_sentiment_embedding(self, embedding: Optional[Any]) -> Dict[str, float]:
        """
        Perform embedding-based sentiment analysis.
        Compute similarity to pre-computed positive/negative reference embeddings.
        
        Args:
            embedding: Sentence embedding (numpy array, shape (384,) or similar)
            
        Returns:
            Dict with {positive_score, negative_score}
        """
        if embedding is None:
            return {"positive_score": 0.0, "negative_score": 0.0}

        # support serialized bytes -> numpy buffer
        if isinstance(embedding, (bytes, bytearray)):
            try:
                embedding = np.frombuffer(embedding, dtype=np.float32)
            except Exception:
                return {"positive_score": 0.0, "negative_score": 0.0}
            return {"positive_score": 0.0, "negative_score": 0.0}

        try:
            emb_arr = np.asarray(embedding, dtype=float)
            if emb_arr.size == 0:
                return {"positive_score": 0.0, "negative_score": 0.0}
            emb_norm = emb_arr / (np.linalg.norm(emb_arr) + 1e-7)
            # ensure 2D arrays for sklearn cosine_similarity
            pos_sim = float(cosine_similarity(np.atleast_2d(emb_norm), np.atleast_2d(self.positive_embedding))[0, 0])
            neg_sim = float(cosine_similarity(np.atleast_2d(emb_norm), np.atleast_2d(self.negative_embedding))[0, 0])

            # map cosine [-1,1] -> [0,1]
            pos_sim = (pos_sim + 1) / 2
            neg_sim = (neg_sim + 1) / 2

            # Strengthen negative embedding signal slightly (scale)
            neg_sim = min(1.0, neg_sim * 1.12)
            pos_sim = min(1.0, pos_sim * 1.0)

            return {"positive_score": float(pos_sim), "negative_score": float(neg_sim)}
        except Exception as e:
            self.logger.warning(f"Error in embedding-based sentiment: {e}")
            return {"positive_score": 0.0, "negative_score": 0.0}

    def hybrid_sentiment_label(
        self,
        vader_scores: Dict[str, float],
        embedding_scores: Dict[str, float],
        text: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Combine VADER and embedding scores to get final sentiment label.

        Args:
            vader_scores: Output from VADER
            embedding_scores: Output from embedding similarity
            text: Optional raw review text for negative keyword boosting

        Returns:
            Sentiment label: "positive", "negative", or "neutral"
        """
        text = text or ""
        pos_emb = embedding_scores.get("positive_score", 0.0)
        neg_emb = embedding_scores.get("negative_score", 0.0)

        # Prefer explicit VADER pos/neg scores when available.
        vader_pos_score = float(vader_scores.get("pos", 0.0))
        vader_neg_score = float(vader_scores.get("neg", 0.0))
        compound = float(vader_scores.get("compound", 0.0))

        # language-aware VADER weighting
        lang = (language or "").lower()
        if self.sia is None:
            vader_weight = 0.0
        else:
            if lang in ("hi", "hindi"):
                vader_weight = 0.0
            elif lang == "mixed":
                vader_weight = 0.15
            else:
                vader_weight = 0.6

        negative_boost = self._negative_keyword_boost(text)
        positive_boost = self._positive_keyword_boost(text)

        emb_weight = max(0.0, 1.0 - vader_weight)

        # Combined positive and negative signals
        combined_pos = vader_weight * vader_pos_score + (emb_weight * 0.5) * pos_emb + positive_boost
        combined_neg = vader_weight * vader_neg_score + (emb_weight * 0.5) * neg_emb + negative_boost

        # Single scalar score: positive minus negative plus a small compound influence
        combined_score = float(combined_pos - combined_neg + 0.5 * vader_weight * compound)

        # If a strong negative compound exists, bias towards negative
        if compound <= STRONG_NEGATIVE_COMPOUND:
            combined_score -= 0.08

        label = "positive" if combined_score >= POSITIVE_THRESHOLD else "negative"

        # Log negative detection cases
        if label == "negative" or negative_boost > 0:
            self.logger.info(
                "Sentiment (BIN): review=%s compound=%.3f vader_pos=%.3f vader_neg=%.3f pos_emb=%.3f neg_emb=%.3f neg_boost=%.3f pos_boost=%.3f combined_score=%.3f label=%s",
                (text[:80] + "...") if len(text) > 80 else text or "N/A",
                compound,
                vader_pos_score,
                vader_neg_score,
                pos_emb,
                neg_emb,
                negative_boost,
                positive_boost,
                combined_score,
                label,
            )

        return label

    def extract_aspects(self, text: str, tokens: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract aspect-based sentiments from review text.
        Uses keyword matching (fast-path) for common automotive aspects.
        
        Args:
            text: Review text
            tokens: Optional list of tokens (for efficiency)
            
        Returns:
            List of aspect dicts: [{"aspect": str, "sentiment": str, "score": float}, ...]
        """
        aspects = []
        text_lower = text.lower()
        
        # Keyword-based aspect extraction
        for aspect_name, keywords in config.ASPECT_KEYWORDS.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            
            if found_keywords:
                # Compute aspect-specific sentiment
                # Extract sentence/context around keywords for more accurate sentiment
                context_words = []
                for kw in found_keywords:
                    idx = text_lower.find(kw)
                    if idx >= 0:
                        start = max(0, idx - 50)
                        end = min(len(text), idx + len(kw) + 50)
                        context = text[start:end]
                        context_words.append(context)  # type: ignore[reportUnknownMemberType]
                
                aspect_context = " ".join(context_words)  # type: ignore[reportUnknownArgumentType]
                vader_scores = self.analyze_sentiment_vader(aspect_context)
                
                # Map compound score to sentiment label
                compound = vader_scores.get("compound", 0.0)
                if compound >= 0.05:
                    aspect_sentiment = "positive"
                    aspect_score = vader_scores.get("pos", 0.0)
                elif compound <= -0.05:
                    aspect_sentiment = "negative"
                    aspect_score = vader_scores.get("neg", 0.0)
                else:
                    aspect_sentiment = "positive"
                    aspect_score = vader_scores.get("pos", 0.0)
                
                aspects.append({  # type: ignore[reportUnknownMemberType]
                    "aspect": aspect_name,
                    "sentiment": aspect_sentiment,
                    "score": float(aspect_score),
                    "keywords_found": found_keywords
                })
        
        return aspects  # type: ignore[return-value]


def analyze_sentiment_batch(
    df: pd.DataFrame,
    analyzer: Optional[SentimentAnalyzer] = None,
    use_parallel: bool = config.USE_PARALLEL_PROCESSING
) -> pd.DataFrame:
    """
    Perform sentiment analysis on a batch of reviews.
    
    Args:
        df: DataFrame with columns [review_id, preprocessed_text, tokens, embedding]
        analyzer: SentimentAnalyzer instance (created if None)
        use_parallel: Use ThreadPoolExecutor for parallel analysis
        
    Returns:
        DataFrame with added columns [sentiment, aspects]
    """
    if analyzer is None:
        analyzer = SentimentAnalyzer()
    
    logger.info(f"Performing sentiment analysis on {len(df)} reviews...")
    
    results: Dict[str, Any] = {
        "review_id": [],
        "sentiment": [],
        "aspects": []
    }
    
    records = cast(List[Dict[str, Any]], df.to_dict(orient="records"))
    
    def analyze_single_review(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Analyze sentiment for a single review."""
        review_id = row.get("review_id")
        review_id = str(review_id) if review_id is not None else ""
        text = row.get("preprocessed_text", "")
        tokens = row.get("tokens", [])

        try:
            # VADER analysis
            vader_scores = analyzer.analyze_sentiment_vader(text)

            # Embedding-based analysis (if embedding available)
            embedding = row.get("embedding")
            embedding_scores: Dict[str, float] = {}
            if embedding is not None and isinstance(embedding, (np.ndarray, bytes)):
                try:
                    if isinstance(embedding, bytes):
                        embedding = np.frombuffer(embedding, dtype=np.float32)
                    embedding_scores = analyzer.analyze_sentiment_embedding(embedding)  # type: ignore[reportUnknownArgumentType]
                except Exception:
                    embedding_scores = {"positive_score": 0.0, "negative_score": 0.0}

            # Determine language if available from preprocessing and pass through
            language = row.get("language_detected")

            # Determine label and score (pass language for language-aware weighting)
            label = analyzer.hybrid_sentiment_label(vader_scores, embedding_scores, text=text, language=language)
            
            sentiment_dict: Dict[str, Any] = {
                "label": label,
                "language_detected": language,
                "vader_compound": float(vader_scores.get("compound", 0.0)),
                "vader_positive": float(vader_scores.get("pos", 0.0)),
                "vader_negative": float(vader_scores.get("neg", 0.0)),
                "vader_neutral": float(vader_scores.get("neu", 0.0)),
                "embedding_positive": float(embedding_scores.get("positive_score", 0.0)),
                "embedding_negative": float(embedding_scores.get("negative_score", 0.0))
            }
            
            # Extract aspects
            aspects_list = analyzer.extract_aspects(text, tokens)
            
            return review_id, sentiment_dict, aspects_list  # type: ignore[return-value]
        
        except Exception as e:
            logger.warning(f"Error analyzing sentiment for {review_id}: {e}")
            return review_id, {"label": "negative"}, []  # type: ignore[return-value]
    
    if use_parallel and config.MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = [executor.submit(analyze_single_review, row) for row in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Sentiment Analysis"):
                try:
                    review_id, sentiment_dict, aspects_list = future.result()
                    results["review_id"].append(review_id)  # type: ignore[reportUnknownMemberType]
                    results["sentiment"].append(sentiment_dict)  # type: ignore[reportUnknownMemberType]
                    results["aspects"].append(aspects_list)  # type: ignore[reportUnknownMemberType]
                except Exception as e:
                    logger.error(f"Failed to retrieve sentiment result: {e}")
    else:
        for row in tqdm(records, total=len(records), desc="Sentiment Analysis"):
            review_id, sentiment_dict, aspects_list = analyze_single_review(row)
            results["review_id"].append(review_id)  # type: ignore[reportUnknownMemberType]
            results["sentiment"].append(sentiment_dict)  # type: ignore[reportUnknownMemberType]
            results["aspects"].append(aspects_list)  # type: ignore[reportUnknownMemberType]
    
    result_df = pd.DataFrame(results)
    logger.info(f"Sentiment analysis completed for {len(result_df)} reviews")
    
    return result_df
