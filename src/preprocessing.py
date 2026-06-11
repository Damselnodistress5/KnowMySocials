"""
Preprocessing Module
Comprehensive text preprocessing pipeline for NLP applications.
Includes cleaning, tokenization, lemmatization, and domain-specific normalization.
"""

# pyright: reportMissingTypeStubs=false
import re
import pandas as pd
from typing import Any, List, Tuple, Dict, Set, Optional
from tqdm import tqdm
import threading
import concurrent.futures

# NLP libraries
spacy_available = False
spacy: Any = None
try:
    import spacy  # type: ignore[import]
    spacy_available = True
except ImportError:
    spacy = None
    spacy_available = False

nltk_available = False
nltk: Any = None
stopwords: Any = None
word_tokenize: Any = None
try:
    import nltk  # type: ignore[import]
    from nltk.corpus import stopwords  # type: ignore[import]
    from nltk.tokenize import word_tokenize  # type: ignore[import]
    nltk_available = True
except ImportError:
    nltk = None
    stopwords = None
    word_tokenize = None
    nltk_available = False

import config
from src.utils import setup_logger


logger = setup_logger(__name__)

# Hybrid helpers: keyword extraction and lightweight rule-based analysis
def extract_keywords(text: str) -> List[str]:
    """Extract domain keywords from text using simple regex/keyword match.

    Returns a list of matched keywords (lowercased, unique).
    """
    if not text:
        return []
    text_lower = text.lower()
    found: List[str] = []
    for kw in config.HYBRID_KEYWORDS:
        # match word boundaries and also accept underscores (e.g., fuel_efficiency)
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            found.append(kw)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for k in found:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def is_complex_review(text: str, keywords: List[str]) -> bool:
    """Decide whether a review is complex enough for AI processing.

    Simple heuristic: if the number of distinct keywords exceeds a threshold,
    or if the text length is long, mark as complex.
    """
    if not config.USE_HYBRID_WORKFLOW:
        return True

    kw_count = len(keywords)
    # Consider length-based complexity too
    length_threshold = 200
    text_len = len(text or "")

    if kw_count >= config.HYBRID_COMPLEXITY_THRESHOLD:
        return True
    if text_len > length_threshold:
        return True
    return False


def rule_based_analysis(text: str, keywords: List[str]) -> Dict[str, Any]:
    """Perform lightweight rule-based tagging and rudimentary sentiment/topic guess.

    Returns a dict with keys: `rule_topics`, `rule_tags`, `rule_summary`.
    """
    tags: List[str] = []
    topics: List[str] = []

    # Map keywords to coarse topics/tags
    mapping = {
        "mileage": "fuel_efficiency",
        "fuel": "fuel_efficiency",
        "fuel_efficiency": "fuel_efficiency",
        "suspension": "ride_quality",
        "comfort": "ride_quality",
        "seat": "ride_quality",
        "service": "after_sales",
        "brake": "braking",
        "braking": "braking",
        "vibration": "nvh",
        "noise": "nvh",
        "heat": "thermal",
        "pickup": "performance",
        "storage": "practicality",
        "engine": "performance",
    }

    for kw in keywords:
        t = mapping.get(kw, None)
        if t and t not in topics:
            topics.append(t)
        # basic tags
        if kw in ("service", "maintenance"):
            tags.append("needs_service")
        if kw in ("vibration", "noise"):
            tags.append("nvv_issues")

    # Derive a short summary
    if topics:
        summary = "; ".join(topics)
    elif keywords:
        summary = ", ".join(keywords[:3])
    else:
        summary = "general"

    return {
        "rule_topics": topics,
        "rule_tags": list(set(tags)),
        "rule_summary": summary
    }


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline.
    Handles cleaning, tokenization, lemmatization, and normalization.
    """
    
    def __init__(self):
        """Initialize preprocessor with spaCy and NLTK models."""
        self.logger = setup_logger(__name__)
        
        # Load spaCy model
        self._load_spacy_model()
        
        # Download NLTK data
        self._setup_nltk()
        
        # Load custom stopwords
        self.stopwords = self._load_stopwords()
    
    def _load_spacy_model(self):
        """Load spaCy English model for lemmatization."""
        if not spacy_available:
            self.logger.warning("spaCy not available. Lemmatization disabled.")
            self.nlp = None
            return
        
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
            self.logger.info(f"Loaded spaCy model: {config.SPACY_MODEL}")
        except OSError:
            self.logger.warning(f"spaCy model '{config.SPACY_MODEL}' not found.")
            self.logger.info("Download it using: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        if not nltk_available:
            self.logger.warning("NLTK not available.")
            return
        
        # Ensure NLTK looks in the project-specific data directory first
        data_dir = str(config.NLTK_DATA_DIR)
        if data_dir not in nltk.data.path:
            nltk.data.path.insert(0, data_dir)
        self.logger.info(f"NLTK data path: {data_dir}")
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK 'punkt' tokenizer...")
            nltk.download('punkt', download_dir=data_dir, quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            self.logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', download_dir=data_dir, quiet=True)
    
    def _load_stopwords(self) -> Set[str]:
        """
        Load combined stopwords from NLTK and custom automotive stopwords.
        
        Returns:
            Set of stopwords
        """
        stopwords_set: Set[str] = set()
        
        # Add NLTK English stopwords
        if nltk_available:
            try:
                stopwords_set.update(stopwords.words('english'))
            except Exception as e:
                self.logger.warning(f"Could not load NLTK stopwords: {e}")
        
        # Add custom automotive domain stopwords
        stopwords_set.update(config.AUTOMOTIVE_STOPWORDS)
        
        self.logger.info(f"Loaded {len(stopwords_set)} total stopwords")
        return stopwords_set
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with HTML tags removed
        """
        html_pattern = r'<.*?>'
        return re.sub(html_pattern, '', text)
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters but keep basic punctuation.
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
        """
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r"[^\w\s\.\,\!\?\'\"-]", " ", text)
        return text
    
    def remove_extra_spaces(self, text: str) -> str:
        """
        Remove extra whitespace and normalize spaces.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized spaces
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        return text.strip()
    
    def normalize_hinglish(self, text: str) -> str:
        """
        Normalize Hinglish terms to English equivalents.
        Applies custom domain-specific mappings.
        
        Args:
            text: Input text (possibly containing Hinglish)
            
        Returns:
            Normalized text
        """
        text_lower = text.lower()
        
        # Apply Hinglish replacements
        for hinglish_term, english_term in config.HINGLISH_REPLACEMENTS.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(hinglish_term) + r'\b'
            text_lower = re.sub(pattern, english_term, text_lower, flags=re.IGNORECASE)
        
        return text_lower
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if nltk_available:
            try:
                tokens = word_tokenize(text)
                return tokens
            except Exception as e:
                self.logger.warning(f"NLTK tokenization failed: {e}. Using simple split.")
        
        # Fallback to simple split
        return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered token list
        """
        filtered_tokens = [
            token for token in tokens 
            if token.lower() not in self.stopwords and len(token) > 1
        ]
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Lemmatize tokens using spaCy.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tuple of (original_tokens, lemmatized_tokens)
        """
        if self.nlp is None:
            self.logger.warning("spaCy model not available. Returning original tokens.")
            return tokens, tokens
        
        try:
            # Join tokens back to text for spaCy processing
            text = ' '.join(tokens)
            doc = self.nlp(text)
            
            # Extract lemmas
            lemmas = [token.lemma_.lower() for token in doc]
            
            return tokens, lemmas
            
        except Exception as e:
            self.logger.warning(f"Lemmatization failed: {e}. Returning original tokens.")
            return tokens, tokens
    
    def clean_text(self, text: str) -> str:
        """
        Apply comprehensive text cleaning pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Remove special characters
        text = self.remove_special_characters(text)
        
        # Normalize Hinglish
        text = self.normalize_hinglish(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove extra spaces
        text = self.remove_extra_spaces(text)
        
        return text
    
    def preprocess_single_text(self, text: str) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary containing preprocessing results
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        original_tokens, lemmas = self.lemmatize_tokens(filtered_tokens)
        
        # Calculate metadata
        hinglish_score = self._calculate_hinglish_score(text)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "tokens": original_tokens,
            "lemmas": lemmas,
            "token_count": len(original_tokens),
            "unique_tokens": len(set(original_tokens)),
            "hinglish_score": hinglish_score
        }

    def hybrid_preprocess(self, text: str) -> Dict[str, Any]:
        """Hybrid preprocessing entrypoint.

        - Run fast regex/keyword extraction
        - Decide whether review is simple or complex
        - If simple: apply fast rule_based_analysis and return lightweight result
        - If complex: fall back to full `preprocess_single_text` for AI path
        """
        # Fast cleaning (regex-based) to normalize text for keyword extraction
        fast_clean = self.remove_urls(text)
        fast_clean = self.remove_html_tags(fast_clean)
        fast_clean = self.remove_special_characters(fast_clean)
        fast_clean = self.remove_extra_spaces(fast_clean).lower()

        # Extract keywords quickly
        keywords = extract_keywords(fast_clean)

        # Decide complexity
        complex_flag = is_complex_review(text, keywords)

        if not complex_flag:
            # Fast path: perform rule-based analysis and minimal tokenization
            kb = rule_based_analysis(fast_clean, keywords)
            tokens = self.tokenize_text(fast_clean)
            filtered = self.remove_stopwords(tokens)
            lemmas = filtered  # don't run spaCy on fast path

            return {
                "original_text": text,
                "cleaned_text": fast_clean,
                "tokens": filtered,
                "lemmas": lemmas,
                "token_count": len(filtered),
                "unique_tokens": len(set(filtered)),
                "hinglish_score": self._calculate_hinglish_score(text),
                "is_complex": False,
                "rule_topics": kb.get("rule_topics", []),
                "rule_tags": kb.get("rule_tags", []),
                "rule_summary": kb.get("rule_summary", "general")
            }

        # Complex -> full preprocessing
        result = self.preprocess_single_text(text)
        result["is_complex"] = True
        result["rule_topics"] = keywords
        result["rule_tags"] = []
        result["rule_summary"] = ""
        return result
    
    def _calculate_hinglish_score(self, text: str) -> int:
        """
        Calculate percentage of Hinglish terms in text.
        
        Args:
            text: Input text
            
        Returns:
            Hinglish score (0-100)
        """
        text_lower = text.lower()
        hinglish_count = 0
        
        for hinglish_term in config.HINGLISH_REPLACEMENTS.keys():
            pattern = r'\b' + re.escape(hinglish_term) + r'\b'
            matches = len(re.findall(pattern, text_lower, flags=re.IGNORECASE))
            hinglish_count += matches
        
        # Split text into words for scoring
        words = text.split()
        if len(words) == 0:
            return 0
        
        score = int((hinglish_count / len(words)) * 100)
        return min(score, 100)  # Cap at 100


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing function.
    Apply full preprocessing pipeline to DataFrame.
    
    Args:
        df: DataFrame with 'Review_Text' column
        
    Returns:
        DataFrame with preprocessing results
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PHASE")
    logger.info("=" * 60)
    
    try:
        logger.info(f"Preprocessing {len(df)} reviews...")

        # Thread-local storage so each worker thread gets its own TextPreprocessor
        thread_local = threading.local()

        def _process_item(item: Tuple[Any, pd.Series]) -> Tuple[Any, Optional[Dict[str, Any]], Optional[Exception]]:
            """Worker that ensures a per-thread TextPreprocessor and runs hybrid_preprocess.

            Returns tuple (idx, result_dict or None, exception or None)
            """
            idx, row = item
            try:
                preproc = getattr(thread_local, "preprocessor", None)
                if preproc is None:
                    # instantiate a TextPreprocessor per thread to avoid sharing spaCy models
                    thread_local.preprocessor = TextPreprocessor()
                    preproc = thread_local.preprocessor

                review_text = row.get("Review_Text", "")
                result = preproc.hybrid_preprocess(review_text)

                # Add metadata
                result["review_id"] = row.get("Review_ID", f"UNKNOWN_{idx}")
                result["model_name"] = row.get("Model_Name", "Unknown")
                result["text_length"] = len(review_text)
                result["language_detected"] = "mixed"

                # Ensure tokens/lemmas are strings for storage
                result["tokens"] = " ".join(result["tokens"]) if isinstance(result.get("tokens"), list) else result.get("tokens", "")
                result["lemmas"] = " ".join(result["lemmas"]) if isinstance(result.get("lemmas"), list) else result.get("lemmas", "")
                result["preprocessed_text"] = result.get("cleaned_text", "")

                return (idx, result, None)
            except Exception as e:
                return (idx, None, e)

        preprocessed_data: List[Dict[str, Any]] = []

        # Decide whether to run in parallel
        from config import USE_PARALLEL_PROCESSING, MAX_WORKERS

        if USE_PARALLEL_PROCESSING and len(df) > 1:
            # Use ThreadPoolExecutor for I/O/CPU-mixed preprocessing; per-thread TextPreprocessor
            max_workers = min(MAX_WORKERS, len(df))
            futures: List[concurrent.futures.Future[Any]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                for idx, row in df.iterrows():
                    futures.append(ex.submit(_process_item, (idx, row)))

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preprocessing"):
                    try:
                        idx, result, err = future.result()
                        if err:
                            logger.warning(f"Error preprocessing row {idx}: {err}")
                            continue
                        assert result is not None
                        preprocessed_data.append(result)
                    except Exception as e:
                        logger.warning(f"Unexpected error in worker: {e}")
                        continue

        else:
            # Sequential fallback (safe, simpler for small datasets)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
                idx, result, err = _process_item((idx, row))
                if err:
                    logger.warning(f"Error preprocessing row {idx}: {err}")
                    continue
                assert result is not None
                preprocessed_data.append(result)

        preprocessed_df = pd.DataFrame(preprocessed_data)

        logger.info(f"Preprocessing completed: {len(preprocessed_df)} reviews processed")
        logger.info(f"Preprocessed DataFrame columns: {list(preprocessed_df.columns)}")
        logger.info("=" * 60)

        return preprocessed_df
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise
