"""
Configuration module for KnowMySocials NLP pipeline.
Manages database connections, paths, and pipeline parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = Path(__file__).parent

# Load environment variables from .env if present
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database Configuration
# Update these with your PostgreSQL credentials
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "knowmysocials_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# SQLAlchemy Database URL (PostgreSQL)
# The application now requires PostgreSQL. Configure connection via environment
# variables: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD. We build the
# SQLAlchemy URL here and expose it as `DATABASE_URL` for the rest of the app.
PASSWORD_PART = f":{DB_PASSWORD}" if DB_PASSWORD else ""
_database_url = f"postgresql+psycopg2://{DB_USER}{PASSWORD_PART}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Public constant used by SQLAlchemy helpers
DATABASE_URL = _database_url

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "pipeline.log"

# Optional test overrides: use SQLite for quick local runs and override CSV path.

# Allow overriding the CSV used for quick test runs (e.g., SAMPLE_5_REVIEWS)
_sample_csv_env = os.getenv("SAMPLE_CSV")
if _sample_csv_env:
    _csv_candidate = DATA_DIR / _sample_csv_env
else:
    # Data Collection Configuration
    default_sample = DATA_DIR / "sample_reviews.csv"
    fallback_sample = DATA_DIR / "sample_100_reviews.csv"
    _csv_candidate = default_sample if default_sample.exists() else fallback_sample

# Single assignment to uppercase constant to satisfy static analyzers
CSV_FILE_PATH = _csv_candidate
CHUNK_SIZE = 1000  # Process CSV in chunks for large files

# Preprocessing Configuration
# Custom stop words for automotive domain
AUTOMOTIVE_STOPWORDS = {
    "bike", "motorcycl", "road", "drive", "go", "come", "way", "day",
    "leke", "gai", "aati", "hora", "hai", "ko", "ne", "se", "ke", "aur",
    "ek", "ye", "bhai", "jo", "ho", "lo", "diya", "kar", "aa", "raha",
    "pe", "par", "mein", "me", "nahi", "na", "the", "tha", "hain", "hona"
}

# Hinglish normalization mappings
HINGLISH_REPLACEMENTS = {
    "kmpl": "kilometers_per_liter",
    "mileage": "fuel_efficiency",
    "activa": "activa",
    "cbr": "cbr",
    "fireblade": "fireblade",
    "shine": "shine",
    "dio": "dio",
    "bhai": "brother",
    "zindabad": "long_live",
    "chala": "ran",
    "saath": "together",
    "chhoti": "small",
    "cheezein": "things",
    "karti": "causing",
    "hain": "are",
    "flyover": "highway",
    "sadak": "road"
}

# spaCy model configuration
SPACY_MODEL = "en_core_web_sm"

# NLTK data storage path inside the project venv
NLTK_DATA_DIR = PROJECT_ROOT / ".venv" / "nltk_data"
NLTK_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Hybrid workflow toggle: when True, run fast rule-based preprocessing first
# and only send "complex" reviews to expensive AI stages (embeddings, topics,
# clustering). Toggle to False to always run full AI pipeline.
USE_HYBRID_WORKFLOW = True

# Parallel processing settings
# When True the pipeline will use concurrent.futures to parallelize
# preprocessing and embedding generation where appropriate.
USE_PARALLEL_PROCESSING = True

# Maximum number of workers to use for ThreadPool/ProcessPool executors.
# Default to number of CPUs or 8 if os.cpu_count() is None.
MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.cpu_count() or 8))

# Keywords used by the hybrid rule-based stage to quickly detect topics/concerns
# (mileage, suspension, service, vibration, comfort, heat, pickup, storage, braking)
HYBRID_KEYWORDS = [
    "mileage", "suspension", "service", "vibration", "comfort", "heat",
    "pickup", "storage", "brake", "braking", "engine", "pickup", "noise",
    "vibration", "fuel", "fuel_efficiency", "seat", "comfort"
]

# Number of distinct keywords above which a review is considered "complex" and
# therefore routed to the AI processing path. Tune as needed.
HYBRID_COMPLEXITY_THRESHOLD = 2

# Pipeline result table names
EMBEDDINGS_TABLE = "review_embeddings"
TOPIC_ASSIGNMENTS_TABLE = "review_topics"
TOPIC_SUMMARY_TABLE = "topic_summary"
CLUSTER_ASSIGNMENTS_TABLE = "review_clusters"

# Database table names
RAW_DATA_TABLE = "raw_reviews"
PREPROCESSED_DATA_TABLE = "preprocessed_reviews"
METADATA_TABLE = "review_metadata"

# New consolidated processed table for the 4-table refactor
PROCESSED_DATA_TABLE = "processed_reviews"

# Topic master and pipeline run tables
TOPIC_MASTER_TABLE = "topic_master"
PIPELINE_RUNS_TABLE = "pipeline_runs"

# ============================================================================
# SENTIMENT ANALYSIS & ASPECT EXTRACTION CONFIGURATION
# ============================================================================

# VADER model (built into nltk, no download needed)
USE_VADER_SENTIMENT = True

# Embedding-based sentiment similarity threshold
# Reviews above this similarity to negative/positive embeddings are classified
SENTIMENT_SIMILARITY_THRESHOLD = 0.3

# Aspect keywords for fast-path aspect extraction (rule-based)
ASPECT_KEYWORDS = {
    "mileage": ["mileage", "fuel", "kmpl", "efficiency", "consumption", "economy"],
    "engine": ["engine", "pickup", "acceleration", "torque", "power", "rev", "rpm"],
    "comfort": ["comfort", "seat", "ride", "smooth", "suspension", "handling"],
    "design": ["design", "look", "color", "aesthetic", "styling", "finish"],
    "build_quality": ["build", "quality", "durability", "solid", "finish", "paint"],
    "heat": ["heat", "temperature", "hot", "cooling", "overheat"],
    "noise": ["noise", "sound", "vibration", "rattle", "squeak"],
    "service": ["service", "maintenance", "spare", "parts", "repair", "warranty"],
    "braking": ["brake", "braking", "stopping", "abs"],
    "storage": ["storage", "compartment", "boot", "trunk", "space"],
}

# ============================================================================
# SUMMARIZATION CONFIGURATION
# ============================================================================

# Summarization model (BART or T5 via transformers)
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # or "t5-base" for T5

# Max length for generated summaries (tokens)
SUMMARY_MAX_LENGTH = 50
SUMMARY_MIN_LENGTH = 10

# Batch size for summarization (process multiple reviews at once for efficiency)
SUMMARIZATION_BATCH_SIZE = 8

# Zero-shot classification labels for aspect-level summarization
ZERO_SHOT_LABELS = [
    "mileage and fuel efficiency",
    "engine performance and pickup",
    "comfort and ride quality",
    "design and aesthetics",
    "build quality and durability",
    "heat and temperature management",
    "noise and vibration",
    "maintenance and service",
    "braking and safety",
    "storage and practicality"
]

# ============================================================================
# TREND PREDICTION CONFIGURATION
# ============================================================================

# Use Prophet for historical trend analysis (requires statsmodels + pystan/cmdstanpy)
USE_PROPHET_TRENDS = True

# Optional: use LSTM for future prediction (requires tensorflow/keras)
USE_LSTM_PREDICTION = False

# Number of future periods to forecast (in terms of review batches or time units)
FORECAST_PERIODS = 12

# Prophet seasonality settings
PROPHET_SEASONALITY_MODE = "additive"  # or "multiplicative"
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = False
PROPHET_DAILY_SEASONALITY = False

# Trend analysis window: analyze trends within the last N reviews per model
TREND_WINDOW_SIZE = 100

# ============================================================================
# CACHING & OPTIMIZATION
# ============================================================================

# Cache embeddings to avoid redundant computation
CACHE_EMBEDDINGS = True

# Cache summarization results
CACHE_SUMMARIES = True

# Max cache size (number of items)
MAX_CACHE_SIZE = 5000

# ============================================================================
# EXPORT & REPORTING
# ============================================================================

# Output CSV filename for final processed reviews
EXPORT_CSV_FILENAME = "reviews_final_structured.csv"

# Include all sentiment/aspect/summary details in export
EXPORT_DETAILED_SENTIMENT = True

# Include trend predictions in export
EXPORT_TREND_PREDICTIONS = True
