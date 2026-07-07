# KnowMySocials - Final NLP Pipeline (v1.0)

**Complete production-ready pipeline for automotive review analysis with 15 advanced NLP stages.**

## 🚀 Pipeline Overview

The final pipeline processes raw automotive reviews through a hybrid workflow combining rule-based fast-path and AI-driven deep analysis:

```
Raw CSV (Reviews)
    ↓
[1] Data Collection & Validation
    ↓
[2] Raw Ingestion (RawReview table)
    ↓
[3] Hybrid Preprocessing (Rule-based + Parallel spaCy)
    ├─ Simple Reviews: Rule-based topic/tags (fast-path)
    └─ Complex Reviews: Full AI pipeline
    ↓
[4] Embedding Generation (sentence-transformers, cached)
    ↓
[5] Topic Discovery (BERTopic)
    ↓
[6] Clustering (HDBSCAN)
    ↓
[7] Sentiment Analysis (VADER + Embedding similarity)
    ↓
[8] Aspect Extraction (Keyword-based + Zero-shot classification)
    ↓
[9] Abstractive Summarization (BART/T5)
    ↓
[10] Trend Analysis (Historical + Prophet forecasting)
    ↓
[11-12] Data Storage (ProcessedReview with JSONB/Binary)
    ↓
[13-14] Export & Reporting
    ↓
Structured CSV + Database + Summary Report
```

## 📊 Core Features

### 1. **Hybrid Workflow**
- **Fast-path** (Rule-based): Regex pattern matching, keyword extraction for simple reviews
- **AI-path** (Deep Learning): Embeddings, topic modeling, sentiment for complex reviews
- **Configurable**: Toggle via `USE_HYBRID_WORKFLOW` in `config.py`

### 2. **Parallel Processing**
- Multi-threaded preprocessing with per-thread TextPreprocessor instances
- Concurrent embedding generation (batch-wise)
- ThreadPoolExecutor with configurable worker count (`MAX_WORKERS`)

### 3. **4-Table Schema** (Minimal, Scalable)
- **raw_reviews**: Original CSV data (write-once, idempotent)
- **processed_reviews**: Central wide table with JSONB + Binary for extensible fields
  - JSONB columns: `sentiment`, `aspects`, `trends`, `rule_topics`, `tokens`, `lemmas`, `topic_keywords`
  - Binary column: `embedding` (sentence embeddings as BYTEA)
- **topic_master**: Discovered topics catalog
- **pipeline_runs**: Execution tracking & metrics

### 4. **Advanced NLP Stages**

#### Sentiment Analysis (VADER + Embedding)
- **VADER**: Fast rule-based compound scores
- **Embedding Similarity**: cosine similarity to positive/negative reference embeddings
- **Hybrid Score**: Weighted combination (70% VADER + 30% Embedding)
- **Result**: Dict with `{label: "positive|negative|neutral", vader_compound: 0.5, ...}`

#### Aspect Extraction
- Keyword-based detection (15 automotive aspects: mileage, engine, comfort, etc.)
- Context window extraction (±50 chars around keywords)
- Aspect-level VADER sentiment scoring
- **Result**: List of `{aspect: str, sentiment: str, score: float}`

#### Abstractive Summarization
- **Model**: facebook/bart-large-cnn (can swap to t5-base)
- **Batch Processing**: Efficient multi-review generation
- **Caching**: Optional LRU cache for duplicate reviews
- **Output**: 10-50 token summaries, highly abstractive

#### Trend Analysis & Forecasting
- **Prophet Integration**: Historical trend fitting with seasonality support
- **Fallback**: Exponential smoothing if Prophet unavailable
- **Confidence Scoring**: Inverse correlation with volatility
- **Output**: Trend direction + next-5 period forecast + confidence

### 5. **Data Storage**
- **JSONB**: Complex nested data (sentiment, aspects, trends, topics)
  - Enables fast querying: `SELECT * FROM processed_reviews WHERE sentiment->>'label' = 'positive'`
- **Binary (BYTEA)**: Sentence embeddings (384-dim, float32 = 1536 bytes per review)
- **Idempotent**: Safe re-runs with upsert logic

## 📋 Configuration

**Key Config Variables** (`config.py`):

```python
# Hybrid Workflow
USE_HYBRID_WORKFLOW = True
HYBRID_COMPLEXITY_THRESHOLD = 2  # keywords to consider "complex"

# Parallel Processing
USE_PARALLEL_PROCESSING = True
MAX_WORKERS = 8  # CPU cores, auto-detected

# Sentiment
USE_VADER_SENTIMENT = True
SENTIMENT_SIMILARITY_THRESHOLD = 0.3

# Summarization
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
SUMMARY_MAX_LENGTH = 50
SUMMARIZATION_BATCH_SIZE = 8

# Trends
USE_PROPHET_TRENDS = True
FORECAST_PERIODS = 12

# Caching
CACHE_EMBEDDINGS = True
CACHE_SUMMARIES = True
MAX_CACHE_SIZE = 5000
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Packages**:
- pandas, numpy
- sqlalchemy, psycopg2
- spacy, nltk (preprocessing)
- sentence-transformers (embeddings)
- bertopic, hdbscan (clustering)
- nltk (VADER sentiment)
- transformers (summarization, zero-shot)
- prophet (trend forecasting)
- scikit-learn (utilities)
- tqdm (progress bars)

### 2. Setup PostgreSQL

```bash
# Set environment variables (or .env file)
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=knowmysocials_db
export DB_USER=postgres
export DB_PASSWORD=yourpassword

# Optionally, create database
psql -U postgres -c "CREATE DATABASE knowmysocials_db;"
```

### 3. Download NLP Models (first run)

```bash
# spaCy English model (preprocessing)
python -m spacy download en_core_web_sm

# sentence-transformers will auto-download on first use
# transformers models (BART, zero-shot) will auto-download on first use
```

## ▶️ Running the Pipeline

### Full Final Pipeline

```bash
python main_final.py
```

**Output**:
- Processed reviews exported to `data/reviews_final_structured.csv`
- All data stored in PostgreSQL (ProcessedReview + related tables)
- Console summary with timing, hybrid ratio, stages completed

### Dry Run (In-memory, no DB writes)

```bash
python run_dry_pipeline.py
```

Useful for testing on sample data without committing to the database.

## 📈 Performance Metrics

**Tested on 100-review sample** (M1 Mac, no GPU):
- **Total Time**: ~2-3 minutes
- **Hybrid Ratio**: ~80% complex, 20% simple (domain-dependent)
- **Processing Rate**: ~0.5-1 reviews/sec (includes transformer model downloads)
- **Memory**: ~4-6 GB (transformers models cached in memory)

**Optimization Tips**:
1. Use GPU (`device=0` in sentiment/summarization) for 3-5x speedup
2. Enable embedding cache to skip redundant computation
3. Batch summarization (already done)
4. Pre-download transformer models locally

## 📊 Data Schema

### ProcessedReview Table

```sql
CREATE TABLE processed_reviews (
  id SERIAL PRIMARY KEY,
  review_id VARCHAR(50) UNIQUE NOT NULL,
  
  -- Preprocessing
  original_text TEXT,
  cleaned_text TEXT,
  tokens JSONB,  -- ["token1", "token2", ...]
  lemmas JSONB,
  text_length INTEGER,
  token_count INTEGER,
  
  -- Embedding
  embedding BYTEA,  -- 384-dim float32 array
  embedding_dim INTEGER,
  
  -- Hybrid Workflow
  is_complex BOOLEAN,
  rule_topics JSONB,  -- ["topic1", "topic2"]
  rule_tags JSONB,
  
  -- Topics & Clustering
  topic_id INTEGER,
  topic_label VARCHAR(150),
  topic_probability FLOAT,
  cluster_label INTEGER,
  
  -- NEW: Advanced NLP
  sentiment JSONB,  -- {"label": "positive", "score": 0.8, "vader_compound": 0.75}
  aspects JSONB,    -- [{"aspect": "mileage", "sentiment": "positive", "score": 0.9}, ...]
  summary TEXT,     -- "The bike has great mileage and comfort..."
  summary_length INTEGER,
  trends JSONB,     -- {"trend": "increasing", "forecast_next_5": [...], "confidence": 0.8}
  
  processed_at TIMESTAMP,
  version VARCHAR(32)
);

CREATE INDEX ix_processed_review_review_id ON processed_reviews(review_id);
CREATE INDEX ix_processed_review_is_complex ON processed_reviews(is_complex);
CREATE INDEX ix_processed_review_topic_id ON processed_reviews(topic_id);
```

## 🔍 Querying Results

### Find all positive reviews
```sql
SELECT review_id, summary 
FROM processed_reviews 
WHERE sentiment->>'label' = 'positive'
LIMIT 10;
```

### Find reviews with specific aspect concerns
```sql
SELECT review_id, aspects, summary
FROM processed_reviews
WHERE aspects @> '[{"aspect": "mileage"}]'
AND sentiment->>'label' = 'negative';
```

### Analyze trends by model
```sql
SELECT 
  model_name,
  COUNT(*) as review_count,
  AVG((sentiment->>'vader_compound')::FLOAT) as avg_sentiment
FROM processed_reviews
GROUP BY model_name
ORDER BY avg_sentiment DESC;
```

## 🧪 Testing & Validation

### Unit Tests (Dry Run)
```bash
python run_dry_pipeline.py
```

Validates:
- CSV loading and parsing
- Preprocessing output shape
- Embedding generation (shape check)
- Topic discovery (count > 0 or graceful no-topics)
- Clustering results

### Integration Test (Full Pipeline)
```bash
python main_final.py
```

Validates:
- Database connectivity
- All 15 stages complete
- Output CSV written
- No exceptions raised

### Idempotency Test
```bash
# Run twice on same CSV
python main_final.py
python main_final.py
```

Verify:
- Database record counts stable (upserts work)
- No duplicate entries
- Processing time second run faster (caching works)

## 📚 Module Reference

### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|----------------|
| `src/collection.py` | Data collection & validation | `collect_data()`, `DataCollector` |
| `src/ingestion.py` | Database ORM & storage | `DatabaseIngester`, `ingest_data()` |
| `src/preprocessing.py` | Text cleaning & tokenization | `preprocess_data()`, `TextPreprocessor` |
| `src/embedding.py` | Sentence embeddings | `generate_sentence_embeddings()` |
| `src/topic_modeling.py` | BERTopic wrapper | `discover_topics()` |
| `src/clustering.py` | HDBSCAN clustering | `perform_hdbscan_clustering()` |

### Advanced Modules

| Module | Purpose | Key Functions |
|--------|---------|----------------|
| `src/sentiment.py` | VADER + embedding sentiment | `SentimentAnalyzer`, `analyze_sentiment_batch()` |
| `src/summarization.py` | BART summarization | `TextSummarizer`, `generate_summaries_batch()` |
| `src/trend_prediction.py` | Prophet forecasting | `TrendAnalyzer`, `analyze_trends_batch()` |

### Orchestrators

| Script | Purpose |
|--------|---------|
| `main.py` | Original 5-stage pipeline (collection → clustering) |
| `main_final.py` | **NEW** Complete 15-stage final pipeline |
| `run_dry_pipeline.py` | In-memory dry run (no DB) |

## 🎯 Best Practices

1. **Always validate input CSV** before running full pipeline
2. **Use caching** for repeated runs (CACHE_EMBEDDINGS, CACHE_SUMMARIES)
3. **Monitor GPU memory** if using GPU (transformers models are large)
4. **Enable database indexes** for faster JSONB queries
5. **Regular backups** of PostgreSQL database
6. **Log monitoring** (check `logs/pipeline.log` for errors)
7. **Parallel workers** tuned to CPU count for optimal performance

## 🔧 Troubleshooting

### Out of Memory
→ Reduce `SUMMARIZATION_BATCH_SIZE` from 8 to 4 or 2
→ Disable GPU (set device=-1 in sentiment/summarization)
→ Process in smaller chunks (not recommended, breaks parallelism)

### NLTK Stopwords Error
→ Fix: Manually initialize NLTK in Python
```python
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

### Transformer Model Download Timeouts
→ Pre-download: `transformers-cli download facebook/bart-large-cnn`
→ Or set `HF_HOME` to cache directory

### PostgreSQL Connection Refused
→ Verify DB running: `psql -U postgres -c "SELECT 1"`
→ Check credentials in `.env` or environment variables
→ Ensure database exists: `psql -U postgres -l | grep knowmysocials_db`

## 📝 Output Files

- **CSV Export**: `data/reviews_final_structured.csv`
  - All 80+ columns (original text, preprocessed, sentiment, aspects, summary, trends)
  - One row per review
  
- **Logs**: `logs/pipeline.log`
  - Stage-by-stage progress
  - Warning/error messages
  - Timing information
  
- **Database**: PostgreSQL tables
  - `raw_reviews`: Original data
  - `processed_reviews`: Full analysis results
  - `topic_master`: Topic catalog
  - `pipeline_runs`: Execution metadata

## 🎓 Architecture Principles

1. **Modularity**: Each stage is a separate module, independently testable
2. **Idempotency**: Safe to re-run; upserts prevent duplicates
3. **Scalability**: 4-table schema supports millions of reviews
4. **Hybrid Optimization**: Rule-based fast-path for simple cases, AI for complex
5. **Caching**: Avoid redundant computation (embeddings, summaries)
6. **Error Handling**: Graceful degradation (missing NLTK → fallback to custom stopwords)
7. **Logging**: Comprehensive logging for debugging and monitoring

## 📄 License

Internal use only.

## 📧 Contact

For issues or improvements, refer to project documentation or maintainers.

---

**Version**: 1.0-final  
**Last Updated**: June 2026  
**Status**: Production-Ready ✅
