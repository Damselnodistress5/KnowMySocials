# KnowMySocials - QUICK REFERENCE GUIDE

## 🎯 One-Page Overview

### What is KnowMySocials?
A production-ready NLP pipeline that transforms raw automotive reviews into structured, AI-analyzed data with sentiment, topics, trends, and predictions.

### How to Run?
```bash
# Full pipeline (collection → sentiment → trends → export)
python main_final.py

# Dry run (test without database)
python run_dry_pipeline.py
```

### Output
- PostgreSQL tables with 4-table schema (raw_reviews, processed_reviews, topic_master, pipeline_runs)
- CSV export: `data/reviews_final_structured.csv` with all analysis results
- Console summary: execution time, hybrid ratio, stages completed

---

## 📊 Pipeline Stages

| # | Stage | Input | Output | Time |
|---|-------|-------|--------|------|
| 1-3 | Collection → Preprocessing | CSV | Cleaned tokens | 0.1s |
| 4-6 | Embeddings → Topics → Clusters | Tokens | Topic ID, embeddings | 1.0s |
| **7** | **Sentiment** (NEW) | Text | Label + score | 0.1s |
| **8** | **Aspects** (NEW) | Text | [aspect, sentiment] | 0.05s |
| **9** | **Summarization** (NEW) | Text | Summary | 2.0s |
| **10** | **Trends** (NEW) | Sentiment history | Trend + forecast | 0.05s |
| 11-15 | Storage → Export → Report | All above | DB + CSV + Report | 0.2s |

---

## 🔧 Key Files

```
KnowMySocials/
├── main_final.py                 # ← Main entry point (15-stage orchestrator)
├── config.py                     # Configuration (models, thresholds, flags)
├── run_dry_pipeline.py           # Test script (no DB writes)
├── src/
│   ├── sentiment.py              # NEW: VADER + embedding sentiment
│   ├── summarization.py          # NEW: BART abstractive summaries
│   ├── trend_prediction.py       # NEW: Prophet forecasting
│   ├── preprocessing.py          # Text cleaning, tokenization
│   ├── embedding.py              # sentence-transformers
│   ├── topic_modeling.py         # BERTopic wrapper
│   ├── clustering.py             # HDBSCAN wrapper
│   ├── ingestion.py              # PostgreSQL ORM & storage
│   ├── collection.py             # CSV loading & validation
│   └── utils.py                  # Logging, DB helpers
├── data/
│   ├── sample_reviews.csv        # Test data
│   └── reviews_final_structured.csv  # Output (created by pipeline)
├── logs/
│   └── pipeline.log              # Detailed execution logs
├── README_FINAL.md               # Complete documentation
└── IMPLEMENTATION_SUMMARY.md     # This guide
```

---

## ⚙️ Configuration (config.py)

### Toggle Features
```python
USE_HYBRID_WORKFLOW = True           # Rule-based fast-path for simple reviews
USE_PARALLEL_PROCESSING = True       # Multi-threaded processing
USE_VADER_SENTIMENT = True           # VADER sentiment analysis
USE_PROPHET_TRENDS = True            # Prophet trend forecasting
CACHE_EMBEDDINGS = True              # Skip duplicate computations
CACHE_SUMMARIES = True               # Cache summary results
```

### Tuning
```python
HYBRID_COMPLEXITY_THRESHOLD = 2      # Keywords → "complex" review
MAX_WORKERS = 8                      # Thread pool size (auto-detect CPU)
SENTIMENT_SIMILARITY_THRESHOLD = 0.3 # Embedding similarity cutoff
SUMMARIZATION_BATCH_SIZE = 8         # Process 8 reviews at once
FORECAST_PERIODS = 12                # Predict next 12 periods
```

### Models
```python
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Or "t5-base"
SPACY_MODEL = "en_core_web_sm"                     # Tokenization
PROPHET_SEASONALITY_MODE = "additive"              # Or "multiplicative"
```

---

## 📦 Database Schema (ProcessedReview)

**Main Table** (`processed_reviews`): Central storage for all analysis

```sql
-- NEW JSONB columns for advanced NLP:
sentiment JSONB         -- {"label": "positive", "vader_compound": 0.75, ...}
aspects JSONB           -- [{"aspect": "mileage", "sentiment": "positive", "score": 0.9}]
summary TEXT            -- "The bike offers excellent fuel efficiency..."
trends JSONB            -- {"trend": "increasing", "forecast_next_5": [...], "confidence": 0.8}

-- Existing columns still present:
tokens JSONB, lemmas JSONB, is_complex BOOLEAN, embedding BYTEA, etc.
```

**Query Examples**:
```sql
-- Find positive reviews
SELECT review_id, summary FROM processed_reviews 
WHERE sentiment->>'label' = 'positive';

-- Find reviews mentioning specific aspects
SELECT review_id, aspects FROM processed_reviews 
WHERE aspects @> '[{"aspect": "mileage"}]'::jsonb;

-- Analyze sentiment by model
SELECT model_name, AVG((sentiment->>'vader_compound')::FLOAT) as avg_sentiment
FROM processed_reviews GROUP BY model_name;
```

---

## 🚀 Common Commands

### Full Pipeline Run
```bash
python main_final.py
```
Output: Processes all reviews through 15 stages, stores in DB, exports CSV

### Dry Run (Testing)
```bash
python run_dry_pipeline.py
```
Output: In-memory processing, shows pipeline flow without DB writes

### Check Logs
```bash
tail -f logs/pipeline.log
```
Output: Real-time pipeline progress and any errors

### Query Database
```bash
psql -U postgres -d knowmysocials_db -c "SELECT COUNT(*) FROM processed_reviews;"
```
Output: Total processed reviews in database

### Export Specific Reviews
```bash
psql -U postgres -d knowmysocials_db -c "
  COPY (SELECT review_id, summary FROM processed_reviews 
        WHERE sentiment->>'label' = 'negative')
  TO '/tmp/negative_reviews.csv' WITH CSV HEADER;
"
```

---

## 🧪 Testing

### Validate Setup
```bash
python -c "
import config
from src.utils import get_db_engine
print('✓ Config loaded')
engine = get_db_engine()
print('✓ Database connected')
"
```

### Run on Sample Data
```bash
# Single-review test (in-memory)
python run_dry_pipeline.py

# Multi-review test (100 reviews)
python main_final.py
```

### Check Output
```bash
# View final CSV
head -20 data/reviews_final_structured.csv

# Count records
wc -l data/reviews_final_structured.csv

# Query database
psql -U postgres -d knowmysocials_db -c "SELECT * FROM processed_reviews LIMIT 5;"
```

---

## 🔍 Understanding Output

### Console Report (After Pipeline Completes)
```
📊 EXECUTION REPORT

  Total Execution Time:    145.32s
  Total Reviews Processed: 100
  Processing Rate:         0.68 reviews/sec

  Hybrid Workflow:
    • Simple (Rule-based):   20
    • Complex (AI Pipeline): 80
    • Complex Ratio:         80.0%

  Stages Completed: 15/15
    1. collection
    2. ingestion
    3. preprocessing
    ...
    15. export

  📁 Export Location: data/reviews_final_structured.csv
```

### CSV Columns (Final Export)
- review_id, model_name, original_text, preprocessed_text
- is_complex, token_count, hinglish_score
- **sentiment** (JSON string), **aspects** (JSON string)
- **summary**, **trends** (JSON string)
- topic_label, cluster_label
- [60+ columns total]

### Database Records
**raw_reviews** (5 cols): ID, text, model, date, source  
**processed_reviews** (30+ cols): Full analysis (all above + embeddings, sentiment, etc.)  
**topic_master** (4 cols): Topic ID, name, keywords, first_seen  
**pipeline_runs** (6 cols): Run ID, status, timing, review counts

---

## 🐛 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "ModuleNotFoundError: pandas" | Missing dependencies | `pip install -r requirements.txt` |
| "Connection refused (PostgreSQL)" | DB not running | `sudo service postgresql start` or `brew services start postgresql` |
| "Type of 'to_dict' is partially unknown" | pandas stubs | Ignore (runtime works fine, static check noise) |
| "NLTK stopwords error" | Missing NLTK data | `python -m nltk.downloader stopwords` |
| "Out of memory" | Transformers too large | Reduce `SUMMARIZATION_BATCH_SIZE = 4` or use GPU |
| "Timeout downloading models" | Network issue | Pre-download: `transformers-cli download facebook/bart-large-cnn` |

---

## 🎓 Architecture Principles

1. **Modular**: Each NLP stage is independent, testable
2. **Idempotent**: Safe to re-run; upserts prevent duplicates
3. **Hybrid**: Rule-based fast-path + AI for complex cases
4. **Scalable**: 4-table schema handles millions of reviews
5. **Efficient**: Caching, batch processing, parallel threading
6. **Observable**: Comprehensive logging, progress bars, reports

---

## 📈 Performance Tips

✅ **Enable GPU**: Set device=0 in sentiment/summarization (3-5x faster)  
✅ **Cache Results**: CACHE_EMBEDDINGS=True, CACHE_SUMMARIES=True  
✅ **Batch Processing**: Increase batch size if memory allows  
✅ **Parallel Workers**: Set MAX_WORKERS to CPU count  
✅ **Index Database**: Create indexes on frequently queried columns  
✅ **Monitor Logs**: Check for bottlenecks in pipeline.log  

---

## 📞 Support

- **Documentation**: See README_FINAL.md for detailed guides
- **Code Comments**: Each module has inline documentation
- **Examples**: See query examples in Database Schema section above
- **Logs**: Check logs/pipeline.log for detailed execution trace

---

## 📋 Checklist Before Production

- [ ] PostgreSQL database created and accessible
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] NLP models downloaded (spaCy, transformers)
- [ ] Test run successful (`python run_dry_pipeline.py`)
- [ ] Full pipeline tested (`python main_final.py`)
- [ ] Database queries working (sample SELECT statements)
- [ ] CSV export validated (columns, data quality)
- [ ] Logs reviewed for errors/warnings
- [ ] Performance tuned (batch size, workers, etc.)
- [ ] Backup strategy in place (PostgreSQL backups)

---

## 🚀 Quick Start (< 5 mins)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Setup DB
export DB_HOST=localhost DB_PORT=5432 DB_NAME=knowmysocials_db DB_USER=postgres
psql -U postgres -c "CREATE DATABASE knowmysocials_db;"

# 3. Download models
python -m spacy download en_core_web_sm

# 4. Test
python run_dry_pipeline.py

# 5. Run
python main_final.py

# 6. Query
psql -U postgres -d knowmysocials_db -c "SELECT COUNT(*) FROM processed_reviews;"
```

---

**Version**: 1.0  
**Last Updated**: June 2026  
**Status**: ✅ Production-Ready
