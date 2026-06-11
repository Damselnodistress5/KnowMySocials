# KnowMySocials NLP Pipeline - Quick Setup Guide

**Complete setup time: 15-20 minutes**

## 🎯 In 30 Seconds

```bash
cd /Users/sepia/KnowMySocials
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
# Edit .env with your PostgreSQL credentials if needed
brew services start postgresql  # If not already running
psql -U postgres -c "CREATE DATABASE knowmysocials_db;"
python main.py
```

## ✅ Prerequisites Checklist

- [ ] Python 3.10+ installed (`python3 --version`)
- [ ] PostgreSQL installed (`psql --version`)
- [ ] PostgreSQL running (`brew services list`)
- [ ] VS Code installed (optional)

## 📦 Installation Steps

### 1. Create Virtual Environment
```bash
cd /Users/sepia/KnowMySocials
python3 -m venv venv
source venv/bin/activate  # (venv) should appear in prompt
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure Database
```bash
cp .env.example .env
# Only edit .env if your PostgreSQL setup differs from defaults
```

### 4. Create PostgreSQL Database
```bash
psql -U postgres -c "CREATE DATABASE knowmysocials_db;"
```

### 5. Run Pipeline
```bash
python main.py
```

## 📊 Expected Output

```
======================================================================
  KNOWMYSOCIALS NLP PIPELINE - STARTING
======================================================================

✓ Collected 5500 reviews
✓ Ingested raw reviews into PostgreSQL
✓ Preprocessed 5500 reviews
✓ Ingested preprocessed data and metadata

======================================================================
  PIPELINE EXECUTION SUMMARY
======================================================================

Raw Reviews Table: 5500 records
Preprocessed Reviews Table: 5500 records
Metadata Table: 5500 records

======================================================================
  PIPELINE COMPLETED SUCCESSFULLY ✓
======================================================================
```

## 📁 Project Files Overview

| File | Purpose |
|------|---------|
| `main.py` | **START HERE** - Main pipeline orchestrator |
| `config.py` | Configuration (DB credentials, paths, NLP settings) |
| `requirements.txt` | All Python dependencies |
| `.env.example` | Template for environment variables |
| `README.md` | Complete documentation |
| `GETTING_STARTED.py` | Detailed setup guide (this file) |
| `src/collection.py` | Load CSV data (5500+ rows) |
| `src/ingestion.py` | PostgreSQL storage & ORM models |
| `src/preprocessing.py` | Text cleaning, tokenization, lemmatization |
| `src/utils.py` | Logging & database utilities |
| `data/honda_reviews.csv` | Input data (1.8 MB) |
| `logs/pipeline.log` | Execution logs (auto-created) |

## 🗂️ Project Architecture

```
Data Collection         Data Ingestion          Preprocessing
    (CSV)         →    (PostgreSQL)      →    (NLP Pipeline)
 
Load 5500 reviews → Store in raw_reviews → Clean & normalize text
Validate schema   → Create tables          → Tokenize & lemmatize
Check quality     → Store metadata         → Calculate metrics
                  ↓
              Preprocessed Data
              (PostgreSQL Tables:
               - raw_reviews
               - preprocessed_reviews
               - review_metadata)
```

## 🔍 Verify Success

```bash
# View logs
tail -50 logs/pipeline.log

# Query database
psql -U postgres -d knowmysocials_db -c "SELECT COUNT(*) FROM raw_reviews;"
psql -U postgres -d knowmysocials_db -c "SELECT COUNT(*) FROM preprocessed_reviews;"
psql -U postgres -d knowmysocials_db -c "SELECT COUNT(*) FROM review_metadata;"

# View sample
psql -U postgres -d knowmysocials_db -c "
  SELECT review_id, original_text, hinglish_score 
  FROM preprocessed_reviews 
  LIMIT 1;"
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `command not found: python3` | Install Python: https://python.org/downloads/ |
| `Connection refused` (PostgreSQL) | Start PostgreSQL: `brew services start postgresql` |
| `ModuleNotFoundError` | Ensure venv activated: `source venv/bin/activate` |
| `Database does not exist` | Create: `psql -U postgres -c "CREATE DATABASE knowmysocials_db;"` |
| `spaCy model not found` | Download: `python -m spacy download en_core_web_sm` |

## 🆚 VS Code Setup (Optional)

1. **Open project in VS Code**
   ```bash
   code /Users/sepia/KnowMySocials
   ```

2. **Install extensions**
   - Python (by Microsoft)
   - Pylance (by Microsoft)

3. **Select Python interpreter**
   - `Cmd+Shift+P` → "Python: Select Interpreter" → Choose `./venv/bin/python`

4. **Run pipeline in VS Code**
   - `Ctrl+F5` (Run Without Debugging)
   - Or `F5` (Start Debugging)

## 💾 Database Configuration

**Default settings** (in `config.py`):
```python
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "knowmysocials_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
```

**To customize**, edit `.env`:
```bash
cp .env.example .env
nano .env
```

Then set your values:
```
DB_HOST=your_host
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

## 📊 What Gets Processed

The pipeline processes **5,500 motorcycle reviews** containing:

- **Source platforms:** Team-BHP, YouTube, Instagram, Facebook
- **Languages:** Hinglish (Hindi-English mixed)
- **Motorcycle models:** Shine 125, Activa 125, Dio, etc.
- **User types:** Family User, Track Rider, Beginner Rider
- **Review content:** 
  - Mileage/efficiency (e.g., "55+ kmpl")
  - Performance (e.g., "smooth engine", "no vibrations")
  - Reliability (e.g., "routine service only")
  - Comfort (e.g., "hard seat", "dim headlight")

## 🎓 Learning the Codebase

### Understand Data Flow
1. Read `main.py` - See overall pipeline orchestration
2. Read `src/collection.py` - See CSV loading
3. Read `src/ingestion.py` - See database operations
4. Read `src/preprocessing.py` - See NLP processing

### Key Concepts

**Data Collection** (`collection.py`):
- Loads CSV efficiently using pandas
- Validates schema and data quality
- Supports chunked reading for large files

**Data Ingestion** (`ingestion.py`):
- SQLAlchemy ORM models for type safety
- Three tables: raw reviews, preprocessed reviews, metadata
- Batch insertion with error handling

**Preprocessing** (`preprocessing.py`):
- Comprehensive text cleaning pipeline
- Hinglish-to-English normalization
- NLTK tokenization & stopword removal
- spaCy lemmatization
- Automotive domain-specific processing

## 🚀 Next Steps

After successful execution:

1. **Explore the data**
   ```bash
   psql -U postgres -d knowmysocials_db
   SELECT * FROM preprocessed_reviews LIMIT 5;
   ```

2. **Analyze preprocessing results**
   ```python
   # See GETTING_STARTED.py for code examples
   ```

3. **Extend the pipeline**
   - Add sentiment analysis
   - Add feature extraction
   - Add visualizations
   - See README.md for architecture patterns

4. **Deploy to production**
   - Add Docker containerization
   - Add CI/CD pipeline
   - Set up scheduled batch jobs
   - Expose API endpoints

## 📚 File Sizes & Performance

```
Data file:           1.8 MB (5500 reviews)
Venv size:          ~500 MB (including spaCy model)
Expected runtime:   5-15 minutes
Memory usage:       ~200-400 MB during execution
Database size:      ~50 MB (after full pipeline)
```

## ✨ Features

✅ **Production-ready code** - Clean, modular, well-documented  
✅ **Error handling** - Graceful failure recovery  
✅ **Logging** - Full execution logs to file and console  
✅ **Progress tracking** - Visual progress bars (tqdm)  
✅ **Scalable** - Supports large files via chunked processing  
✅ **Extensible** - Easy to add new preprocessing steps  
✅ **Modular** - Each component can be used independently  
✅ **Type hints** - Full Python type annotations  
✅ **Database** - Proper PostgreSQL schema with ORM  
✅ **NLP** - Advanced text processing with spaCy & NLTK  

## 📞 Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Check logs: `tail -50 logs/pipeline.log`
3. Review code comments in each module
4. Check GETTING_STARTED.py for detailed explanations

---

**Ready to start? Run this command:**
```bash
cd /Users/sepia/KnowMySocials && source venv/bin/activate && python main.py
```

Good luck! 🎉
