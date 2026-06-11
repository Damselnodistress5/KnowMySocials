# KnowMySocials NLP Pipeline

A production-ready Python NLP pipeline for processing motorcycle/automotive reviews from social media platforms. Designed to extract, store, and preprocess Hinglish (Hindi-English mixed) text data.

## 📋 Project Overview

This project implements a complete data processing pipeline with three main phases:

1. **Data Collection** - Load reviews from CSV files efficiently
2. **Data Ingestion** - Store raw data in PostgreSQL with proper schema
3. **Preprocessing** - Clean, tokenize, lemmatize, and normalize text

## 🗂️ Project Structure

```
knowmysocials/
├── data/
│   └── honda_reviews.csv          # Input data (5500+ rows)
├── src/
│   ├── __init__.py               # Package initialization
│   ├── collection.py             # Data collection module
│   ├── ingestion.py              # Database ingestion module
│   ├── preprocessing.py          # Text preprocessing module
│   └── utils.py                  # Utility functions (logging, DB)
├── logs/
│   └── pipeline.log              # Execution logs
├── main.py                        # Main pipeline orchestrator
├── config.py                      # Configuration & constants
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## ⚙️ Technical Stack

- **Python 3.10+**
- **Data Processing:** pandas, numpy
- **Database:** PostgreSQL, SQLAlchemy, psycopg2
- **NLP:** spaCy, NLTK
- **Progress Tracking:** tqdm
- **Configuration:** python-dotenv

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10 or higher** installed
2. **PostgreSQL** database running locally or remotely
3. **Git** (optional, for version control)

### Installation

#### 1. Clone/Setup Project
```bash
cd /Users/sepia/KnowMySocials
```

#### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# NLTK data will auto-download on first run
```

#### 4. Setup Database Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your PostgreSQL credentials
nano .env
```

Example `.env` content:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowmysocials_db
DB_USER=postgres
DB_PASSWORD=your_password
```

#### 5. Create PostgreSQL Database
```sql
-- Connect to PostgreSQL
psql -U postgres

-- Create database
CREATE DATABASE knowmysocials_db;

-- Verify creation
\l
```

### 6. Run the Pipeline
```bash
python main.py
```

## 📊 Module Descriptions

### `collection.py` - Data Collection
**Purpose:** Load CSV data efficiently with validation

**Key Classes:**
- `DataCollector` - Handles CSV loading and validation

**Key Methods:**
- `load_full_data()` - Load entire CSV into memory
- `load_chunked_data()` - Stream processing for large files
- `validate_data_schema()` - Verify required columns
- `check_data_quality()` - Report null values, duplicates, memory usage

**Usage:**
```python
from src.collection import collect_data

df = collect_data()  # Loads from config.CSV_FILE_PATH
```

### `ingestion.py` - Database Ingestion
**Purpose:** Store data in PostgreSQL with proper ORM models

**Key Classes:**
- `RawReview` - SQLAlchemy model for raw reviews
- `PreprocessedReview` - Model for preprocessed text
- `ReviewMetadata` - Model for text metadata
- `DatabaseIngester` - Manages DB operations

**Key Methods:**
- `create_tables()` - Create PostgreSQL tables
- `ingest_raw_data()` - Store raw reviews
- `ingest_preprocessed_data()` - Store preprocessed text
- `ingest_metadata()` - Store text metrics

**Usage:**
```python
from src.ingestion import ingest_data, DatabaseIngester

ingest_data(raw_df)

ingester = DatabaseIngester()
count = ingester.get_record_count('raw_reviews')
```

### `preprocessing.py` - Text Preprocessing
**Purpose:** Comprehensive NLP preprocessing pipeline

**Key Classes:**
- `TextPreprocessor` - Main preprocessing engine

**Key Methods:**
- `clean_text()` - Remove URLs, HTML, special chars, normalize
- `tokenize_text()` - Split into word tokens
- `remove_stopwords()` - Filter common words
- `lemmatize_tokens()` - Reduce to word roots
- `preprocess_single_text()` - Full pipeline for one text

**Preprocessing Steps:**
1. URL removal
2. HTML tag removal
3. Special character removal
4. Hinglish normalization (e.g., "kmpl" → "kilometers_per_liter")
5. Lowercasing
6. Space normalization
7. Tokenization
8. Stopword removal
9. Lemmatization

**Usage:**
```python
from src.preprocessing import preprocess_data

preprocessed_df = preprocess_data(raw_df)
```

### `utils.py` - Utilities
**Purpose:** Shared utilities for logging and database connections

**Key Functions:**
- `setup_logger()` - Configure logging to file and console
- `get_db_engine()` - Create SQLAlchemy engine
- `get_db_connection()` - Context manager for DB sessions
- `validate_csv_exists()` - File validation

### `config.py` - Configuration
**Purpose:** Centralized configuration management

**Key Settings:**
- Database connection string
- File paths
- Logging configuration
- Preprocessing parameters
- Domain-specific stopwords
- Hinglish mappings

## 📝 CSV Data Format

Expected columns in input CSV:
- `Review_ID` - Unique review identifier
- `Model_Name` - Motorcycle/vehicle model (e.g., "Activa 125", "Shine 125")
- `Review_Date` - Date of review
- `Location` - Geographic location
- `User_Type` - User category (e.g., "Family User", "Track Rider")
- `Source` - Platform source (e.g., "Team-BHP", "YouTube Comments")
- `Ownership_Months` - Duration of ownership
- `Review_Text` - Review content (Hinglish or English)

## 🗄️ Database Schema

### raw_reviews
```sql
id (PK)              - Auto-increment primary key
review_id (UNIQUE)   - External review identifier
model_name           - Vehicle model
review_date          - Review timestamp
location             - Geographic location
user_type            - User category
source               - Data source
ownership_months     - Ownership duration
review_text          - Raw review text
created_at           - Ingestion timestamp
updated_at           - Last update timestamp
```

### preprocessed_reviews
```sql
id (PK)              - Auto-increment primary key
review_id (UNIQUE)   - Links to raw_reviews
original_text        - Original review text
preprocessed_text    - Cleaned text
tokens               - Space-separated tokens
lemmas               - Space-separated lemmas
model_name           - Vehicle model
created_at           - Processing timestamp
updated_at           - Last update timestamp
```

### review_metadata
```sql
id (PK)              - Auto-increment primary key
review_id (UNIQUE)   - Links to raw_reviews
text_length          - Character count
token_count          - Total tokens
unique_tokens        - Unique tokens
language_detected    - Language classification (mixed for Hinglish)
hinglish_score       - Percentage of Hinglish content (0-100)
processed_at         - Processing timestamp
```

## 📈 Processing Features

### Hinglish Normalization
Automatically converts Hinglish terms to English:
- "kmpl" → "kilometers_per_liter"
- "mileage" → "fuel_efficiency"
- "leke" → "take"
- And 20+ more domain-specific mappings

### Automotive Domain Stopwords
Custom stopwords specific to motorcycle reviews:
- Common verbs: "go", "come", "drive"
- Hinglish particles: "hai", "ko", "ne", "se", "ke"
- Domain terms: "bike", "motorcycl", "road"

### Text Metrics
Calculates for each review:
- **Token Count:** Total number of words
- **Unique Tokens:** Distinct word count
- **Hinglish Score:** Percentage of Hinglish content
- **Text Length:** Character count

## 🔍 Output & Verification

After running the pipeline, check results:

```bash
# View recent logs
tail -50 logs/pipeline.log

# Query database (PostgreSQL)
psql -U postgres -d knowmysocials_db

# Check record counts
SELECT COUNT(*) FROM raw_reviews;
SELECT COUNT(*) FROM preprocessed_reviews;
SELECT COUNT(*) FROM review_metadata;

# View sample preprocessed review
SELECT review_id, original_text, preprocessed_text, hinglish_score 
FROM preprocessed_reviews 
LIMIT 1;
```

## 📌 Answers to Key Questions

### ✅ Can we develop this entirely in Visual Studio Code?

**Yes, absolutely!**
- VS Code has excellent Python support via the Python extension
- Install "Python" extension by Microsoft
- Install "Pylance" for advanced type checking and autocomplete
- PostgreSQL can run locally or via Docker Desktop
- Use VS Code's integrated terminal for running commands
- Built-in debugger for stepping through code
- Extensions like "Thunder Client" for testing APIs (if added later)

**Setup in VS Code:**
1. Open workspace: `File → Open Folder → Select /Users/sepia/KnowMySocials`
2. Install extensions: Python, Pylance, PostgreSQL (optional)
3. Select Python interpreter: `Cmd+Shift+P → Python: Select Interpreter → Choose venv`
4. Run main script: `Cmd+Shift+D → Run and Debug → Python`

### ✅ Recommended Project Folder Structure

The provided structure is production-ready and follows best practices:

```
src/           - All application modules (separation of concerns)
data/          - Input data files (CSVs, datasets)
logs/          - Application logs (auto-created)
config.py      - Centralized configuration (no hardcoding)
main.py        - Single entry point for the pipeline
requirements.txt - All dependencies pinned to versions
.env.example   - Template for environment variables
```

**Benefits:**
- **Modularity:** Each component (collection, ingestion, preprocessing) is independent
- **Maintainability:** Easy to locate and modify specific functionality
- **Scalability:** Can add new modules (e.g., analysis.py, feature_extraction.py)
- **Testing:** Each module can be unit tested independently
- **CI/CD:** Clear structure for deployment automation

### ✅ Required Installations (requirements.txt)

```
pandas==2.2.0           - Data manipulation & analysis
numpy==1.26.4           - Numerical computations
sqlalchemy==2.0.23      - ORM for database operations
psycopg2-binary==2.9.9  - PostgreSQL adapter
spacy==3.7.2            - NLP library for lemmatization
nltk==3.8.1             - NLP toolkit (tokenization, stopwords)
python-dotenv==1.0.0    - Environment variable management
tqdm==4.66.1            - Progress bars for loops
```

**Post-Installation Commands:**
```bash
# Download spaCy English model (required for lemmatization)
python -m spacy download en_core_web_sm

# NLTK data auto-downloads on first use
# But you can pre-download with:
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Total Installation Size:** ~500 MB (mostly spaCy model)

## 🛠️ Troubleshooting

### PostgreSQL Connection Error
```
Error: could not connect to server: Connection refused
```
**Solution:**
```bash
# On macOS
brew services start postgresql
# or
pg_ctl -D /usr/local/var/postgres start

# On Linux
sudo service postgresql start
```

### spaCy Model Not Found
```
Error: Can't find model "en_core_web_sm"
```
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Import Errors
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Database Permission Error
```
FATAL: role "postgres" does not exist
```
**Solution:**
1. Check PostgreSQL status
2. Update credentials in `.env` file
3. Verify database user has create table privileges

## 📚 Sample Code Usage

### Load and Preprocess a Single Review
```python
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()

review = "Shine 125 leke life set ho gayi bhai. Mileage 55+ aa raha hai."
result = preprocessor.preprocess_single_text(review)

print(f"Original: {result['original_text']}")
print(f"Cleaned: {result['cleaned_text']}")
print(f"Tokens: {result['tokens']}")
print(f"Hinglish Score: {result['hinglish_score']}%")
```

### Query Preprocessed Data
```python
from src.utils import get_db_connection
from src.ingestion import PreprocessedReview

with get_db_connection() as session:
    reviews = session.query(PreprocessedReview).filter(
        PreprocessedReview.hinglish_score > 50
    ).limit(10).all()
    
    for review in reviews:
        print(f"{review.review_id}: {review.preprocessed_text[:100]}")
```

## 🔐 Security Considerations

- **Never commit `.env` file** - Add to `.gitignore`
- **Use strong PostgreSQL passwords** in production
- **Sanitize user inputs** before database operations (SQLAlchemy handles this)
- **Use environment variables** for sensitive data
- **Keep logs in .gitignore** - Logs may contain sensitive info

## 📄 License

This project is part of the KnowMySocials initiative.

## 🤝 Contributing

For bug reports or feature requests, please contact the development team.

---

**Last Updated:** June 2, 2026  
**Python Version:** 3.10+  
**Status:** Production Ready ✓
