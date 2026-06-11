"""
GETTING STARTED GUIDE - KnowMySocials NLP Pipeline
====================================================

This guide walks you through setting up and running the NLP pipeline
on macOS using Visual Studio Code.
"""

# ==============================================================================
# STEP 1: INSTALL PREREQUISITES
# ==============================================================================

"""
Before starting, ensure you have:

1. Python 3.10+ installed
   Check: python3 --version
   Install from: https://www.python.org/downloads/

2. PostgreSQL installed and running
   On macOS:
      brew install postgresql
      brew services start postgresql
   
   Or use PostgreSQL.app: https://postgresapp.com/

3. Visual Studio Code (optional but recommended)
   Download: https://code.visualstudio.com/

4. VS Code Extensions (if using VS Code):
   - Python (Microsoft)
   - Pylance (Microsoft)
   - Psycopg/PostgreSQL Tools (optional)
"""

# ==============================================================================
# STEP 2: SETUP PROJECT ENVIRONMENT
# ==============================================================================

"""
Open Terminal and navigate to project:

$ cd /Users/sepia/KnowMySocials

Create Python virtual environment:

$ python3 -m venv venv

Activate virtual environment:

$ source venv/bin/activate

You should see (venv) prefix in your terminal.
"""

# ==============================================================================
# STEP 3: INSTALL PYTHON DEPENDENCIES
# ==============================================================================

"""
With virtual environment activated:

$ pip install --upgrade pip
$ pip install -r requirements.txt

Download spaCy language model (required for lemmatization):

$ python -m spacy download en_core_web_sm

This downloads a 40MB model - this is normal.
"""

# ==============================================================================
# STEP 4: CONFIGURE DATABASE CONNECTION
# ==============================================================================

"""
The project is pre-configured to connect to PostgreSQL.

Default settings (in config.py):
- Host: localhost
- Port: 5432
- Database: knowmysocials_db
- User: postgres
- Password: postgres

If your PostgreSQL setup differs, create a .env file:

$ cp .env.example .env

Then edit .env with your actual credentials:

$ nano .env

Example .env file:
---
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowmysocials_db
DB_USER=postgres
DB_PASSWORD=your_secure_password
---

Save and exit (Ctrl+O, Enter, Ctrl+X in nano)
"""

# ==============================================================================
# STEP 5: CREATE DATABASE
# ==============================================================================

"""
Connect to PostgreSQL:

$ psql -U postgres

At the PostgreSQL prompt (postgres=#), create database:

postgres=# CREATE DATABASE knowmysocials_db;

Verify creation:

postgres=# \\l

You should see knowmysocials_db in the list.

Exit PostgreSQL:

postgres=# \\q
"""

# ==============================================================================
# STEP 6: RUN THE PIPELINE
# ==============================================================================

"""
Ensure virtual environment is still activated (you should see (venv) prefix).

From the project root directory:

$ python main.py

The pipeline will execute:
1. PHASE 1: DATA COLLECTION
   - Loads 5500+ reviews from data/honda_reviews.csv
   - Validates data schema
   - Checks data quality

2. PHASE 2: DATA INGESTION - RAW DATA
   - Creates PostgreSQL tables
   - Stores raw reviews

3. PHASE 3: PREPROCESSING
   - Cleans text (removes URLs, HTML, special chars)
   - Normalizes Hinglish terms
   - Tokenizes and lemmatizes text
   - Calculates text metrics

4. PHASE 4: DATA INGESTION - PREPROCESSED DATA
   - Stores preprocessed reviews
   - Stores metadata (token counts, Hinglish scores)

Expected execution time: 5-15 minutes (depending on system performance)
"""

# ==============================================================================
# STEP 7: VERIFY RESULTS
# ==============================================================================

"""
After successful execution, verify data was stored:

Option A: View logs
$ tail -50 logs/pipeline.log

Option B: Query database
$ psql -U postgres -d knowmysocials_db

Check record counts:

knowmysocials_db=# SELECT COUNT(*) FROM raw_reviews;
knowmysocials_db=# SELECT COUNT(*) FROM preprocessed_reviews;
knowmysocials_db=# SELECT COUNT(*) FROM review_metadata;

View sample preprocessed review:

knowmysocials_db=# SELECT review_id, original_text, preprocessed_text, 
                         hinglish_score 
                  FROM preprocessed_reviews LIMIT 1;

Exit database:

knowmysocials_db=# \\q
"""

# ==============================================================================
# STEP 8: EXPLORE IN VISUAL STUDIO CODE (Optional)
# ==============================================================================

"""
1. Open VS Code
2. Click File → Open Folder
3. Navigate to /Users/sepia/KnowMySocials
4. Click Open

Configure Python Interpreter:
1. Press Cmd+Shift+P
2. Type "Python: Select Interpreter"
3. Choose the one with "./venv/bin/python"

Run main.py in VS Code:
1. Click Run → Run Without Debugging (or press Ctrl+F5)
2. View output in the Terminal panel at the bottom

Debug main.py:
1. Set breakpoints by clicking on line numbers
2. Click Run → Start Debugging (or press F5)
3. Use Debug panel to inspect variables, step through code
"""

# ==============================================================================
# STEP 9: NEXT STEPS - EXTENDING THE PIPELINE
# ==============================================================================

"""
The pipeline architecture supports easy extensions:

1. ADD NEW PREPROCESSING STEPS:
   - Edit src/preprocessing.py
   - Add method to TextPreprocessor class
   - Call it in preprocess_single_text()

2. ADD SENTIMENT ANALYSIS:
   - Create src/analysis.py with SentimentAnalyzer class
   - Call from main.py after preprocessing

3. ADD FEATURE EXTRACTION:
   - Create src/features.py
   - Extract TF-IDF, word embeddings, etc.
   - Store in new database tables

4. ADD DATA VISUALIZATION:
   - Create src/visualization.py
   - Use matplotlib or plotly for charts
   - Visualize token distributions, Hinglish scores, etc.

5. ADD BATCH PROCESSING:
   - Modify main.py to accept command-line arguments
   - Process multiple CSV files
   - Support scheduled batch runs

6. ADD API LAYER:
   - Create Flask/FastAPI app
   - Expose preprocessing endpoints
   - Run as web service
"""

# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

"""
PROBLEM: "command not found: python3"
SOLUTION: Install Python from https://www.python.org/downloads/

PROBLEM: "psql: command not found"
SOLUTION: Install PostgreSQL from https://www.postgresql.org/download/

PROBLEM: "psycopg2: could not connect to server: Connection refused"
SOLUTION: 
  - Check if PostgreSQL is running: brew services list
  - Start PostgreSQL: brew services start postgresql
  - Or use PostgreSQL.app

PROBLEM: "ModuleNotFoundError: No module named 'pandas'"
SOLUTION:
  - Ensure virtual environment is activated (see (venv) prefix)
  - Run: pip install -r requirements.txt

PROBLEM: "spaCy model not found"
SOLUTION:
  - Run: python -m spacy download en_core_web_sm

PROBLEM: "Database 'knowmysocials_db' does not exist"
SOLUTION:
  - Connect to PostgreSQL: psql -U postgres
  - Create database: CREATE DATABASE knowmysocials_db;

PROBLEM: "Permission denied" when accessing logs/
SOLUTION:
  - Fix permissions: chmod -R 755 logs/
  - Or delete logs directory and let it auto-create
"""

# ==============================================================================
# USEFUL COMMANDS
# ==============================================================================

"""
# Activate virtual environment
source venv/bin/activate

# Deactivate virtual environment
deactivate

# Install specific package
pip install package_name==version

# Upgrade pip
pip install --upgrade pip

# View installed packages
pip list

# Start PostgreSQL (macOS)
brew services start postgresql

# Stop PostgreSQL (macOS)
brew services stop postgresql

# Connect to PostgreSQL
psql -U postgres

# List all databases
psql -U postgres -l

# Run Python script
python main.py

# Run Python in interactive mode
python

# Check Python version
python --version

# Find Python location
which python

# Check disk usage
du -sh /Users/sepia/KnowMySocials

# View file sizes
ls -lh data/
ls -lh logs/
"""

# ==============================================================================
# QUICK REFERENCE: PROJECT STRUCTURE
# ==============================================================================

"""
/Users/sepia/KnowMySocials/
│
├── data/
│   └── honda_reviews.csv              [5500+ review records]
│
├── src/
│   ├── __init__.py                    [Package initialization]
│   ├── collection.py                  [Load data from CSV]
│   ├── ingestion.py                   [Store in PostgreSQL]
│   ├── preprocessing.py               [Clean and preprocess text]
│   └── utils.py                       [Logging, DB utilities]
│
├── logs/
│   └── pipeline.log                   [Execution logs]
│
├── main.py                            [Main pipeline orchestrator - RUN THIS]
├── config.py                          [Configuration constants]
├── requirements.txt                   [Python dependencies]
├── .env.example                       [Template for .env]
├── .env                               [Your database credentials - CREATE THIS]
├── .gitignore                         [Git ignore patterns]
└── README.md                          [Full documentation]
"""

# ==============================================================================
# EXPECTED OUTPUT
# ==============================================================================

"""
When you run 'python main.py', you should see:

======================================================================
  KNOWMYSOCIALS NLP PIPELINE - STARTING
======================================================================

======================================================================
  PHASE 1: DATA COLLECTION
======================================================================

2024-06-02 14:45:32,123 - src.collection - INFO - Loading CSV file: data/honda_reviews.csv
2024-06-02 14:45:32,456 - src.collection - INFO - Successfully loaded 5500 rows from CSV
100%|██████████| 5500/5500 [00:05<00:00, 1000.00it/s]
✓ Collected 5500 reviews
✓ Memory usage: 2.45 MB

======================================================================
  PHASE 2: DATA INGESTION - RAW DATA
======================================================================

[Progress bar showing ingestion...]
✓ Ingested raw reviews into PostgreSQL
✓ Total records in raw_reviews: 5500

======================================================================
  PHASE 3: PREPROCESSING
======================================================================

[Progress bar showing preprocessing...]
✓ Preprocessed 5500 reviews

======================================================================
  PHASE 4: DATA INGESTION - PREPROCESSED DATA
======================================================================

[Progress bars for ingestion...]
✓ Ingested 5500 preprocessed reviews
✓ Ingested metadata for 5500 reviews

======================================================================
  PIPELINE EXECUTION SUMMARY
======================================================================

Raw Reviews Table: 5500 records
Preprocessed Reviews Table: 5500 records
Metadata Table: 5500 records

======================================================================
  SAMPLE PREPROCESSED REVIEW (First record)
======================================================================

Review ID: REV00001
Model: Shine 125

Original Text:
  Shine 125 leke life set ho gayi bhai. Mileage 55+ aa raha hai...

Cleaned Text:
  shine kilometers_per_liter life set fuel_efficiency aa raha hai...

Tokens (28 total, 22 unique):
  shine kilometers_per_liter life set fuel_efficiency aa raha...

Lemmas:
  shine kilometer_per_liter life set fuel_efficiency aa raha...

Metadata:
  Text Length: 150 characters
  Hinglish Score: 45%

======================================================================
  PIPELINE COMPLETED SUCCESSFULLY ✓
======================================================================
"""

# ==============================================================================
# SUPPORT & RESOURCES
# ==============================================================================

"""
Documentation:
- README.md - Full project documentation
- config.py - Configuration options and explanations
- Each Python file has docstrings explaining functions/classes

Official Resources:
- Python: https://docs.python.org/3/
- pandas: https://pandas.pydata.org/docs/
- SQLAlchemy: https://docs.sqlalchemy.org/
- spaCy: https://spacy.io/usage
- NLTK: https://www.nltk.org/howto/

PostgreSQL:
- psql Commands: https://www.postgresql.org/docs/current/app-psql.html
- SQL Docs: https://www.postgresql.org/docs/current/

VS Code:
- Python Extension Guide: https://marketplace.visualstudio.com/items?itemName=ms-python.python
- Debugging: https://code.visualstudio.com/docs/python/debugging
"""

print(__doc__)
