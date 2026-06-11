"""
Data Collection Module
Handles loading and validation of raw data from CSV files.
Supports large file processing with pandas chunking.
"""

import pandas as pd
from pathlib import Path
from typing import Generator, Tuple, List, Dict, Any

import config
from src.utils import setup_logger, validate_csv_exists


logger = setup_logger(__name__)


class DataCollector:
    """
    Handles data collection from CSV files.
    Supports efficient reading of large files using chunking.
    """
    
    def __init__(self, csv_path: Path = config.CSV_FILE_PATH):
        """
        Initialize DataCollector.
        
        Args:
            csv_path: Path to CSV file to read
        """
        self.csv_path = csv_path
        self.logger = setup_logger(__name__)
        
    def load_full_data(self) -> pd.DataFrame:
        """
        Load entire CSV file into DataFrame.
        Use this for smaller files (< 1GB).
        
        Returns:
            pandas DataFrame containing all CSV data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            pd.errors.ParserError: If CSV is malformed
        """
        if not validate_csv_exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        try:
            self.logger.info(f"Loading CSV file: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            self.logger.info(f"Successfully loaded {len(df)} rows from CSV")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
        except pd.errors.ParserError as e:
            self.logger.error(f"Error parsing CSV file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading CSV: {e}")
            raise
    
    def load_chunked_data(self, chunk_size: int = config.CHUNK_SIZE) -> Generator[pd.DataFrame, None, None]:
        """
        Load CSV file in chunks for memory-efficient processing.
        Ideal for large files (5500+ rows).
        
        Args:
            chunk_size: Number of rows per chunk (default: 1000)
            
        Yields:
            pandas DataFrame chunks
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            pd.errors.ParserError: If CSV is malformed
        """
        if not validate_csv_exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        try:
            self.logger.info(f"Loading CSV in chunks of {chunk_size} rows: {self.csv_path}")
            
            chunk_reader = pd.read_csv(self.csv_path, chunksize=chunk_size)
            total_rows = 0
            
            for chunk_num, chunk in enumerate(chunk_reader, 1):
                total_rows += len(chunk)
                self.logger.info(f"Loaded chunk {chunk_num} ({len(chunk)} rows, total: {total_rows})")
                yield chunk
            
            self.logger.info(f"Completed loading {total_rows} total rows in chunks")
            
        except pd.errors.ParserError as e:
            self.logger.error(f"Error parsing CSV file during chunked load: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during chunked data load: {e}")
            raise
    
    def validate_data_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame contains expected columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid: bool, missing_columns: list)
        """
        required_columns = {
            "Review_ID", "Model_Name", "Review_Date", 
            "Location", "User_Type", "Source", "Ownership_Months", "Review_Text"
        }
        
        actual_columns = set(df.columns)
        missing_columns = list(required_columns - actual_columns)
        
        if missing_columns:
            self.logger.warning(f"Missing columns: {missing_columns}")
            return False, missing_columns
        
        self.logger.info(f"Data schema validation successful. Columns: {list(df.columns)}")
        return True, []  # type: ignore
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report: Dict[str, Any] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        self.logger.info(f"Data Quality Report: {quality_report}")
        
        # Log warnings for data quality issues
        for col, null_count in quality_report["null_counts"].items():
            if null_count > 0:
                percentage = (null_count / len(df)) * 100
                self.logger.warning(f"Column '{col}' has {null_count} null values ({percentage:.2f}%)")
        
        if quality_report["duplicate_rows"] > 0:
            self.logger.warning(f"Found {quality_report['duplicate_rows']} duplicate rows")
        
        return quality_report


def collect_data(csv_path: Path = config.CSV_FILE_PATH) -> pd.DataFrame:
    """
    Main function to collect data from CSV.
    Validates data and returns DataFrame.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Validated pandas DataFrame
        
    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If data schema or quality checks fail
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA COLLECTION PHASE")
    logger.info("=" * 60)
    
    collector = DataCollector(csv_path)
    
    # Load data
    df = collector.load_full_data()
    
    # Validate schema
    is_valid, missing_cols = collector.validate_data_schema(df)
    if not is_valid:
        raise ValueError(f"Data validation failed. Missing columns: {missing_cols}")
    
    # Check data quality
    quality_report = collector.check_data_quality(df)
    logger.info(f"Data quality summary: {quality_report}")
    
    logger.info("Data collection phase completed successfully")
    logger.info("=" * 60)
    
    return df
