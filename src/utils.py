"""
Utility functions for logging, database connections, and helpers.
Provides common utilities used across the NLP pipeline.
"""

import logging
import sys
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

import config


def setup_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(config.LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_db_engine():
    """
    Create and return SQLAlchemy engine for PostgreSQL connection.
    
    Returns:
        SQLAlchemy engine instance
        
    Raises:
        Exception: If database connection fails
    """
    # The application uses PostgreSQL via the `psycopg2` DBAPI. `config.DATABASE_URL`
    # must therefore be of the form:
    #   postgresql+psycopg2://user:password@host:port/dbname
    # We configure a sensible connection pool for production workloads and enable
    # `pool_pre_ping` to avoid stale connections in long-running processes.
    try:
        engine = create_engine(
            config.DATABASE_URL,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connection before use
            pool_size=10,
            max_overflow=20,
            future=True,
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        logger = setup_logger(__name__)
        logger.error(f"Failed to connect to database: {e}")
        raise


def get_db_session():
    """
    Get a database session factory.
    
    Returns:
        SQLAlchemy sessionmaker instance
    """
    engine = get_db_engine()
    return sessionmaker(bind=engine)


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Automatically handles connection and session cleanup.
    
    Usage:
        with get_db_connection() as session:
            # Use session
    """
    engine = get_db_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger = setup_logger(__name__)
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def validate_csv_exists(filepath: Path) -> bool:
    """
    Validate that CSV file exists and is readable.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        True if file exists and is readable, False otherwise
    """
    logger = setup_logger(__name__)
    
    if not filepath.exists():
        logger.error(f"CSV file not found: {filepath}")
        return False
    
    if not filepath.is_file():
        logger.error(f"Path is not a file: {filepath}")
        return False
    
    if not os.access(filepath, os.R_OK):
        logger.error(f"CSV file is not readable: {filepath}")
        return False
    
    logger.info(f"CSV file validated: {filepath}")
    return True
