"""
Data Ingestion Module
Handles storing raw and metadata into PostgreSQL database.
Creates proper schema and manages database operations.
"""

import io
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any, cast, List, Dict
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    LargeBinary,
    Float,
    Boolean,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import text
from tqdm import tqdm

import config
from src.utils import setup_logger, get_db_engine, get_db_connection


logger = setup_logger(__name__)
Base = declarative_base()


# --- Minimal 4-table schema implemented below ---


class RawReview(Base):
    """Raw original reviews. Small table with one row per source review.

    Kept minimal: stores original text and source metadata. Primary key is
    `review_id` (unique). This table mirrors the original incoming CSV and is
    write-once/idempotent.
    """
    __tablename__ = config.RAW_DATA_TABLE

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(50), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    review_date = Column(DateTime, nullable=True)
    location = Column(String(200))
    user_type = Column(String(100))
    source = Column(String(100))
    ownership_months = Column(Integer)
    review_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ProcessedReview(Base):
    """Central wide table that stores ALL derived information for a review.

    This replaces multiple specialized tables by using JSONB for extensible
    fields (rule topics/tags, topic keywords) and a binary column for
    embeddings. Fields are nullable so partial processing is supported and the
    pipeline can incrementally update rows (idempotent upserts).
    """
    __tablename__ = config.PROCESSED_DATA_TABLE

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(50), unique=True, nullable=False, index=True)

    # Preprocessing
    original_text = Column(Text, nullable=False)
    cleaned_text = Column(Text)
    tokens = Column(JSONB)  # list[str]
    lemmas = Column(JSONB)  # list[str]
    text_length = Column(Integer)
    token_count = Column(Integer)
    unique_tokens = Column(Integer)
    hinglish_score = Column(Integer)

    # Hybrid workflow
    is_complex = Column(Boolean, nullable=False, default=True, index=True)
    rule_topics = Column(JSONB)  # list[str]
    rule_tags = Column(JSONB)    # list[str]
    rule_summary = Column(Text)

    # Embedding stored as binary blob (efficient for Postgres BYTEA)
    embedding = Column(LargeBinary)
    embedding_dim = Column(Integer)

    # Topic/cluster outputs (single assignment stored here)
    topic_id = Column(Integer, index=True)
    topic_label = Column(String(150))
    topic_probability = Column(Float)  # type: ignore
    topic_keywords = Column(JSONB)

    cluster_label = Column(Integer)
    cluster_probability = Column(Float)  # type: ignore
    is_noise = Column(Boolean, default=False)

    # Metadata
    processed_at = Column(DateTime)
    version = Column(String(32))


class TopicMaster(Base):
    """Master list of discovered topics. Avoids duplicates and stores keywords."""
    __tablename__ = config.TOPIC_MASTER_TABLE

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, unique=True, nullable=False, index=True)
    topic_name = Column(String(150))
    keywords = Column(JSONB)
    description = Column(Text)
    first_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PipelineRun(Base):
    """Optional table to track pipeline executions and metrics."""
    __tablename__ = config.PIPELINE_RUNS_TABLE

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), unique=True, nullable=False, index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime)
    total_reviews = Column(Integer)
    complex_reviews = Column(Integer)
    processing_time = Column(Float)  # type: ignore
    status = Column(String(50))


# Create helpful indexes (some duplicated via Column(index=True) above but kept explicit)
Index("ix_processed_review_review_id", ProcessedReview.review_id)
Index("ix_processed_review_topic_id", ProcessedReview.topic_id)
Index("ix_processed_review_is_complex", ProcessedReview.is_complex)


class DatabaseIngester:
    """Handles database operations and data ingestion."""
    
    def __init__(self):
        """Initialize database ingester."""
        self.logger = setup_logger(__name__)
        self.engine = get_db_engine()
    
    def create_tables(self) -> bool:
        """
        Create database tables if they don't exist.
        
        Returns:
            True if tables created successfully
            
        Raises:
            Exception: If table creation fails
        """
        try:
            self.logger.info("Creating database tables...")
            Base.metadata.create_all(self.engine)
            self.logger.info("Tables created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    def drop_tables(self, confirm: bool = False) -> bool:
        """
        Drop all tables (use with caution).
        
        Args:
            confirm: Must be True to actually drop tables
            
        Returns:
            True if tables dropped
        """
        if not confirm:
            self.logger.warning("Drop tables called without confirmation. Skipping.")
            return False
        
        try:
            self.logger.warning("Dropping all tables...")
            Base.metadata.drop_all(self.engine)
            self.logger.info("Tables dropped successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error dropping tables: {e}")
            raise
    
    def ingest_raw_data(self, df: pd.DataFrame) -> int:
        """
        Ingest raw review data into PostgreSQL.
        
        Args:
            df: DataFrame containing raw review data
            
        Returns:
            Number of records ingested
            
        Raises:
            Exception: If ingestion fails
        """
        self.logger.info(f"Ingesting {len(df)} raw reviews into database...")
        
        # Create tables if they don't exist
        self.create_tables()
        
        try:
            ingested_count = 0
            failed_count = 0
            
            with get_db_connection() as session:
                records = cast(List[Dict[str, Any]], cast(Any, df.to_dict(orient="records")))  # type: ignore[reportUnknownMemberType]
                for idx, row in enumerate(tqdm(records, total=len(records), desc="Ingesting raw data")):
                    try:
                        # Parse review date
                        review_date_raw: Any = row.get("Review_Date")
                        review_date = datetime.now(timezone.utc)
                        if isinstance(review_date_raw, datetime):
                            review_date = review_date_raw
                        elif isinstance(review_date_raw, str) and review_date_raw.strip():
                            try:
                                review_date = datetime.fromisoformat(review_date_raw)
                            except ValueError:
                                try:
                                    review_date = datetime.strptime(review_date_raw, "%Y-%m-%d %H:%M:%S")
                                except ValueError:
                                    self.logger.debug(f"Unrecognized Review_Date format: {review_date_raw}")
                        
                        # Create review object
                        review = RawReview(
                            review_id=row["Review_ID"],
                            model_name=row.get("Model_Name", "Unknown"),
                            review_date=review_date,
                            location=row.get("Location", None),
                            user_type=row.get("User_Type", None),
                            source=row.get("Source", None),
                            ownership_months=self._parse_ownership_months(row.get("Ownership_Months", 0)),
                            review_text=row["Review_Text"]
                        )
                        
                        # Skip if review already exists (idempotent ingest)
                        exists = session.query(RawReview).filter_by(review_id=row["Review_ID"]).first()
                        if exists:
                            self.logger.debug(f"Skipping existing review {row['Review_ID']}")
                        else:
                            session.add(review)
                            ingested_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to ingest row {idx}: {e}")
                        failed_count += 1
                        continue
                
                session.commit()
            
            self.logger.info(f"Raw data ingestion completed: {ingested_count} successful, {failed_count} failed")
            return ingested_count
            
        except Exception as e:
            self.logger.error(f"Error during raw data ingestion: {e}")
            raise
    
    def _parse_ownership_months(self, raw_value: Any) -> int | None:
        """Safely parse ownership months from raw input."""
        if raw_value is None or raw_value == "":
            return None
        try:
            return int(raw_value)
        except (ValueError, TypeError):
            return None
    
    def ingest_preprocessed_data(self, df: pd.DataFrame) -> int:
        """
        Ingest preprocessed review data into PostgreSQL.
        
        Args:
            df: DataFrame containing preprocessed data
                Expected columns: review_id, original_text, preprocessed_text, 
                                  tokens, lemmas, model_name
            
        Returns:
            Number of records ingested
        """
        self.logger.info(f"Ingesting {len(df)} preprocessed reviews into database...")
        
        try:
            ingested_count = 0

            with get_db_connection() as session:
                records = cast(List[Dict[str, Any]], cast(Any, df.to_dict(orient="records")))  # type: ignore[reportUnknownMemberType]
                for idx, row in enumerate(tqdm(records, total=len(records), desc="Ingesting preprocessed data")):
                    try:
                        review_id = row.get("review_id")
                        if not review_id:
                            self.logger.warning(f"Missing review_id for preprocessed row {idx}")
                            continue

                        # Upsert into central ProcessedReview table
                        existing = session.query(ProcessedReview).filter_by(review_id=review_id).first()
                        if existing:
                            # Use cast to appease static type checkers when assigning ORM Columns
                            er = cast(Any, existing)
                            # Update fields that are part of preprocessing
                            er.original_text = row.get("original_text", er.original_text)
                            er.cleaned_text = row.get("preprocessed_text", er.cleaned_text)
                            er.tokens = row.get("tokens") if row.get("tokens") is not None else er.tokens
                            er.lemmas = row.get("lemmas") if row.get("lemmas") is not None else er.lemmas
                            er.is_complex = bool(row.get("is_complex", er.is_complex))
                            er.rule_topics = row.get("rule_topics") if row.get("rule_topics") is not None else er.rule_topics
                            er.rule_tags = row.get("rule_tags") if row.get("rule_tags") is not None else er.rule_tags
                            er.rule_summary = row.get("rule_summary", er.rule_summary)
                        else:
                            processed = ProcessedReview(
                                review_id=review_id,
                                original_text=row.get("original_text", ""),
                                cleaned_text=row.get("preprocessed_text", ""),
                                tokens=row.get("tokens"),
                                lemmas=row.get("lemmas"),
                                is_complex=bool(row.get("is_complex", True)),
                                rule_topics=row.get("rule_topics"),
                                rule_tags=row.get("rule_tags"),
                                rule_summary=row.get("rule_summary", ""),
                                processed_at=datetime.now(timezone.utc),
                                version=getattr(config, "PIPELINE_VERSION", "v1")
                            )
                            session.add(processed)
                            ingested_count += 1

                    except Exception as e:
                        self.logger.warning(f"Failed to ingest preprocessed row {idx}: {e}")
                        continue

                session.commit()

            self.logger.info(f"Preprocessed data ingestion completed: {ingested_count} records")
            return ingested_count

        except Exception as e:
            self.logger.error(f"Error during preprocessed data ingestion: {e}")
            raise
    
    def ingest_metadata(self, df: pd.DataFrame) -> int:
        """
        Ingest review metadata into PostgreSQL.
        
        Args:
            df: DataFrame containing metadata
                Expected columns: review_id, text_length, token_count, 
                                  unique_tokens, language_detected, hinglish_score
            
        Returns:
            Number of metadata records ingested
        """
        self.logger.info(f"Ingesting metadata for {len(df)} reviews...")
        
        try:
            ingested_count = 0

            with get_db_connection() as session:
                records = cast(List[Dict[str, Any]], cast(Any, df.to_dict(orient="records")))  # type: ignore[reportUnknownMemberType]
                for idx, row in enumerate(tqdm(records, total=len(records), desc="Ingesting metadata")):
                    try:
                        review_id = row.get("review_id")
                        if not review_id:
                            self.logger.warning(f"Missing review_id for metadata row {idx}")
                            continue

                        existing = session.query(ProcessedReview).filter_by(review_id=review_id).first()
                        if existing:
                            er = cast(Any, existing)
                            er.text_length = int(row.get("text_length", er.text_length or 0))
                            er.token_count = int(row.get("token_count", er.token_count or 0))
                            er.unique_tokens = int(row.get("unique_tokens", er.unique_tokens or 0))
                            er.hinglish_score = int(row.get("hinglish_score", er.hinglish_score or 0))
                            er.processed_at = datetime.now(timezone.utc)
                        else:
                            # create a lightweight processed row so metadata has a place
                            processed = ProcessedReview(
                                review_id=review_id,
                                original_text=row.get("original_text", ""),
                                text_length=int(row.get("text_length", 0)),
                                token_count=int(row.get("token_count", 0)),
                                unique_tokens=int(row.get("unique_tokens", 0)),
                                hinglish_score=int(row.get("hinglish_score", 0)),
                                processed_at=datetime.now(timezone.utc),
                                version=getattr(config, "PIPELINE_VERSION", "v1")
                            )
                            session.add(processed)
                            ingested_count += 1

                    except Exception as e:
                        self.logger.warning(f"Failed to ingest metadata row {idx}: {e}")
                        continue

                session.commit()

            self.logger.info(f"Metadata ingestion completed: {ingested_count} records")
            return ingested_count

        except Exception as e:
            self.logger.error(f"Error during metadata ingestion: {e}")
            raise

    def fetch_preprocessed_reviews(self) -> pd.DataFrame:
        """Fetch preprocessed review text from PostgreSQL."""
        try:
            # Fetch useful fields including hybrid markers so callers can decide
            # which reviews need AI processing.
            query_text = (
                f"SELECT review_id, cleaned_text as preprocessed_text, /*model_name,*/ is_complex, "
                f"rule_topics, rule_tags, rule_summary FROM {config.PROCESSED_DATA_TABLE} ORDER BY id"
            )
            # Execute using SQLAlchemy and convert results to DataFrame to avoid DB-API cursor issues.
            with self.engine.connect() as conn:
                result = conn.execute(text(query_text))
                rows = result.fetchall()
                cols = list(result.keys())
            # Convert SQLAlchemy Row objects to sequences and ensure columns is a list
            df = pd.DataFrame(rows, columns=cols)
            self.logger.info(f"Fetched {len(df)} preprocessed reviews from database")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching preprocessed reviews: {e}")
            raise

    def fetch_complex_preprocessed_reviews(self) -> pd.DataFrame:
        """Fetch only the reviews marked as complex (require AI processing)."""
        try:
            query_text = (
                f"SELECT review_id, cleaned_text as preprocessed_text FROM {config.PROCESSED_DATA_TABLE} "
                f"WHERE is_complex = true ORDER BY id"
            )
            with self.engine.connect() as conn:
                result = conn.execute(text(query_text))
                rows = result.fetchall()
                cols = list(result.keys())
            df = pd.DataFrame(rows, columns=cols)
            self.logger.info(f"Fetched {len(df)} complex preprocessed reviews from database")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching complex preprocessed reviews: {e}")
            raise

    def _serialize_embedding(self, embedding: np.ndarray[Any, Any]) -> bytes:
        """Serialize numpy embedding into bytes for binary storage."""
        buffer = io.BytesIO()
        np.save(buffer, embedding, allow_pickle=False)
        return buffer.getvalue()

    def ingest_embeddings(self, review_ids: list[str], embeddings: np.ndarray[Any, Any]) -> int:
        """Store review embeddings in PostgreSQL."""
        if len(review_ids) != embeddings.shape[0]:
            raise ValueError("review_ids length must match number of embeddings")

        self.logger.info(f"Ingesting embeddings for {len(review_ids)} reviews...")
        ingested_count = 0
        with get_db_connection() as session:
            for review_id, embedding in tqdm(zip(review_ids, embeddings), total=len(review_ids), desc="Ingesting embeddings"):
                try:
                    existing = session.query(ProcessedReview).filter_by(review_id=review_id).first()
                    serialized = self._serialize_embedding(embedding)
                    if existing:
                        er = cast(Any, existing)
                        er.embedding = serialized
                        er.embedding_dim = int(embedding.shape[0])
                    else:
                        processed = ProcessedReview(
                            review_id=review_id,
                            original_text="",
                            embedding=serialized,
                            embedding_dim=int(embedding.shape[0]),
                            processed_at=datetime.now(timezone.utc),
                            version=getattr(config, "PIPELINE_VERSION", "v1")
                        )
                        session.add(processed)
                        ingested_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to ingest embedding for {review_id}: {e}")
                    continue

            session.commit()

        self.logger.info(f"Embedding ingestion completed: {ingested_count} records")
        return ingested_count

    def ingest_topic_assignments(self, df: pd.DataFrame) -> int:
        """Store topic assignments for each review."""
        self.logger.info(f"Ingesting topic assignments for {len(df)} reviews...")
        ingested_count = 0
        with get_db_connection() as session:
            records = cast(List[Dict[str, Any]], cast(Any, df.to_dict(orient="records")))  # type: ignore[reportUnknownMemberType]
            for idx, row in enumerate(tqdm(records, total=len(records), desc="Ingesting topics")):
                try:
                    review_id = row.get("review_id")
                    if not review_id:
                        self.logger.warning(f"Missing review_id for topic row {idx}")
                        continue

                    _tid = row.get("topic_id")
                    topic_id = int(_tid) if (_tid is not None and str(_tid).strip() != "") else None
                    topic_label = row.get("topic_label")
                    topic_prob = float(row.get("topic_probability", 0.0)) if row.get("topic_probability") is not None else None
                    topic_keywords = row.get("topic_keywords")

                    # Update ProcessedReview with assignment
                    existing = session.query(ProcessedReview).filter_by(review_id=review_id).first()
                    if existing:
                        er = cast(Any, existing)
                        er.topic_id = topic_id
                        er.topic_label = topic_label
                        er.topic_probability = topic_prob
                        er.topic_keywords = topic_keywords
                    else:
                        processed = ProcessedReview(
                            review_id=review_id,
                            original_text="",
                            topic_id=topic_id,
                            topic_label=topic_label,
                            topic_probability=topic_prob,
                            topic_keywords=topic_keywords,
                            processed_at=datetime.now(timezone.utc),
                            version=getattr(config, "PIPELINE_VERSION", "v1")
                        )
                        session.add(processed)
                        ingested_count += 1

                    # Ensure topic master has an entry
                    if topic_id is not None and topic_id >= 0:
                        tm = session.query(TopicMaster).filter_by(topic_id=topic_id).first()
                        if not tm:
                            master = TopicMaster(
                                topic_id=topic_id,
                                topic_name=topic_label or f"topic_{topic_id}",
                                keywords=topic_keywords,
                                description="",
                                first_seen=datetime.now(timezone.utc)
                            )
                            session.add(master)

                except Exception as e:
                    self.logger.warning(f"Failed to ingest topic row {idx}: {e}")
                    continue

            session.commit()

        self.logger.info(f"Topic assignment ingestion completed: {ingested_count} records")
        return ingested_count

    def ingest_topic_summary(self, df: pd.DataFrame) -> int:
        """Store topic summary metadata for each discovered topic."""
        self.logger.info(f"Ingesting topic summary for {len(df)} topics...")
        ingested_count = 0
        with get_db_connection() as session:
            records = cast(List[Dict[str, Any]], cast(Any, df.to_dict(orient="records")))  # type: ignore[reportUnknownMemberType]
            for idx, row in enumerate(tqdm(records, total=len(records), desc="Ingesting topic summary")):
                try:
                    _tid = row.get("topic_id")
                    topic_id = int(_tid) if (_tid is not None and str(_tid).strip() != "") else None
                    if topic_id is None:
                        self.logger.warning(f"Missing topic_id for summary row {idx}")
                        continue

                    topic_label = row.get("Name") or row.get("topic_label")
                    keywords = row.get("keywords")

                    existing = session.query(TopicMaster).filter_by(topic_id=topic_id).first()
                    if existing:
                        tm = cast(Any, existing)
                        tm.topic_name = topic_label or tm.topic_name
                        tm.keywords = keywords or tm.keywords
                    else:
                        master = TopicMaster(
                            topic_id=topic_id,
                            topic_name=topic_label or f"topic_{topic_id}",
                            keywords=keywords,
                            description="",
                            first_seen=datetime.now(timezone.utc)
                        )
                        session.add(master)
                        ingested_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to ingest topic summary row {idx}: {e}")
                    continue

            session.commit()

        self.logger.info(f"Topic summary ingestion completed: {ingested_count} topics")
        return ingested_count

    def ingest_cluster_assignments(self, df: pd.DataFrame) -> int:
        """Store cluster assignments for each review."""
        self.logger.info(f"Ingesting cluster assignments for {len(df)} reviews...")
        ingested_count = 0
        with get_db_connection() as session:
            records = cast(List[Dict[str, Any]], cast(Any, df.to_dict(orient="records")))  # type: ignore[reportUnknownMemberType]
            for idx, row in enumerate(tqdm(records, total=len(records), desc="Ingesting clusters")):
                try:
                    review_id = row.get("review_id")
                    if not review_id:
                        self.logger.warning(f"Missing review_id for cluster row {idx}")
                        continue

                    cluster_label = int(row.get("cluster_label", -1)) if row.get("cluster_label") is not None else None
                    cluster_prob = float(row.get("cluster_probability", 0.0)) if row.get("cluster_probability") is not None else None
                    is_noise = bool(row.get("is_noise", cluster_label == -1))

                    existing = session.query(ProcessedReview).filter_by(review_id=review_id).first()
                    if existing:
                        er = cast(Any, existing)
                        er.cluster_label = cluster_label
                        er.cluster_probability = cluster_prob
                        er.is_noise = is_noise
                    else:
                        processed = ProcessedReview(
                            review_id=review_id,
                            original_text="",
                            cluster_label=cluster_label,
                            cluster_probability=cluster_prob,
                            is_noise=is_noise,
                            processed_at=datetime.now(timezone.utc),
                            version=getattr(config, "PIPELINE_VERSION", "v1")
                        )
                        session.add(processed)
                        ingested_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to ingest cluster row {idx}: {e}")
                    continue

            session.commit()

        self.logger.info(f"Cluster ingestion completed: {ingested_count} records")
        return ingested_count

    def get_record_count(self, table_name: str) -> int:
        """
        Get count of records in a table.
        
        Args:
            table_name: Name of table to count
            
        Returns:
            Number of records in table
        """
        try:
            with get_db_connection() as session:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar_one()
                return int(count)
        except Exception as e:
            self.logger.error(f"Error getting record count from {table_name}: {e}")
            return 0


def ingest_data(df: pd.DataFrame) -> bool:
    """
    Main function to ingest raw data into PostgreSQL.
    Creates tables and stores data.
    
    Args:
        df: Raw DataFrame from data collection phase
        
    Returns:
        True if ingestion successful
        
    Raises:
        Exception: If ingestion fails
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION PHASE")
    logger.info("=" * 60)
    
    try:
        ingester = DatabaseIngester()
        
        # Create tables
        ingester.create_tables()
        
        # Ingest raw data
        ingested_count = ingester.ingest_raw_data(df)
        
        logger.info(f"Ingested {ingested_count} reviews into raw data table")
        logger.info("Data ingestion phase completed successfully")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise
