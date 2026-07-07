from concurrent.futures import ThreadPoolExecutor
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import os
import re
import threading
import uuid

import numpy as np
import pandas as pd

SentimentDict = Dict[str, Any]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(
    title="KnowMySocials Backend",
    description="FastAPI backend for the KnowMySocials Streamlit frontend.",
    version="1.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=8)
analysis_store: Dict[str, Dict[str, Any]] = {}
store_lock = threading.Lock()

POSITIVE_WORDS = {
    "great",
    "excellent",
    "love",
    "best",
    "smooth",
    "reliable",
    "fantastic",
    "clean",
    "comfortable",
    "stylish",
    "feature",
    "value",
    "fast",
    "efficient",
    "support",
    "helpful",
    "happy",
}
NEGATIVE_WORDS = {
    "bad",
    "slow",
    "noise",
    "expensive",
    "issue",
    "problem",
    "delay",
    "poor",
    "broken",
    "late",
    "wait",
    "service",
    "complaint",
    "annoying",
    "worse",
    "bekar", 
    "kharab", 
    "disappointed", 
    "waste", 
    "problem", 
    "issue", 
    "rust",
    "heat", 
    "vibration", 
    "loose", 
    "hard", 
    "expensive", 
    "pathetic", 
    "frustrating",
    "weak", 
    "noise", 
    "uncomfortable", 
    "dim", 
    "shame", 
    "headache", 
    "nightmare",
    "terrible", 
    "worst", 
    "bad", 
    "poor", 
    "cheap", 
    "rude", 
    "टूट", 
    "टूटा", 
    "बेकार",
    "खराब"
}
SERVICE_KEYWORDS = ["service", "repair", "maintenance", "warranty", "support"]
VALUE_KEYWORDS = ["value", "price", "money", "budget", "worth"]
MARKETING_ANGLES = ["safety", "connectivity", "design", "fuel", "comfort"]


class ReviewItem(BaseModel):
    review_id: Optional[str] = Field(default=None, example="1")  # type: ignore[call-arg]
    text: str = Field(..., example="The new ride is very comfortable and smooth.")  # type: ignore[call-arg]
    region: Optional[str] = Field(default="Unknown", example="North")  # type: ignore[call-arg]
    model: Optional[str] = Field(default="Unknown", example="Zeta")  # type: ignore[call-arg]
    feature: Optional[str] = Field(default="General", example="Safety")  # type: ignore[call-arg]
    campaign: Optional[str] = Field(default="Q3 Launch", example="Q3 Launch")  # type: ignore[call-arg]
    price_tier: Optional[str] = Field(default="Midrange", example="Midrange")  # type: ignore[call-arg]


class AnalyzeRequest(BaseModel):
    reviews: List[ReviewItem]


class AnalyzeResponse(BaseModel):
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str


class TaskResult(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    data: Optional[Dict[str, Any]] = None


def update_task(task_id: str, **kwargs: Any) -> None:
    with store_lock:
        if task_id in analysis_store:
            analysis_store[task_id].update(kwargs)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [token for token in text.split() if len(token) > 2]


def generate_embedding(text: str, dim: int = 64) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big", signed=False) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.standard_normal(dim).tolist()


def compute_sentiment(text: str) -> Dict[str, Any]:
    tokens = tokenize(clean_text(text))
    positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    total = max(1, len(tokens))
    score = (positive - negative) / total
    label = "positive" if score >= 0.0 else "negative"
    return {
        "score": round(float(score), 3),
        "label": label,
        "positive_terms": [token for token in tokens if token in POSITIVE_WORDS],
        "negative_terms": [token for token in tokens if token in NEGATIVE_WORDS],
    }


def get_sentiment_label(sentiment: SentimentDict) -> str:
    return str(sentiment.get("label", "negative"))


def get_sentiment_score(sentiment: SentimentDict) -> float:
    return float(sentiment.get("score", 0.0))


def serialize_review(review: ReviewItem) -> Dict[str, Any]:
    if hasattr(review, "model_dump"):
        return review.model_dump()
    return review.dict()  # type: ignore[call-arg]


def preprocess_review(item: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = clean_text(str(item.get("text", "")))
    return {
        **item,
        "clean_text": cleaned,
        "tokens": tokenize(cleaned),
    }


def review_embedding(item: Dict[str, Any]) -> List[float]:
    return generate_embedding(str(item.get("clean_text", "")))


def review_sentiment(item: Dict[str, Any]) -> Dict[str, Any]:
    return compute_sentiment(str(item.get("clean_text", "")))


def build_aggregations(df: pd.DataFrame) -> Dict[str, Any]:
    sentiment_list: List[SentimentDict] = [s if isinstance(s, dict) else {} for s in df["sentiment"].tolist()]
    sentiment_labels: List[str] = [get_sentiment_label(s) for s in sentiment_list]
    sentiment_scores: List[float] = [get_sentiment_score(s) for s in sentiment_list]

    df["sentiment_label"] = sentiment_labels
    df["sentiment_score"] = sentiment_scores

    total_reviews = len(df)
    positive = sum(1 for label in sentiment_labels if label == "positive")
    negative = sum(1 for label in sentiment_labels if label == "negative")
    average_score = round(float(sum(sentiment_scores) / max(1, len(sentiment_scores))), 3)

    summary: Dict[str, Any] = {
        "total_reviews": total_reviews,
        "positive": positive,
        "negative": negative,
        "average_score": average_score,
    }

    trend: List[Dict[str, Any]] = []
    for idx in range(len(sentiment_scores)):
        window = sentiment_scores[max(0, idx - 4) : idx + 1]
        trend.append({"step": idx + 1, "score": round(float(sum(window) / len(window)), 3)})

    keyword_counts = [token for text in df["clean_text"].astype(str).tolist() for token in tokenize(text)]
    keyword_counts_df = pd.Series(keyword_counts).value_counts().head(15).reset_index()
    keyword_counts_df.columns = ["keyword", "count"]
    keyword_counts_list = keyword_counts_df.to_dict(orient="records")

    feature_counts: Dict[Tuple[str, str], int] = {}
    feature_totals: Dict[str, int] = {}
    for feature, label in zip(df["feature"].astype(str).tolist(), sentiment_labels):
        feature_counts[(feature, label)] = feature_counts.get((feature, label), 0) + 1
        feature_totals[feature] = feature_totals.get(feature, 0) + 1

    feature_sentiment: List[Dict[str, Any]] = []
    for feature in sorted(feature_totals.keys()):
        feature_sentiment.append(
            {
                "feature": feature,
                "positive": feature_counts.get((feature, "positive"), 0),
                "negative": feature_counts.get((feature, "negative"), 0),
            }
        )

    aspect_heatmap: Dict[str, List[Any]] = {"x": [], "y": ["positive", "negative"], "z": []}
    if feature_sentiment:
        aspect_heatmap["x"] = [item["feature"] for item in feature_sentiment]
        for sentiment_label in ["positive", "negative"]:
            aspect_heatmap["z"].append([item.get(sentiment_label, 0) for item in feature_sentiment])

    region_scores: Dict[str, float] = {}
    region_counts: Dict[str, int] = {}
    for region, score in zip(df["region"].astype(str).tolist(), sentiment_scores):
        region_scores[region] = region_scores.get(region, 0.0) + score
        region_counts[region] = region_counts.get(region, 0) + 1
    region_issues: List[Dict[str, Any]] = [
        {"region": region, "average_score": round(region_scores[region] / region_counts[region], 3)}
        for region in sorted(region_scores)
    ]

    clean_texts = df["clean_text"].astype(str).tolist()
    service_issues: List[Dict[str, Any]] = []
    for keyword in SERVICE_KEYWORDS:
        count = sum(1 for text in clean_texts if keyword in text)
        if count > 0:
            service_issues.append({"issue": keyword, "count": count})

    negative_reviews: List[Dict[str, Any]] = [
        {
            "review_id": str(row["review_id"]),
            "text": str(row["text"]),
            "region": str(row["region"]),
            "model": str(row["model"]),
            "feature": str(row["feature"]),
            "sentiment_score": float(row["sentiment_score"]),
        }
        for row in df.to_dict(orient="records")
        if str(row.get("sentiment_label", "negative")) == "negative"
    ]
    negative_reviews.sort(key=lambda review: review["sentiment_score"])
    negative_reviews = negative_reviews[:5]

    model_performance_counts: Dict[Tuple[str, str], int] = {}
    for model, label in zip(df["model"].astype(str).tolist(), sentiment_labels):
        model_performance_counts[(model, label)] = model_performance_counts.get((model, label), 0) + 1
    model_performance: List[Dict[str, Any]] = [
        {
            "model": model,
            "positive": model_performance_counts.get((model, "positive"), 0),
            "negative": model_performance_counts.get((model, "negative"), 0),
        }
        for model in sorted({model for model, _ in model_performance_counts})
    ]

    forecast: List[Dict[str, Any]] = []
    if len(trend) > 2:
        last_scores = [step["score"] for step in trend[-3:]]
        forecast = [
            {"day": f"T+{i+1}", "forecast": round(float(last_scores[-1] + 0.02 * (i + 1)), 3)}
            for i in range(7)
        ]

    value_scores: List[Dict[str, Any]] = []
    for keyword in VALUE_KEYWORDS:
        frequency = sum(1 for text in clean_texts if keyword in text)
        if frequency > 0:
            value_scores.append({"keyword": keyword, "frequency": frequency})

    selling_points: List[Dict[str, Any]] = [
        {"point": keyword, "score": sum(1 for text in clean_texts if keyword in text)}
        for keyword in MARKETING_ANGLES
    ]

    feature_order = sorted(feature_totals.keys(), key=lambda item: (-feature_totals[item], item))
    campaign_suggestions = [
        {"campaign": feature, "suggestion": f"Promote {feature} with testimonials and short video content."}
        for feature in feature_order[:5]
    ]

    dealer_actions = [
        {"action": "Improve service desk responsiveness.", "priority": "High"},
        {"action": "Track regular dealer follow-ups.", "priority": "Medium"},
        {"action": "Prepare local issue escalation plans.", "priority": "Medium"},
    ]

    return {
        "summary": summary,
        "trend": trend,
        "keyword_counts": keyword_counts_list,
        "feature_sentiment": feature_sentiment,
        "aspect_heatmap": aspect_heatmap,
        "region_issues": region_issues,
        "service_issues": service_issues,
        "negative_reviews": negative_reviews,
        "model_performance": model_performance,
        "forecast": forecast,
        "value_scores": value_scores,
        "selling_points": selling_points,
        "campaign_suggestions": campaign_suggestions,
        "dealer_actions": dealer_actions,
    }


def write_csv_result(task_id: str, df: pd.DataFrame) -> None:
    path = os.path.join(RESULTS_DIR, f"analysis_{task_id}.csv")
    df.to_csv(path, index=False)


def run_full_analysis(task_id: str, reviews: List[Dict[str, Any]]) -> None:
    try:
        update_task(task_id, status="running", progress=5, message="Starting analysis")

        inputs: List[Dict[str, Any]] = []
        for idx, review in enumerate(reviews):
            inputs.append(
                {
                    "review_id": str(review.get("review_id") or idx + 1),
                    "text": str(review.get("text", "")),
                    "region": str(review.get("region", "Unknown")),
                    "model": str(review.get("model", "Unknown")),
                    "feature": str(review.get("feature", "General")),
                    "campaign": str(review.get("campaign", "General")),
                    "price_tier": str(review.get("price_tier", "Midrange")),
                }
            )

        update_task(task_id, progress=15, message="Preprocessing reviews")
        with ThreadPoolExecutor(max_workers=8) as pool:
            processed: List[Dict[str, Any]] = list(pool.map(preprocess_review, inputs))

        update_task(task_id, progress=30, message="Generating embeddings")
        with ThreadPoolExecutor(max_workers=8) as pool:
            embeddings: List[List[float]] = list(pool.map(review_embedding, processed))

        for idx, embedding in enumerate(embeddings):
            processed[idx]["embedding"] = embedding

        update_task(task_id, progress=50, message="Computing sentiment")
        with ThreadPoolExecutor(max_workers=8) as pool:
            sentiments: List[Dict[str, Any]] = list(pool.map(review_sentiment, processed))

        for idx, sentiment in enumerate(sentiments):
            processed[idx]["sentiment"] = sentiment

        update_task(task_id, progress=70, message="Building aggregates")
        df = pd.DataFrame(processed)
        result = build_aggregations(df)

        update_task(task_id, progress=80, message="Persisting results")
        executor.submit(write_csv_result, task_id, df)

        update_task(task_id, progress=95, message="Finalizing")
        update_task(task_id, status="completed", progress=100, message="Analysis complete", data=result)
    except Exception as exc:
        update_task(task_id, status="failed", progress=100, message=f"Analysis failed: {exc}")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> AnalyzeResponse:
    if len(request.reviews) == 0:
        raise HTTPException(status_code=400, detail="At least one review is required.")

    task_id = uuid.uuid4().hex
    with store_lock:
        analysis_store[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "message": "Analysis queued",
            "data": None,
        }

    background_tasks.add_task(run_full_analysis, task_id, [serialize_review(review) for review in request.reviews])
    return AnalyzeResponse(task_id=task_id, status="queued", message="Analysis task created.")


@app.get("/status/{task_id}", response_model=StatusResponse)
def status(task_id: str) -> StatusResponse:
    with store_lock:
        task = analysis_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return StatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress=int(task["progress"]),
        message=str(task["message"]),
    )


@app.get("/results/{task_id}", response_model=TaskResult)
def results(task_id: str) -> TaskResult:
    with store_lock:
        task = analysis_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskResult(
        task_id=task["task_id"],
        status=task["status"],
        progress=int(task["progress"]),
        message=str(task["message"]),
        data=task.get("data"),
    )


@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "KnowMySocials backend is running."}
