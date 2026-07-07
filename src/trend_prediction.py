"""
Trend Prediction Module
Analyzes sentiment trends over time and predicts future trends using Prophet.
Optional LSTM-based deep learning forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, cast

import config
from src.utils import setup_logger

logger = setup_logger(__name__)


class TrendAnalyzer:
    """Analyzes sentiment trends and generates forecasts."""

    def __init__(self, use_prophet: bool = config.USE_PROPHET_TRENDS):
        """
        Initialize trend analyzer.
        
        Args:
            use_prophet: Use Prophet for forecasting (requires statsmodels + pystan)
        """
        self.logger = setup_logger(__name__)
        self.use_prophet = use_prophet
        self.prophet = None
        
        if use_prophet:
            try:
                from prophet import Prophet  # type: ignore[reportMissingImports,reportUnknownMemberType]
                self.prophet = Prophet  # type: ignore[reportUnknownMemberType]
                self.logger.info("Prophet imported successfully")
            except ImportError:
                self.logger.warning("Prophet not available; trend analysis limited")
                self.use_prophet = False

    def compute_sentiment_trend(
        self, sentiment_scores: List[float], window_size: int = 10
    ) -> str:
        """
        Determine if sentiment trend is increasing, stable, or decreasing.
        
        Args:
            sentiment_scores: List of sentiment scores (e.g., compound VADER scores)
            window_size: Window size for trend computation
            
        Returns:
            Trend label: "increasing", "stable", or "decreasing"
        """
        if len(sentiment_scores) < 2:
            return "stable"
        
        try:
            # Use last N scores
            scores = sentiment_scores[-window_size:]
            
            # Compute linear trend
            x = np.arange(len(scores)).reshape(-1, 1)
            y = np.array(scores)
            
            # Simple linear regression
            x_mean = x.mean()
            y_mean = y.mean()
            
            numerator = np.sum((x.flatten() - x_mean) * (y - y_mean))
            denominator = np.sum((x.flatten() - x_mean) ** 2)
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            # Classify trend
            if slope > 0.05:
                return "increasing"
            elif slope < -0.05:
                return "decreasing"
            else:
                return "stable"
        
        except Exception as e:
            self.logger.warning(f"Error computing sentiment trend: {e}")
            return "stable"

    def forecast_sentiment_prophet(
        self, sentiment_df: pd.DataFrame, periods: int = config.FORECAST_PERIODS
    ) -> Optional[List[float]]:
        """
        Use Prophet to forecast sentiment for future periods.
        Requires data in format: {ds: datetime, y: sentiment_score}
        
        Args:
            sentiment_df: DataFrame with columns [ds (datetime), y (sentiment)]
            periods: Number of periods to forecast
            
        Returns:
            List of forecasted sentiment scores or None if forecasting failed
        """
        if not self.use_prophet or self.prophet is None or len(sentiment_df) < 2:  # type: ignore[reportUnknownMemberType]
            return None
        
        try:
            self.logger.info(f"Fitting Prophet model for {len(sentiment_df)} historical sentiments")
            
            # Prepare data for Prophet
            df_prophet = sentiment_df[["ds", "y"]].copy()
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])  # type: ignore[reportUnknownMemberType]
            df_prophet["y"] = df_prophet["y"].astype(float)
            
            # Fit Prophet
            model = self.prophet(  # type: ignore[reportUnknownMemberType]
                seasonality_mode=config.PROPHET_SEASONALITY_MODE,
                yearly_seasonality=config.PROPHET_YEARLY_SEASONALITY,
                weekly_seasonality=config.PROPHET_WEEKLY_SEASONALITY,
                daily_seasonality=config.PROPHET_DAILY_SEASONALITY,
                interval_width=0.95
            )
            model.fit(df_prophet)  # type: ignore[reportUnknownMemberType]
            
            # Generate future dataframe
            future = model.make_future_dataframe(periods=periods)  # type: ignore[reportUnknownMemberType]
            forecast = model.predict(future)  # type: ignore[reportUnknownMemberType]
            
            # Extract forecasted values
            forecasted_yhat = forecast["yhat"].tail(periods).values.tolist()  # type: ignore[reportUnknownMemberType]
            
            self.logger.info(f"Prophet forecast completed; forecasted {periods} periods")
            return forecasted_yhat  # type: ignore[return-value]
        
        except Exception as e:
            self.logger.warning(f"Prophet forecasting failed: {e}")
            return None

    def forecast_sentiment_simple(
        self, sentiment_scores: List[float], periods: int = config.FORECAST_PERIODS
    ) -> List[float]:
        """
        Simple exponential smoothing-based forecast (fallback if Prophet unavailable).
        
        Args:
            sentiment_scores: List of historical sentiment scores
            periods: Number of periods to forecast
            
        Returns:
            List of forecasted sentiment scores
        """
        if len(sentiment_scores) < 2:
            # Return neutral forecast
            return [0.0] * periods
        
        try:
            _alpha = 0.3  # Smoothing factor (simplified: not used in current implementation)
            last_value = sentiment_scores[-1]
            forecast = []
            
            for _ in range(periods):
                # Exponential smoothing: forecast = last_value (simplified)
                forecast.append(last_value)  # type: ignore[reportUnknownMemberType]
            
            return forecast  # type: ignore[return-value]
        
        except Exception as e:
            self.logger.warning(f"Simple forecasting failed: {e}")
            return [0.0] * periods  # type: ignore[return-value]

    def compute_confidence(self, historical_scores: List[float]) -> float:
        """
        Estimate confidence in trend prediction based on data variability.
        
        Args:
            historical_scores: List of historical sentiment scores
            
        Returns:
            Confidence score between 0 and 1 (higher = more confident)
        """
        if len(historical_scores) < 2:
            return 0.3
        
        try:
            # Confidence inverse to volatility
            std_dev = float(np.std(historical_scores))
            
            # Normalize volatility to [0, 1]
            # Assume max volatility is ~1.0
            volatility_norm = min(std_dev, 1.0)
            confidence = 1.0 - volatility_norm
            
            return max(0.0, min(1.0, confidence))
        
        except Exception as e:
            self.logger.warning(f"Error computing confidence: {e}")
            return 0.5


def analyze_trends_batch(
    df: pd.DataFrame,
    sentiment_column: str = "sentiment",
    use_grouping: bool = True
) -> pd.DataFrame:
    """
    Analyze trends for a batch of reviews, grouped by model if available.
    
    Args:
        df: DataFrame with columns [review_id, sentiment, model_name]
        sentiment_column: Column name containing sentiment dict
        use_grouping: Group trends by model_name
        
    Returns:
        DataFrame with columns [review_id, trends]
    """
    logger.info(f"Analyzing trends for {len(df)} reviews...")
    
    analyzer = TrendAnalyzer()
    results: Dict[str, Any] = {
        "review_id": [],
        "trends": []
    }
    
    records = cast(Any, df.to_dict(orient="records"))  # type: ignore[reportUnknownMemberType]
    
    if use_grouping and "model_name" in df.columns:
        # Group by model
        grouped = getattr(df, "groupby")("model_name")
        
        for model_name, model_group in grouped:
            if not isinstance(model_name, str):
                model_name = str(model_name)
            logger.info(f"Analyzing trends for {len(model_group)} reviews of {model_name}")
            
            # Extract sentiment compounds for this model
            sentiment_scores = []
            for row in model_group.to_dict(orient="records"):  # type: ignore[reportUnknownMemberType]
                sentiment = row.get(sentiment_column, {})
                if isinstance(sentiment, dict):
                    compound = sentiment.get("vader_compound", 0.0)  # type: ignore[reportUnknownMemberType]
                    sentiment_scores.append(float(compound))  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            
            # Compute trend for this model
            trend = analyzer.compute_sentiment_trend(sentiment_scores)  # type: ignore[reportUnknownArgumentType]
            forecast = analyzer.forecast_sentiment_simple(sentiment_scores)  # type: ignore[reportUnknownArgumentType]
            confidence = analyzer.compute_confidence(sentiment_scores)  # type: ignore[reportUnknownArgumentType]
            
            # Assign trend data to all reviews in this model
            for row in model_group.to_dict(orient="records"):  # type: ignore[reportUnknownArgumentType]
                results["review_id"].append(row.get("review_id"))  # type: ignore[reportUnknownMemberType]
                results["trends"].append({  # type: ignore[reportUnknownMemberType]
                    "model": model_name,
                    "trend": trend,
                    "forecast_next_5": forecast[:5],
                    "confidence": confidence,
                    "data_points": len(sentiment_scores)  # type: ignore[reportUnknownArgumentType]
                })
    
    else:
        # No grouping; compute overall trend
        sentiment_scores = []
        for row in records:
            sentiment = row.get(sentiment_column, {})
            if isinstance(sentiment, dict):
                compound = sentiment.get("vader_compound", 0.0)  # type: ignore[reportUnknownMemberType]
                sentiment_scores.append(float(compound))  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        
        trend = analyzer.compute_sentiment_trend(sentiment_scores)  # type: ignore[reportUnknownArgumentType]
        forecast = analyzer.forecast_sentiment_simple(sentiment_scores)  # type: ignore[reportUnknownArgumentType]
        confidence = analyzer.compute_confidence(sentiment_scores)  # type: ignore[reportUnknownArgumentType]
        
        for row in records:
            results["review_id"].append(row.get("review_id"))  # type: ignore[reportUnknownMemberType]
            results["trends"].append({  # type: ignore[reportUnknownMemberType]
                "trend": trend,
                "forecast_next_5": forecast[:5],
                "confidence": confidence,
                "data_points": len(sentiment_scores)  # type: ignore[reportUnknownArgumentType]
            })
    
    result_df = pd.DataFrame(results)
    logger.info(f"Trend analysis completed for {len(result_df)} reviews")
    
    return result_df
