from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import plotly.express as px  # type: ignore[import]
import plotly.graph_objects as go  # type: ignore[import]
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

CREDENTIALS: Dict[str, Tuple[str, str]] = {
    "PR": ("pr", "pr123"),
    "Marketing": ("marketing", "mkt123"),
    "Dealer": ("dealer", "dealer123"),
    "Sales": ("sales", "sales123"),
}

SAMPLE_REVIEWS = [
    {
        "review_id": "1",
        "text": "The new Honda Activa feels smooth, reliable, and comfortable for city rides.",
        "region": "North",
        "model": "Activa",
        "feature": "Comfort",
        "campaign": "Summer Launch",
        "price_tier": "Midrange",
    },
    {
        "review_id": "2",
        "text": "I love the fuel efficiency, but the service center is slow and unresponsive.",
        "region": "West",
        "model": "Dio",
        "feature": "Fuel Efficiency",
        "campaign": "Fuel Saver",
        "price_tier": "Budget",
    },
    {
        "review_id": "3",
        "text": "The design is stylish, but I found the seat uncomfortable on long journeys.",
        "region": "South",
        "model": "CB Shine",
        "feature": "Design",
        "campaign": "Style on Wheels",
        "price_tier": "Premium",
    },
    {
        "review_id": "4",
        "text": "Excellent handling and safety features, yet the delivery was delayed by a week.",
        "region": "East",
        "model": "Unicorn",
        "feature": "Safety",
        "campaign": "Ride Safe",
        "price_tier": "Premium",
    },
    {
        "review_id": "5",
        "text": "The price is reasonable and the engine is powerful for the segment.",
        "region": "North",
        "model": "CB300R",
        "feature": "Power",
        "campaign": "Value Premium",
        "price_tier": "Premium",
    },
]


def initialize_session_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "user" not in st.session_state:
        st.session_state.user = None
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = API_BASE_URL
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None


def authenticate(selected_role: str, username: str, password: str) -> bool:
    creds = CREDENTIALS.get(selected_role)
    return bool(creds and creds[0] == username and creds[1] == password)


def logout() -> None:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user = None
    st.session_state.analysis_results = None
    st.success("You have been logged out.")


def create_review_dataframe(uploaded_file: Optional[Any]) -> pd.DataFrame:
    expected_columns = ["review_id", "text", "region", "model", "feature", "campaign", "price_tier"]
    synonyms = {
        "review_id": ["review_id", "reviewid", "id", "review id", "rid"],
        "text": ["text", "review", "review_text", "comments", "comment", "feedback", "body"],
        "region": ["region", "area", "location", "market"],
        "model": ["model", "product", "vehicle", "model_name"],
        "feature": ["feature", "aspect", "category"],
        "campaign": ["campaign", "campaign_name", "promo", "promotion"],
        "price_tier": ["price_tier", "price tier", "tier", "pricing", "price_level"],
    }

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")

        normalized = {col.strip().lower(): col for col in df.columns}
        renamed: Dict[str, str] = {}
        for target, keys in synonyms.items():
            for key in keys:
                if key in normalized and target not in renamed.values():
                    renamed[normalized[key]] = target
                    break

        if renamed:
            df = df.rename(columns=renamed)

        if "text" not in df.columns:
            text_column = df.columns[0] if len(df.columns) > 0 else None
            if text_column is not None:
                df["text"] = df[text_column].astype(str)

    else:
        df = pd.DataFrame(SAMPLE_REVIEWS)

    if "review_id" not in df.columns:
        df["review_id"] = df.index.astype(str)

    for column, default in [
        ("region", "Unknown"),
        ("model", "Unknown"),
        ("feature", "General"),
        ("campaign", "General"),
        ("price_tier", "Unknown"),
    ]:
        if column not in df.columns:
            df[column] = default

    df = df.copy()
    df["review_id"] = df["review_id"].astype(str)
    df["text"] = df["text"].astype(str)

    # Keep only the expected review columns in the correct order
    reindexed: pd.DataFrame = df.reindex(columns=expected_columns, fill_value="Unknown")  # type: ignore[call-arg]
    return reindexed


PlotlyFigure = Any


def create_download_button_from_figure(fig: PlotlyFigure, label: str, filename: str) -> None:
    """Create a Streamlit download button for a Plotly figure's HTML.

    `fig` is typed as Any because Plotly stubs are not available. We keep the
    `# type: ignore` localized to attribute access where necessary.
    """
    html = fig.to_html(full_html=False, include_plotlyjs="cdn")  # type: ignore[attr-defined]
    st.download_button(label=label, data=html, file_name=filename, mime="text/html")


def call_backend_analyze(reviews: List[Dict[str, Any]]) -> Optional[str]:
    try:
        response = requests.post(
            f"{st.session_state.backend_url}/analyze",
            json={"reviews": reviews},
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("task_id")
    except requests.RequestException as exc:
        st.error(f"Backend request failed: {exc}")
        return None


def call_backend_status(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{st.session_state.backend_url}/status/{task_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Unable to poll backend status: {exc}")
        return None


def call_backend_results(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{st.session_state.backend_url}/results/{task_id}", timeout=15)
        response.raise_for_status()
        return response.json().get("data")
    except requests.RequestException as exc:
        st.error(f"Unable to fetch backend results: {exc}")
        return None


def poll_analysis(task_id: str) -> Optional[Dict[str, Any]]:
    progress_bar = st.progress(0)
    status_text = st.empty()
    while True:
        status = call_backend_status(task_id)
        if status is None:
            return None
        progress = min(max(int(status.get("progress", 0)), 0), 100)
        progress_bar.progress(progress)
        message = status.get("message", "Waiting...")
        status_text.info(f"{message} ({progress}%)")
        if status.get("status") == "completed":
            status_text.success("Analysis complete 🎉")
            return call_backend_results(task_id)
        if status.get("status") == "failed":
            status_text.error(status.get("message", "Analysis failed."))
            return None
        time.sleep(1)


def render_pr_dashboard(data: Dict[str, Any]) -> None:
    summary = data.get("summary", {})
    trend = data.get("trend", [])
    keyword_counts = data.get("keyword_counts", [])

    col1, col2 = st.columns(2)
    with col1:
        pos_count: int = int(summary.get("positive", 0))
        neg_count: int = int(summary.get("negative", 0))
        total_reviews: int = max(1, int(summary.get("total_reviews", 1)))
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=(pos_count - neg_count) / total_reviews,
                title={"text": "Overall Sentiment"},
                gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#0072B2"}},
                delta={"reference": 0.0, "increasing": {"color": "green"}},
            )
        )
        st.plotly_chart(gauge, use_container_width=True)
        create_download_button_from_figure(gauge, "Download Gauge HTML", "pr_sentiment_gauge.html")

    with col2:
        trend_fig: Any = px.line(  # type: ignore
            pd.DataFrame(trend),
            x="step",
            y="score",
            title="Sentiment Trend",
            markers=True,
        )
        trend_fig.update_layout(yaxis_title="Score")
        st.plotly_chart(trend_fig, use_container_width=True)
        create_download_button_from_figure(trend_fig, "Download Trend HTML", "pr_sentiment_trend.html")

    col3, col4 = st.columns(2)
    with col3:
        pie_values: List[int] = [pos_count, neg_count]
        pie_fig: Any = px.pie(  # type: ignore
            names=["positive", "negative"],
            values=pie_values,
            title="Review Sentiment Distribution",
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    with col4:
        top_keywords = pd.DataFrame(keyword_counts).head(10)
        if not top_keywords.empty:
            bar_fig: Any = px.bar(  # type: ignore
                top_keywords,
                x="count",
                y="keyword",
                orientation="h",
                title="Top Review Keywords",
                color_discrete_sequence=["#1f77b4"],
            )
            bar_fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("### Top Negative Reviews")
    negative_reviews = pd.DataFrame(data.get("negative_reviews", []))
    if not negative_reviews.empty:
        st.dataframe(negative_reviews)

    st.markdown("### PR Alerts")
    if summary.get("negative", 0) > summary.get("positive", 0):
        st.error("More negative reviews than positive: review PR messaging.")
    else:
        st.success("PR sentiment is healthy.")


def render_marketing_dashboard(data: Dict[str, Any]) -> None:
    feature_sentiment = pd.DataFrame(data.get("feature_sentiment", []))
    heatmap = data.get("aspect_heatmap", {})
    summary = data.get("summary", {})
    campaign_suggestions = data.get("campaign_suggestions", [])
    region_issues = pd.DataFrame(data.get("region_issues", []))
    negative_reviews = pd.DataFrame(data.get("negative_reviews", []))

    col1, col2 = st.columns(2)
    with col1:
        if not feature_sentiment.empty:
            y_cols: List[str] = [c for c in ["positive", "negative"] if c in feature_sentiment.columns]
            grouped_fig: Any = px.bar(  # type: ignore
                feature_sentiment,
                x="feature",
                y=y_cols,
                title="Feature Sentiment",
            )
            st.plotly_chart(grouped_fig, use_container_width=True)
            create_download_button_from_figure(grouped_fig, "Download Marketing Chart", "marketing_feature_sentiment.html")
    with col2:
        if heatmap and heatmap.get("x"):
            heatmap_fig: Any = go.Figure(  # type: ignore
                data=go.Heatmap(
                    x=heatmap.get("x", []),
                    y=heatmap.get("y", []),
                    z=heatmap.get("z", []),
                    colorscale="Viridis",
                )
            )
            heatmap_fig.update_layout(title="Aspect Heatmap")
            st.plotly_chart(heatmap_fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        mkt_pos: int = int(summary.get("positive", 0))
        mkt_neg: int = int(summary.get("negative", 0))
        pie_fig: Any = px.pie(  # type: ignore
            names=["positive", "negative"],
            values=[mkt_pos, mkt_neg],
            title="Sentiment Distribution",
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    with col4:
        if not region_issues.empty:
            region_fig: Any = px.bar(  # type: ignore
                region_issues,
                x="region",
                y="average_score",
                title="Region Sentiment Score",
                color="average_score",
                color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(region_fig, use_container_width=True)

    if not negative_reviews.empty:
        region_negative = (
            negative_reviews["region"].astype(str)
            .value_counts()
            .reset_index(name="negative_reviews")
        )
        region_negative_fig: Any = px.bar(  # type: ignore
            region_negative,
            x="region",
            y="negative_reviews",
            title="Negative Review Count by Region",
            color="negative_reviews",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(region_negative_fig, use_container_width=True)

    st.markdown("### Top Negative Reviews")
    if not negative_reviews.empty:
        st.dataframe(negative_reviews)
    else:
        st.info("No negative reviews available yet.")

    st.markdown("### Campaign Suggestions")
    for suggestion in campaign_suggestions[:5]:
        st.info(f"{suggestion.get('campaign', 'Campaign')}: {suggestion.get('suggestion', '')}")


def render_dealer_dashboard(data: Dict[str, Any]) -> None:
    region_issues = pd.DataFrame(data.get("region_issues", []))
    service_issues = pd.DataFrame(data.get("service_issues", []))
    negative_reviews = pd.DataFrame(data.get("negative_reviews", []))
    dealer_actions = data.get("dealer_actions", [])

    selected_region = None
    if not region_issues.empty:
        regions = region_issues["region"].astype(str).tolist()
        selected_region = st.selectbox("Select your region", regions, index=0)

    if selected_region is not None:
        selected_region_issues = region_issues[region_issues["region"] == selected_region]
        selected_negative_reviews = negative_reviews[negative_reviews["region"] == selected_region]
    else:
        selected_region_issues = region_issues
        selected_negative_reviews = negative_reviews

    col1, col2 = st.columns(2)
    with col1:
        if not selected_region_issues.empty:
            bar_fig: Any = px.bar(  # type: ignore
                selected_region_issues,
                x="region",
                y="average_score",
                title=f"Issue Score for {selected_region}",
                color="average_score",
                color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(bar_fig, use_container_width=True)
    with col2:
        trend = pd.DataFrame(data.get("trend", []))
        if not trend.empty:
            line_fig: Any = px.line(trend, x="step", y="score", title="Local Sentiment Trend", markers=True)  # type: ignore
            st.plotly_chart(line_fig, use_container_width=True)

    st.markdown("### Service Issues")
    if not service_issues.empty:
        st.bar_chart(service_issues.set_index("issue")["count"])

    st.markdown(f"### Top Negative Reviews — {selected_region}")
    if not selected_negative_reviews.empty:
        st.dataframe(selected_negative_reviews)
    else:
        st.info("No negative reviews found for this region.")

    st.markdown("### Recommended Actions")
    for action in dealer_actions:
        st.success(f"{action.get('action', '')} ({action.get('priority', '')})")


def render_sales_dashboard(data: Dict[str, Any]) -> None:
    model_performance = pd.DataFrame(data.get("model_performance", []))
    forecast = pd.DataFrame(data.get("forecast", []))
    value_scores = pd.DataFrame(data.get("value_scores", []))
    selling_points = pd.DataFrame(data.get("selling_points", []))
    negative_reviews = pd.DataFrame(data.get("negative_reviews", []))

    model_counts = pd.DataFrame()
    if not model_performance.empty:
        model_counts = model_performance.copy()
        if "positive" not in model_counts.columns:
            model_counts["positive"] = 0
        if "negative" not in model_counts.columns:
            model_counts["negative"] = 0
        model_counts["total_reviews"] = model_counts["positive"] + model_counts["negative"]

    col1, col2 = st.columns(2)
    with col1:
        if not model_performance.empty:
            perf_y: List[str] = [c for c in ["positive", "negative"] if c in model_performance.columns]
            performance_fig: Any = px.bar(  # type: ignore
                model_performance,
                x="model",
                y=perf_y,
                title="Model Sentiment Performance",
            )
            st.plotly_chart(performance_fig, use_container_width=True)
    with col2:
        if not forecast.empty:
            forecast_fig: Any = px.line(forecast, x="day", y="forecast", title="Demand Forecast", markers=True)  # type: ignore
            st.plotly_chart(forecast_fig, use_container_width=True)

    if not model_counts.empty:
        st.markdown("### Model Purchase Counts")
        purchase_fig: Any = px.bar(  # type: ignore
            model_counts,
            x="model",
            y="total_reviews",
            title="Model Reviews / Purchases by Volume",
            color="total_reviews",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(purchase_fig, use_container_width=True)
        st.dataframe(model_counts[["model", "positive", "negative", "total_reviews"]].sort_values("total_reviews", ascending=False))

    st.markdown("### Top Negative Reviews")
    if not negative_reviews.empty:
        st.dataframe(negative_reviews)

    if not value_scores.empty:
        st.markdown("### Value Signals")
        st.dataframe(value_scores)

    if not selling_points.empty:
        st.markdown("### Selling Points")
        st.dataframe(selling_points)


def render_dashboard(role: str, results: Dict[str, Any]) -> None:
    st.markdown("---")
    st.subheader(f"{role} Dashboard")
    if role == "PR":
        render_pr_dashboard(results)
    elif role == "Marketing":
        render_marketing_dashboard(results)
    elif role == "Dealer":
        render_dealer_dashboard(results)
    elif role == "Sales":
        render_sales_dashboard(results)
    else:
        st.warning("Role not recognized. Please log in again.")


def login_page() -> None:
    st.sidebar.title("KnowMySocials Login")
    role = st.sidebar.selectbox("Role", ["PR", "Marketing", "Dealer", "Sales"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        if authenticate(role, username, password):
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.user = username
            st.success(f"Welcome, {username}! You are logged in as {role}.")
        else:
            st.error("Invalid credentials. Use the demo credentials from the sidebar.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Demo credentials**")
    st.sidebar.markdown("- PR / pr123")
    st.sidebar.markdown("- Marketing / mkt123")
    st.sidebar.markdown("- Dealer / dealer123")
    st.sidebar.markdown("- Sales / sales123")


def main() -> None:
    st.set_page_config(page_title="KnowMySocials", layout="wide")
    initialize_session_state()

    st.title("KnowMySocials — Honda 2-Wheeler Customer Intelligence")
    st.markdown("A lightweight Streamlit app that connects with the KnowMySocials FastAPI backend.")

    if not st.session_state.logged_in:
        login_page()
        return

    with st.sidebar:
        st.write(f"**Logged in as:** {st.session_state.user} ({st.session_state.role})")
        st.write(f"**Backend URL:** {st.session_state.backend_url}")
        if st.button("Logout"):
            logout()

    st.markdown("### Main dashboard")
    st.markdown(
        "Upload a customer review CSV or use sample data, then click **Run Analysis** to generate your role-specific dashboard."
    )

    with st.expander("Upload customer reviews", expanded=True):
        uploaded_file = st.file_uploader("Upload a CSV with a text column", type=["csv"])
        use_sample = st.checkbox("Use sample reviews", value=False)

        if use_sample:
            review_df = create_review_dataframe(None)
            st.info("Using sample review data.")
        elif uploaded_file is not None:
            review_df = create_review_dataframe(uploaded_file)
            st.success("Uploaded CSV loaded successfully.")
        else:
            review_df = pd.DataFrame(
                columns=["review_id", "text", "region", "model", "feature", "campaign", "price_tier"]
            )
            st.warning("Upload a CSV file or select 'Use sample reviews' to load data.")

        st.dataframe(review_df.head(10))

    if st.button("Run Analysis"):
        if review_df.empty:
            st.warning("Please upload a CSV file or use sample reviews first.")
        else:
            reviews_payload = review_df.to_dict(orient="records")
            task_id = call_backend_analyze(cast(List[Dict[str, Any]], reviews_payload))
            if task_id:
                with st.spinner("Running KnowMySocials analysis..."):
                    results = poll_analysis(task_id)
                    if results is not None:
                        st.session_state.analysis_results = results
                        st.success("Analysis complete.")

    if st.session_state.analysis_results:
        render_dashboard(st.session_state.role, st.session_state.analysis_results)
        json_data = json.dumps(st.session_state.analysis_results, indent=2)
        st.download_button(
            label="Download analysis JSON",
            data=json_data,
            file_name="knowmysocials_results.json",
            mime="application/json",
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with Streamlit, FastAPI, Plotly, and pandas.")


if __name__ == "__main__":
    main()
