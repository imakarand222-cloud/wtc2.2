from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from wtc_model import *

st.set_page_config(page_title="WTC Intelligence Model V2.1", page_icon="🧠", layout="wide")

st.title("WiseTech Global — WTC Intelligence Model V2.1")
st.caption("Deployment-safe hybrid ML: technicals + fundamentals + ASX/global market + announcement/news sentiment")
st.warning("Research/educational analytics only. This is not financial advice or a live trading recommendation.")

# IMPORTANT:
# The model always downloads long history for feature engineering.
# The user-facing date selector controls charts/recent tables only.
MODEL_START = "2016-04-11"
FUNDAMENTALS_PATH = "data/fundamentals.csv"
SENTIMENT_PATH = "data/news_sentiment.csv"

with st.sidebar:
    st.header("Model controls")
    display_start = st.date_input(
        "Display data from",
        value=pd.Timestamp("2024-01-01"),
        help="This only changes charts and displayed history. The ML model always uses long history."
    ).isoformat()
    test_pct = st.slider("Hold-out test set", 10, 35, 20, 5)
    bullish_threshold = st.slider("Bullish confidence", 0.50, 0.80, 0.60, 0.01)
    bearish_threshold = 1 - bullish_threshold

    st.divider()
    st.markdown("**Model history**")
    st.success(f"Automatically fixed at {MODEL_START}")
    st.caption("This prevents insufficient-observation errors from the date selector.")

    st.divider()
    st.markdown("**Optional hybrid data**")
    st.caption("Populate the CSV templates to activate company fundamentals and news sentiment.")

@st.cache_data(ttl=1800, show_spinner=False)
def get_dataset():
    return build_dataset(
        start=MODEL_START,
        fundamentals_csv=FUNDAMENTALS_PATH,
        sentiment_csv=SENTIMENT_PATH
    )

def source_status(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return False
    return df[cols].notna().any().any()

try:
    with st.spinner("Downloading WTC and market history..."):
        df = get_dataset()
except Exception as e:
    st.error("Market-data download failed.")
    st.code(str(e))
    st.info(
        "This is usually temporary Yahoo/yfinance rate limiting on Streamlit Cloud. "
        "Wait a minute and use **Rerun**. V2.1 caches successful downloads for 30 minutes."
    )
    st.stop()

features = usable_feature_columns(df)

# Need enough point-in-time rows after all selected features.
usable = df.dropna(subset=features) if features else pd.DataFrame()
if len(usable) < 260:
    st.error(f"Only {len(usable)} complete ML observations are available.")
    st.info(
        "One or more auxiliary market sources may currently be unavailable. "
        "Use Rerun after a short wait. The app now ignores fully unavailable optional sources."
    )
    st.stop()

# Model-data health panel
st.subheader("Data-source health")
health_cols = st.columns(6)

health = {
    "WTC": df["Close"].notna().sum() > 500 if "Close" in df else False,
    "ASX 200": source_status(df, "asx200_"),
    "NASDAQ": source_status(df, "nasdaq_"),
    "S&P 500": source_status(df, "sp500_"),
    "Fundamentals": df[[c for c in df.columns if c.startswith("fund_")]].notna().any().any()
        if any(c.startswith("fund_") for c in df.columns) else False,
    "News sentiment": (
        df[[c for c in df.columns if c.startswith(("sent_","news_","major_"))]].abs().sum().sum() > 0
        if any(c.startswith(("sent_","news_","major_")) for c in df.columns) else False
    )
}
for col, (name, ok) in zip(health_cols, health.items()):
    with col:
        st.metric(name, "Active" if ok else "Optional / unavailable")

with st.spinner("Training 1-day, 5-day and 20-day models..."):
    try:
        models, metrics, predictions = train_horizon_models(
            df, features, test_fraction=test_pct / 100
        )
        forecast = latest_forecast(df, features, models)
    except Exception as e:
        st.error("Model training failed.")
        st.code(str(e))
        st.stop()

latest_date = pd.Timestamp(forecast["date"].iloc[0]).date()
latest_close = float(forecast["close"].iloc[0])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest market date", str(latest_date))
c2.metric("WTC adjusted close", f"A${latest_close:,.2f}")
c3.metric("Usable ML rows", f"{len(usable):,}")
c4.metric("Hybrid features", str(len(features)))

st.subheader("Multi-horizon intelligence forecast")
cards = st.columns(3)
for col, (_, row) in zip(cards, forecast.iterrows()):
    h = int(row["horizon_days"])
    p = float(row["probability_up"])
    er = float(row["expected_return"])

    if p >= bullish_threshold:
        regime = "BULLISH"
    elif p <= bearish_threshold:
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"

    with col:
        st.markdown(f"### {h}-day")
        st.metric("Probability UP", f"{p:.1%}")
        st.metric("Expected return", f"{er:+.2%}")
        st.write(f"**Signal:** {regime}")

st.subheader("Price history")
display_df = df.loc[pd.Timestamp(display_start):, ["Close"]].dropna()
if display_df.empty:
    display_df = df[["Close"]].tail(252)
st.line_chart(display_df.rename(columns={"Close": "WTC adjusted close"}))

st.subheader("Validation dashboard")
m = pd.DataFrame(metrics).T
m.index = [f"{int(i)}d" for i in m.index]
display_m = m[
    ["accuracy","balanced_accuracy","roc_auc","mae_return","rmse_return","r2_return"]
].copy()
st.dataframe(
    display_m.style.format({
        "accuracy":"{:.1%}",
        "balanced_accuracy":"{:.1%}",
        "roc_auc":"{:.3f}",
        "mae_return":"{:.2%}",
        "rmse_return":"{:.2%}",
        "r2_return":"{:.3f}",
    }),
    use_container_width=True
)

st.subheader("Feature importance")
tabs = st.tabs(["1-day", "5-day", "20-day"])
for tab, h in zip(tabs, HORIZONS):
    with tab:
        imp = (
            pd.DataFrame({
                "Feature": features,
                "Importance": models[h]["classifier"].feature_importances_
            })
            .sort_values("Importance", ascending=False)
            .head(20)
            .set_index("Feature")
        )
        st.bar_chart(imp)

st.subheader("Recent out-of-sample predictions")
selected_h = st.selectbox(
    "Prediction horizon",
    HORIZONS,
    format_func=lambda h: f"{h} trading day(s)"
)
p = predictions[selected_h].copy()
p["actual_direction"] = np.where(
    p[f"target_up_{selected_h}d"] == 1, "UP", "DOWN"
)
p["predicted_direction"] = np.where(p["predicted_up"] == 1, "UP", "DOWN")

recent = p.loc[p.index >= pd.Timestamp(display_start)]
if recent.empty:
    recent = p.tail(40)

st.dataframe(
    recent.tail(60).sort_index(ascending=False),
    use_container_width=True
)

st.download_button(
    "Download prediction history",
    data=p.to_csv().encode("utf-8"),
    file_name=f"WTC_{selected_h}d_hybrid_predictions.csv",
    mime="text/csv"
)

with st.expander("V2.1 deployment fixes"):
    st.markdown("""
- The model history is permanently set to **2016-04-11**.
- The sidebar date changes **displayed data only**, not training history.
- Successful market downloads are cached for **30 minutes**.
- Yahoo download failures now retry and show a readable message instead of crashing.
- Auxiliary market tickers can fail without necessarily killing the entire application.
- Fully unavailable feature columns are excluded automatically.
- A data-source health panel shows which layers are active.
""")
