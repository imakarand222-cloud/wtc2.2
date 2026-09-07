from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from wtc_model import *

st.set_page_config(page_title="WTC Intelligence Model V2", page_icon="🧠", layout="wide")

st.title("WiseTech Global — WTC Intelligence Model V2")
st.caption("Hybrid ML: technicals + fundamentals + ASX/global market + announcement/news sentiment")
st.warning("Research/educational analytics only. This is not financial advice and is not a live trading recommendation.")

with st.sidebar:
    st.header("Model controls")
    start = st.date_input("History starts", value=pd.Timestamp("2016-04-11")).isoformat()
    test_pct = st.slider("Hold-out test set", 10, 35, 20, 5)
    bullish_threshold = st.slider("Bullish confidence", 0.50, 0.80, 0.60, 0.01)
    bearish_threshold = 1 - bullish_threshold
    st.divider()
    st.markdown("**Optional hybrid data**")
    fundamentals_path = "data/fundamentals.csv"
    sentiment_path = "data/news_sentiment.csv"
    st.caption("Populate these CSVs to activate company fundamentals and sentiment.")

@st.cache_data(ttl=3600)
def get_dataset(start_date):
    return build_dataset(
        start=start_date,
        fundamentals_csv=fundamentals_path,
        sentiment_csv=sentiment_path
    )

with st.spinner("Building WTC hybrid dataset and market context..."):
    try:
        df = get_dataset(start)
    except Exception as e:
        st.error(f"Unable to build dataset: {e}")
        st.stop()

features = feature_columns(df)
usable = df.dropna(subset=features)
if len(usable) < 300:
    st.error("Insufficient usable observations. Select an earlier start date.")
    st.stop()

with st.spinner("Training 1-day, 5-day and 20-day models..."):
    models, metrics, predictions = train_horizon_models(df, features, test_fraction=test_pct/100)
    forecast = latest_forecast(df, features, models)

latest_date = pd.Timestamp(forecast["date"].iloc[0]).date()
latest_close = forecast["close"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest market date", str(latest_date))
c2.metric("WTC adjusted close", f"A${latest_close:,.2f}")
c3.metric("Hybrid features", str(len(features)))
c4.metric("Model horizons", "1d / 5d / 20d")

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
        st.metric(f"{h}-day probability UP", f"{p:.1%}")
        st.metric(f"{h}-day expected return", f"{er:+.2%}")
        st.write(f"**Signal:** {regime}")

st.subheader("Price history")
st.line_chart(df[["Close"]].dropna().rename(columns={"Close":"WTC adjusted close"}))

st.subheader("Validation dashboard")
m = pd.DataFrame(metrics).T
m.index = [f"{int(i)}d" for i in m.index]
display_m = m[["accuracy","balanced_accuracy","roc_auc","mae_return","rmse_return","r2_return"]].copy()
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

st.subheader("What is driving the model?")
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

st.subheader("Hybrid feature layers")
technical = [c for c in features if c.startswith("wtc_") or c in ["day_of_week","month"]]
market = [c for c in features if c.startswith(("asx200_","audusd_","nasdaq_","sp500_","xro_","tne_","tech_peer_","risk_on_"))]
fundamental = [c for c in features if c.startswith("fund_")]
sentiment = [c for c in features if c.startswith(("sent_","news_","major_","days_since_"))]

layer_df = pd.DataFrame({
    "Layer": ["WTC technical", "ASX/global market", "WiseTech fundamentals", "Announcements/news"],
    "Feature count": [len(technical), len(market), len(fundamental), len(sentiment)],
    "Status": [
        "Active",
        "Active",
        "Active with data" if df[fundamental].notna().any().any() else "Template ready — add CSV",
        "Active with data" if df[sentiment].abs().sum().sum() > 0 else "Template ready — add CSV",
    ]
})
st.dataframe(layer_df, use_container_width=True, hide_index=True)

st.subheader("Recent out-of-sample predictions")
selected_h = st.selectbox("Prediction horizon", HORIZONS, format_func=lambda h: f"{h} trading day(s)")
p = predictions[selected_h].copy()
p["actual_direction"] = np.where(p[f"target_up_{selected_h}d"] == 1, "UP", "DOWN")
p["predicted_direction"] = np.where(p["predicted_up"] == 1, "UP", "DOWN")
st.dataframe(
    p.tail(40).sort_index(ascending=False),
    use_container_width=True
)

csv = p.to_csv().encode("utf-8")
st.download_button(
    "Download prediction history",
    data=csv,
    file_name=f"WTC_{selected_h}d_hybrid_predictions.csv",
    mime="text/csv"
)

with st.expander("Model architecture and limitations"):
    st.markdown("""
**Classification:** Random Forest estimates the probability that WTC's close is higher after 1, 5 or 20 trading days.

**Regression:** A separate Random Forest estimates the forward percentage return over the same horizon.

**Technical layer:** momentum, volatility, moving averages, RSI, MACD, range, gap and volume.

**Market layer:** ASX 200, AUD/USD, Nasdaq, S&P 500, Xero and TechnologyOne, plus WTC relative strength.

**Fundamental layer:** revenue, EBITDA, margins, growth, guidance, EPS, dividends, FCF and debt when supplied as point-in-time release data.

**Sentiment layer:** announcement/news sentiment, event intensity, recency and counts when supplied.

**Critical limitation:** The optional fundamentals and sentiment CSVs must be timestamped according to when information became publicly available. Back-filling information before publication creates look-ahead leakage.
""")
