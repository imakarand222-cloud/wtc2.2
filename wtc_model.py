
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

WTC = "WTC.AX"
MARKET_TICKERS = {
    "asx200": "^AXJO",
    "audusd": "AUDUSD=X",
    "nasdaq": "^IXIC",
    "sp500": "^GSPC",
    "xro": "XRO.AX",
    "tne": "TNE.AX",
}
HORIZONS = [1, 5, 20]
START_DATE = "2016-04-11"
RANDOM_STATE = 42

def flatten_yf(df):
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return either field/ticker or ticker/field ordering
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


def _download_with_retry(ticker, start=START_DATE, retries=3):
    """
    Download with light retry logic for transient Yahoo/yfinance failures.
    Returns an empty DataFrame on final failure instead of crashing the whole app.
    """
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20,
            )
            if df is not None and not df.empty:
                return flatten_yf(df)
        except Exception as e:
            last_err = e
    return pd.DataFrame()

def download_ohlcv(ticker, start=START_DATE):
    df = _download_with_retry(ticker, start)
    if df.empty:
        raise RuntimeError(
            f"No market data returned for {ticker}. "
            "Yahoo may be rate-limiting this Streamlit deployment. Try again shortly."
        )
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if "Close" not in cols:
        raise RuntimeError(f"Downloaded data for {ticker} did not contain Close prices.")
    # Volume can occasionally be missing for non-equity instruments; WTC itself should have it.
    for c in ["Open","High","Low","Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"]).copy()

def download_close(ticker, start=START_DATE):
    df = _download_with_retry(ticker, start)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    return df["Close"].rename(ticker)

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_technical_features(df):
    x = df.copy()
    for n in [1,2,3,5,10,20,60]:
        x[f"wtc_ret_{n}d"] = x["Close"].pct_change(n)

    for n in [5,10,20,50,100,200]:
        sma = x["Close"].rolling(n).mean()
        x[f"wtc_sma_ratio_{n}"] = x["Close"] / sma - 1

    for n in [5,10,20,60]:
        x[f"wtc_vol_{n}d"] = x["Close"].pct_change().rolling(n).std()

    x["wtc_range_pct"] = (x["High"] - x["Low"]) / x["Close"]
    x["wtc_open_close_pct"] = (x["Close"] - x["Open"]) / x["Open"]
    x["wtc_gap_pct"] = x["Open"] / x["Close"].shift(1) - 1
    x["wtc_volume_chg"] = x["Volume"].pct_change()
    x["wtc_volume_ratio_20"] = x["Volume"] / x["Volume"].rolling(20).mean()
    x["wtc_rsi_14"] = rsi(x["Close"], 14)

    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["wtc_macd"] = ema12 - ema26
    x["wtc_macd_signal"] = x["wtc_macd"].ewm(span=9, adjust=False).mean()
    x["wtc_macd_hist"] = x["wtc_macd"] - x["wtc_macd_signal"]

    rolling_high = x["High"].rolling(252).max()
    rolling_low = x["Low"].rolling(252).min()
    x["wtc_dist_52w_high"] = x["Close"] / rolling_high - 1
    x["wtc_dist_52w_low"] = x["Close"] / rolling_low - 1

    x["day_of_week"] = x.index.dayofweek
    x["month"] = x.index.month
    return x


def add_market_features(base, start=START_DATE):
    """
    Adds market-context features. If an auxiliary ticker fails, the app continues
    with that source omitted rather than terminating the entire pipeline.
    """
    x = base.copy()
    market_series = {}
    for name, ticker in MARKET_TICKERS.items():
        s = download_close(ticker, start)
        if s is not None and len(s) > 0:
            market_series[name] = s

    if market_series:
        market = pd.concat(market_series, axis=1).sort_index()
        market = market.reindex(x.index).ffill()
    else:
        market = pd.DataFrame(index=x.index)

    # Create every expected market feature so the schema is stable.
    for name in MARKET_TICKERS:
        if name in market.columns:
            s = market[name]
            x[f"{name}_ret_1d"] = s.pct_change()
            x[f"{name}_ret_5d"] = s.pct_change(5)
            x[f"{name}_ret_20d"] = s.pct_change(20)
            x[f"{name}_vol_20d"] = s.pct_change().rolling(20).std()
        else:
            x[f"{name}_ret_1d"] = np.nan
            x[f"{name}_ret_5d"] = np.nan
            x[f"{name}_ret_20d"] = np.nan
            x[f"{name}_vol_20d"] = np.nan

    if "asx200_ret_1d" in x:
        x["wtc_minus_asx_1d"] = x["wtc_ret_1d"] - x["asx200_ret_1d"]
        x["wtc_minus_asx_5d"] = x["wtc_ret_5d"] - x["asx200_ret_5d"]
        x["wtc_minus_asx_20d"] = x["wtc_ret_20d"] - x["asx200_ret_20d"]
    else:
        x["wtc_minus_asx_1d"] = np.nan
        x["wtc_minus_asx_5d"] = np.nan
        x["wtc_minus_asx_20d"] = np.nan

    peer_cols = [c for c in ["xro_ret_5d", "tne_ret_5d"] if c in x.columns]
    x["tech_peer_ret_5d"] = x[peer_cols].mean(axis=1) if peer_cols else np.nan
    x["wtc_minus_peers_5d"] = x["wtc_ret_5d"] - x["tech_peer_ret_5d"]

    risk_cols = [c for c in ["nasdaq_ret_5d", "sp500_ret_5d", "asx200_ret_5d"] if c in x.columns]
    if risk_cols:
        ranks = pd.concat([x[c].rank(pct=True) for c in risk_cols], axis=1)
        x["risk_on_score"] = ranks.mean(axis=1)
    else:
        x["risk_on_score"] = np.nan

    return x

def make_fundamentals_frame(index, csv_path=None):
    """
    Point-in-time fundamentals.

    Best practice:
      Create fundamentals.csv with one row per RELEASE DATE, not fiscal period end.
      Required columns:
        date,revenue,ebitda,ebitda_margin,npata,fcf,revenue_growth,ebitda_growth,
        guidance_revenue_mid,guidance_ebitda_mid,guidance_margin_mid,
        eps,dividend,net_debt

    Values are forward-filled ONLY from their public release date, reducing look-ahead bias.
    """
    f = pd.DataFrame(index=index)
    cols = [
        "revenue","ebitda","ebitda_margin","npata","fcf","revenue_growth",
        "ebitda_growth","guidance_revenue_mid","guidance_ebitda_mid",
        "guidance_margin_mid","eps","dividend","net_debt"
    ]
    for c in cols:
        f[f"fund_{c}"] = np.nan

    if csv_path and Path(csv_path).exists():
        raw = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").set_index("date")
        raw = raw.rename(columns={c: f"fund_{c}" for c in raw.columns if c != "date"})
        common = [c for c in raw.columns if c in f.columns]
        f.loc[:, common] = raw[common].reindex(index).ffill()
    return f

def make_sentiment_frame(index, csv_path=None):
    """
    Point-in-time announcement/news features.

    Optional CSV columns:
      datetime,title,source,sentiment,importance,event_type

    sentiment should typically be in [-1, +1].
    importance can be 1=normal, 2=material, 3=major.
    """
    s = pd.DataFrame(index=index)
    base_cols = [
        "sent_mean_1d","sent_mean_3d","sent_mean_7d","sent_weighted_7d",
        "news_count_1d","news_count_7d","major_event_count_20d",
        "days_since_news","days_since_major_event"
    ]
    for c in base_cols:
        s[c] = 0.0

    if not csv_path or not Path(csv_path).exists():
        return s

    raw = pd.read_csv(csv_path, parse_dates=["datetime"])
    raw["date"] = raw["datetime"].dt.tz_localize(None).dt.normalize()
    raw["sentiment"] = pd.to_numeric(raw.get("sentiment", 0), errors="coerce").fillna(0)
    raw["importance"] = pd.to_numeric(raw.get("importance", 1), errors="coerce").fillna(1)
    daily = raw.groupby("date").agg(
        sentiment=("sentiment","mean"),
        weighted_sent=("sentiment", lambda z: float(np.mean(z))),
        news_count=("sentiment","size"),
        major_count=("importance", lambda z: int((z >= 2).sum()))
    )
    # Reindex to trading dates; same-day information is available for next-session targets.
    d = daily.reindex(index).fillna(0)
    for w in [1,3,7]:
        s[f"sent_mean_{w}d"] = d["sentiment"].rolling(w, min_periods=1).mean()
    s["sent_weighted_7d"] = (d["sentiment"] * (1 + d["major_count"])).rolling(7, min_periods=1).mean()
    s["news_count_1d"] = d["news_count"]
    s["news_count_7d"] = d["news_count"].rolling(7, min_periods=1).sum()
    s["major_event_count_20d"] = d["major_count"].rolling(20, min_periods=1).sum()

    event_dates = pd.Series(index=index, dtype="datetime64[ns]")
    last = pd.NaT
    major_last = pd.NaT
    news_gap, major_gap = [], []
    news_days = set(d.index[d["news_count"] > 0])
    major_days = set(d.index[d["major_count"] > 0])
    for dt in index:
        if dt in news_days: last = dt
        if dt in major_days: major_last = dt
        news_gap.append((dt-last).days if pd.notna(last) else 999)
        major_gap.append((dt-major_last).days if pd.notna(major_last) else 999)
    s["days_since_news"] = news_gap
    s["days_since_major_event"] = major_gap
    return s

def add_targets(df):
    x = df.copy()
    for h in HORIZONS:
        x[f"future_return_{h}d"] = x["Close"].shift(-h) / x["Close"] - 1
        x[f"target_up_{h}d"] = (x[f"future_return_{h}d"] > 0).astype(int)
    return x

def build_dataset(start=START_DATE, fundamentals_csv=None, sentiment_csv=None):
    raw = download_ohlcv(WTC, start)
    x = add_technical_features(raw)
    x = add_market_features(x, start)

    fundamentals = make_fundamentals_frame(x.index, fundamentals_csv)
    sentiment = make_sentiment_frame(x.index, sentiment_csv)
    x = x.join(fundamentals).join(sentiment)
    x = add_targets(x)
    return x

def feature_columns(df):
    exclude = {"Open","High","Low","Close","Volume"}
    exclude |= {f"future_return_{h}d" for h in HORIZONS}
    exclude |= {f"target_up_{h}d" for h in HORIZONS}
    return [c for c in df.columns if c not in exclude]


def usable_feature_columns(df):
    """
    Keep features that contain at least some information.
    This lets the app survive when one optional market source is unavailable.
    """
    feats = feature_columns(df)
    return [c for c in feats if df[c].notna().sum() >= 30]


def sanitize_ml_frame(df, features):
    """
    Convert +/-inf and numerically extreme feature values into safe ML inputs.

    pct_change and ratio features can produce infinity when the previous value is
    zero (most commonly volume changes). Tree models ultimately cast inputs to
    float32, so extreme finite values are clipped as an additional guardrail.
    """
    x = df.copy()

    # Force feature columns numeric and eliminate infinities.
    for c in features:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x[features] = x[features].replace([np.inf, -np.inf], np.nan)

    # Clip extreme tails using training-independent absolute safety bounds first.
    # This is not a predictive transformation; it prevents float32 overflow.
    FLOAT32_SAFE = 1e20
    x[features] = x[features].clip(lower=-FLOAT32_SAFE, upper=FLOAT32_SAFE)

    return x

def clean_feature_list(df, features, min_non_null=30):
    """
    Remove columns that are unavailable or become unusable after sanitization.
    """
    x = sanitize_ml_frame(df, features)
    return [
        c for c in features
        if x[c].notna().sum() >= min_non_null
        and x[c].nunique(dropna=True) > 1
    ]

def make_classifier():
    # Random Forest handles nonlinearities and mixed feature scales well.
    return RandomForestClassifier(
        n_estimators=700,
        max_depth=8,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

def make_regressor():
    return RandomForestRegressor(
        n_estimators=700,
        max_depth=8,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

def train_horizon_models(df, features, test_fraction=0.20):
    # Global sanitization removes +/-inf before any sklearn estimator sees data.
    features = clean_feature_list(df, features)
    clean = sanitize_ml_frame(df, features)

    models, metrics, predictions = {}, {}, {}

    for h in HORIZONS:
        target_ret = f"future_return_{h}d"
        target_cls = f"target_up_{h}d"

        d = clean.replace([np.inf, -np.inf], np.nan)
        d[target_ret] = pd.to_numeric(d[target_ret], errors="coerce")
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=features + [target_ret, target_cls]).copy()

        if len(d) < 200:
            raise ValueError(
                f"Only {len(d)} clean observations remain for the {h}-day model."
            )

        split = int(len(d) * (1-test_fraction))
        tr, te = d.iloc[:split].copy(), d.iloc[split:].copy()

        # Final safety clipping. The bounds are intentionally enormous and only
        # protect sklearn's internal float32 conversion from overflow.
        tr.loc[:, features] = tr[features].clip(-1e20, 1e20)
        te.loc[:, features] = te[features].clip(-1e20, 1e20)

        clf = make_classifier()
        reg = make_regressor()
        clf.fit(tr[features], tr[target_cls])
        reg.fit(tr[features], tr[target_ret])

        p = clf.predict(te[features])
        pp = clf.predict_proba(te[features])[:,1]
        r = reg.predict(te[features])

        metrics[h] = {
            "samples_train": len(tr),
            "samples_test": len(te),
            "accuracy": accuracy_score(te[target_cls], p),
            "balanced_accuracy": balanced_accuracy_score(te[target_cls], p),
            "roc_auc": roc_auc_score(te[target_cls], pp),
            "mae_return": mean_absolute_error(te[target_ret], r),
            "rmse_return": mean_squared_error(te[target_ret], r) ** 0.5,
            "r2_return": r2_score(te[target_ret], r),
        }

        pred_df = te[["Close", target_ret, target_cls]].copy()
        pred_df["prob_up"] = pp
        pred_df["expected_return"] = r
        pred_df["predicted_up"] = p
        predictions[h] = pred_df

        clf_final = make_classifier()
        reg_final = make_regressor()
        clf_final.fit(d[features], d[target_cls])
        reg_final.fit(d[features], d[target_ret])
        models[h] = {
            "classifier": clf_final,
            "regressor": reg_final,
            "features": features
        }

    return models, metrics, predictions

def latest_forecast(df, features, models):
    # Use the exact feature set retained during training.
    trained_features = models[HORIZONS[0]].get("features", features)
    clean = sanitize_ml_frame(df, trained_features)
    latest = clean.dropna(subset=trained_features).iloc[[-1]].copy()
    latest.loc[:, trained_features] = latest[trained_features].clip(-1e20, 1e20)

    rows = []
    for h in HORIZONS:
        clf = models[h]["classifier"]
        reg = models[h]["regressor"]
        rows.append({
            "horizon_days": h,
            "date": latest.index[-1],
            "close": float(latest["Close"].iloc[0]),
            "probability_up": float(
                clf.predict_proba(latest[trained_features])[:,1][0]
            ),
            "expected_return": float(
                reg.predict(latest[trained_features])[0]
            ),
        })
    return pd.DataFrame(rows)

