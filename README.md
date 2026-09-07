# WiseTech Global — WTC Intelligence Model V2

Hybrid multi-horizon machine-learning project for WiseTech Global (`WTC.AX`).

## Outputs
For 1, 5 and 20 trading days:
- probability WTC closes higher
- expected forward return
- historical classification accuracy / balanced accuracy / ROC-AUC
- expected-return MAE / RMSE / R²
- feature importance
- recent out-of-sample predictions

## Four feature layers
1. WTC technical/momentum/volatility/volume indicators
2. ASX/global market context — ASX 200, AUD/USD, Nasdaq, S&P 500, Xero, TechnologyOne
3. WiseTech fundamentals — revenue, EBITDA, margin, growth, guidance, EPS, dividends, FCF, debt
4. Announcements/news — sentiment, volume, materiality and event recency

## Files
- `WTC_Intelligence_Model_V2.ipynb` — full research notebook
- `streamlit_app_v2.py` — interactive dashboard
- `wtc_model.py` — reusable data/feature/model library
- `data/fundamentals.csv` — point-in-time fundamental template
- `data/news_sentiment.csv` — announcement/news template
- `requirements.txt`

## Run
```bash
pip install -r requirements.txt
jupyter notebook WTC_Intelligence_Model_V2.ipynb
```

Dashboard:
```bash
streamlit run streamlit_app_v2.py
```

## Point-in-time data rule
Fundamental rows must use the date the result/guidance was PUBLICLY RELEASED, not the fiscal period end.
News/announcement rows must use their actual publication timestamp.
This is essential to avoid look-ahead leakage.

## Sentiment scoring convention
Recommended:
- -1.0 = strongly negative
- -0.5 = moderately negative
- 0.0 = neutral
- +0.5 = moderately positive
- +1.0 = strongly positive

Importance:
- 1 = routine
- 2 = material
- 3 = major / market-moving

## Research warning
This project is an analytical research framework, not financial advice. Production use would require source-quality controls, survivorship/leakage tests, transaction-cost analysis, probability calibration, regime analysis and continuous monitoring.


## V2.1 Streamlit Cloud deployment fix

Use:

```bash
streamlit run streamlit_app_v2_1.py
```

V2.1 separates **model history** from **display history**. The model always begins at
2016-04-11 so a user cannot accidentally create too few observations by changing the
date widget. It also adds retry/caching behavior and graceful handling of unavailable
auxiliary market feeds.


## V2.2 numerical-safety fix

V2.2 fixes the Streamlit error:

`Input X contains infinity or a value too large for dtype('float32')`

Percentage-change and ratio features can become infinite when their denominator is
zero (especially volume changes). Before training and prediction, V2.2 now:

1. coerces every feature to numeric;
2. converts positive/negative infinity to missing values;
3. drops unusable feature columns;
4. clips only astronomically extreme values to float32-safe bounds;
5. drops rows that remain incomplete for the selected feature set; and
6. applies the identical sanitation pipeline to the latest forecast.
