import streamlit as st
import pandas as pd
from datetime import date, timedelta
import sys, pathlib, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import get_data
from src.features import make_features, train_test_split_time
from src.model import train_model, evaluate, forecast_next_days, feature_importance

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor")
st.caption("RandomForest / XGBoost / LinearRegression / ARIMA (close-only). Feature importance + CSV export.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL").upper().strip()
    default_start = date.today() - timedelta(days=365*5)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=date.today())
    model_name = st.selectbox("Model", ["RandomForest", "XGBoost", "LinearRegression", "ARIMA_close_only"])
    horizon = st.number_input("Forecast days", min_value=1, max_value=30, value=5)
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    run_btn = st.button("Run / Refresh", type="primary")

if run_btn: st.experimental_rerun()
if not ticker: st.stop()

with st.spinner("Fetching data from Yahoo Finance..."):
    raw = get_data(ticker, start_date, end_date)
if raw.empty:
    st.error("No data returned. Check ticker or date range."); st.stop()

st.subheader("Historical Prices")
st.line_chart(raw["Close"], height=240)

if model_name == "ARIMA_close_only":
    features_df = make_features(raw)
    model, used_cols, kind = train_model(model_name, features_df)
    metrics = {"MAE": float("nan"), "MAPE": float("nan")}
else:
    features_df = make_features(raw)
    if len(features_df) < 200:
        st.warning("Not enough data after feature engineering. Try a longer date range."); st.stop()
    train_df, test_df = train_test_split_time(features_df, float(test_size))
    model, used_cols, kind = train_model(model_name, train_df)
    metrics = evaluate(model, used_cols, test_df, kind)

c1, c2 = st.columns(2)
with c1: st.metric("MAE", "N/A" if np.isnan(metrics["MAE"]) else f"{metrics['MAE']:.2f}")
with c2: st.metric("MAPE", "N/A" if np.isnan(metrics["MAPE"]) else f"{metrics['MAPE']*100:.2f}%")

if model_name != "ARIMA_close_only":
    X_test = test_df[used_cols]
    y_true = test_df["Target"].rename("Actual")
    y_pred = pd.Series(model.predict(X_test), index=test_df.index, name="Predicted")
    st.subheader("Test Set: Actual vs Predicted (Next-Day Close)")
    st.line_chart(pd.concat([y_true, y_pred], axis=1), height=260)

    st.subheader("Feature Importance / Coefficients")
    imp = feature_importance(model, used_cols) or {}
    if imp:
        imp_df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())}).sort_values("importance", ascending=False)
        st.bar_chart(imp_df.set_index("feature"))
    else:
        st.caption("No importances available for this model.")

st.subheader("Forecast")
future_df = forecast_next_days(model, used_cols if model_name != "ARIMA_close_only" else [], raw, days=int(horizon), model_kind=("close_only" if model_name=="ARIMA_close_only" else "feature_based"))
st.dataframe(future_df.style.format({"PredictedClose": "{:.2f}"}), use_container_width=True)
st.download_button("Download forecast CSV", data=future_df.to_csv().encode("utf-8"), file_name=f"{ticker}_forecast.csv", mime="text/csv")

last_90 = raw[["Close"]].iloc[-90:].rename(columns={"Close": "Historical"})
plot_df = last_90.join(future_df.rename(columns={"PredictedClose": "Forecast"}), how="outer")
st.line_chart(plot_df, height=260)

st.info("Educational demo — not financial advice.")
