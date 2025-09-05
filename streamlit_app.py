# streamlit_app.py
# Streamlit app for Bitcoin LSTM prediction (robust version)

import os
import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model

# TA indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# -------------------------
# Settings
# -------------------------
LOOKBACK = 60
MODEL_PATHS = [
    "final_lstm_model.keras",
    "best_lstm_model.keras",
    "final_lstm_model.h5",
    "best_lstm_model.h5"
]
SCALER_X = "scaler_x.save"
SCALER_Y = "scaler_y.save"

# -------------------------
# Helper: load model
# -------------------------
def try_load_model(paths):
    last_exc = None
    for p in paths:
        if os.path.exists(p):
            try:
                m = load_model(p)
                return m, p
            except Exception as e:
                last_exc = e
    if last_exc is not None:
        raise RuntimeError(f"Model files found but failed to load. Last error: {last_exc}")
    raise FileNotFoundError(f"No model file found. Tried: {paths}")

# -------------------------
# Load model & scalers
# -------------------------
model = None
model_path_used = None
scaler_x = None
scaler_y = None

try:
    model, model_path_used = try_load_model(MODEL_PATHS)
    st.sidebar.success(f"Model loaded from `{model_path_used}`")
except Exception as e:
    st.sidebar.error(f"Model load issue: {e}")

try:
    scaler_x = joblib.load(SCALER_X)
    scaler_y = joblib.load(SCALER_Y)
    st.sidebar.success("Scalers loaded")
except Exception as e:
    st.sidebar.warning(f"Scaler load issue: {e}")

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="Bitcoin LSTM Prediction", layout="wide")
st.title("Bitcoin Price Prediction — LSTM (Demo)")

symbol = st.selectbox("Select coin", ["BTC-USD"], index=0)
days_to_fetch = st.number_input("Days to fetch (history, daily)", min_value=LOOKBACK+1, max_value=2000, value=400, step=1)

if st.button("Get latest prediction"):
    if model is None or scaler_x is None or scaler_y is None:
        st.error("Model or scalers not loaded. Please upload required files.")
    else:
        try:
            # -------------------------
            # 1) Fetch data
            # -------------------------
            df = yf.download(symbol, period=f"{days_to_fetch}d", interval='1d', progress=False)

            if df is None or df.empty:
                st.error("No data returned by yfinance. Try increasing period or check ticker.")
                st.stop()

            st.write("Raw DataFrame preview:", df.head())

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]

            expected_cols = ['open','high','low','close','volume']
            missing = [c for c in expected_cols if c not in df.columns]

            if missing:
                st.error(f"Downloaded data is missing expected columns: {missing}")
                st.write("Available columns:", df.columns.tolist())
                st.stop()

            df = df[expected_cols].copy()
            df.index = pd.to_datetime(df.index)

            # -------------------------
            # 2) Feature engineering
            # -------------------------
            df['returns'] = df['close'].pct_change()
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_h'] = bb.bollinger_hband()
            df['bb_l'] = bb.bollinger_lband()
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
            df['sma50'] = SMAIndicator(df['close'], window=50).sma_indicator()
            df = df.dropna().copy()

            if len(df) < LOOKBACK + 1:
                st.error(f"Not enough rows after indicator calc. Need {LOOKBACK+1}, got {len(df)}.")
                st.stop()

            # -------------------------
            # 3) Prepare features
            # -------------------------
            features = ['close','volume','returns','rsi','macd','macd_signal','bb_h','bb_l','atr','ema20','sma50']
            X = df[features].values

            X_scaled = scaler_x.transform(X)
            X_seq = np.array([X_scaled[-LOOKBACK:]])

            # -------------------------
            # 4) Predict
            # -------------------------
            y_pred_scaled = model.predict(X_seq)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

            st.metric("Predicted next-day close (USD)", f"{y_pred:,.2f}")

            # -------------------------
            # 5) Plot recent closes
            # -------------------------
            df_plot = df.reset_index().tail(120).copy()

            # detect datetime column
            date_col = None
            for c in df_plot.columns:
                if np.issubdtype(df_plot[c].dtype, np.datetime64):
                    date_col = c
                    break
            if date_col is None:
                date_col = df_plot.columns[0]  # fallback
                df_plot[date_col] = pd.to_datetime(df_plot[date_col])

            df_plot['close'] = pd.to_numeric(df_plot['close'], errors='coerce')
            df_plot = df_plot.dropna(subset=[date_col,'close']).copy()

            fig = px.line(df_plot, x=date_col, y='close', title='Recent Close Prices')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.exception(e)
