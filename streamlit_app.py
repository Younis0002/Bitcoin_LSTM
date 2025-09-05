# streamlit_app.py
# Streamlit app for Bitcoin LSTM prediction (robust + helpful error messages)

import os
import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# Keras model loader
from tensorflow.keras.models import load_model

# TA indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# -------------------------
# Settings / constants
# -------------------------
LOOKBACK = 60  # number of past days used by LSTM
# Try common model filenames (.keras preferred; fallback to .h5)
MODEL_PATHS = [
    "final_lstm_model.keras",
    "best_lstm_model.keras",
    "final_lstm_model.h5",
    "best_lstm_model.h5"
]
SCALER_X = "scaler_x.save"
SCALER_Y = "scaler_y.save"

# -------------------------
# Helper: attempt to load model from multiple paths
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
    # if none found or all failed
    if last_exc is not None:
        raise RuntimeError(f"Model files found but failed to load. Last error: {last_exc}")
    raise FileNotFoundError(f"No model file found. Tried: {paths}")

# -------------------------
# Load model & scalers with clear error messages (won't crash app)
# -------------------------
model = None
model_path_used = None
scaler_x = None
scaler_y = None

# Try loading model
try:
    model, model_path_used = try_load_model(MODEL_PATHS)
    st.sidebar.success(f"Model loaded from `{model_path_used}`")
except Exception as e:
    st.sidebar.error(f"Model load issue: {e}")
    model = None

# Try loading scalers
if os.path.exists(SCALER_X) and os.path.exists(SCALER_Y):
    try:
        scaler_x = joblib.load(SCALER_X)
        scaler_y = joblib.load(SCALER_Y)
        st.sidebar.success("Scalers loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load scalers: {e}")
else:
    # clear message but not fatal; notify user
    st.sidebar.warning(f"Scalers not found in repo. Expected `{SCALER_X}` and `{SCALER_Y}`")

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="Bitcoin LSTM Prediction", layout="wide")
st.title("Bitcoin Price Prediction — LSTM (Demo)")

st.markdown(
    """
    **How to use:**  
    - Make sure `final_lstm_model.keras` (or `.h5`) and `scaler_x.save`, `scaler_y.save` are in the same repo.  
    - Select coin (BTC-USD) and click **Get latest prediction**.  
    """
)

# Input controls
symbol = st.selectbox("Select coin (Yahoo Finance ticker)", ["BTC-USD"], index=0)
days_to_fetch = st.number_input("Days to fetch (history, daily)", min_value=LOOKBACK+1, max_value=2000, value=400, step=1)

# Button triggers fetch+predict+plot
if st.button("Get latest prediction"):
    # If model or scalers missing, stop early with friendly message
    if model is None:
        st.error("Model not loaded. Upload model file (`final_lstm_model.keras` or `.h5`) to the repository.")
    elif scaler_x is None or scaler_y is None:
        st.error("Scalers not loaded. Upload `scaler_x.save` and `scaler_y.save` to the repository.")
    else:
       try:
    df = yf.download(symbol, period=f"{days_to_fetch}d", interval='1d', progress=False)

    if df is None or df.empty:
        st.error("No data returned by yfinance — try a larger period or check ticker.")
        st.write("Raw DataFrame:", df)
        st.stop()

    st.write("Raw DataFrame preview:", df.head())

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Expect OHLCV
    expected_cols = ['open','high','low','close','volume']
    missing = [c for c in expected_cols if c not in df.columns]

    if missing:
        st.error(f"Downloaded data is missing expected columns: {missing}")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    df = df[expected_cols].copy()
    df.index = pd.to_datetime(df.index)

except Exception as e:
    st.error(f"Data download failed: {e}")
    st.stop()


            # -------------------------
            # 2) Feature engineering (same as training)
            # -------------------------
            # Add returns
            df['returns'] = df['close'].pct_change()

            # RSI
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

            # MACD and signal
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()

            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_h'] = bb.bollinger_hband()
            df['bb_l'] = bb.bollinger_lband()

            # ATR
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

            # EMA and SMA
            df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
            df['sma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()

            # drop NA rows produced by indicators
            df = df.dropna().copy()

            # check we have enough rows for lookback
            if len(df) < LOOKBACK + 1:
                st.error(f"Not enough data after indicator calculation. Need at least {LOOKBACK+1} rows, got {len(df)}.")
                st.stop()

            # -------------------------
            # 3) Prepare features & scale
            # -------------------------
            features = ['close','volume','returns','rsi','macd','macd_signal','bb_h','bb_l','atr','ema20','sma50']
            # ensure all features exist
            missing = [f for f in features if f not in df.columns]
            if missing:
                st.error(f"Missing features after engineering: {missing}")
                st.stop()

            X = df[features].values  # raw features
            # scaler_x expects same column order used during training
            try:
                X_scaled = scaler_x.transform(X)
            except Exception as e:
                st.error(f"Failed to transform features with scaler_x: {e}")
                st.stop()

            # build sequence for last point
            if X_scaled.shape[0] < LOOKBACK:
                st.error("Not enough scaled rows for LOOKBACK. Increase `days_to_fetch`.")
                st.stop()

            X_seq = np.array([X_scaled[-LOOKBACK:]])  # shape (1, LOOKBACK, n_features)

            # -------------------------
            # 4) Predict
            # -------------------------
            y_pred_scaled = model.predict(X_seq)
            # ensure shape is (n,1)
            y_pred_scaled = np.asarray(y_pred_scaled).reshape(-1, 1)
            try:
                y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]
            except Exception as e:
                st.error(f"Failed to inverse-transform prediction: {e}")
                st.stop()

            # display metric
            st.metric("Predicted next-day close (USD)", f"{y_pred:,.2f}")

            # -------------------------
            # 5) Plot recent actual close prices (robust)
            # -------------------------
            df_plot = df.reset_index().tail(120).copy()  # reset index so date becomes a column

            # 1) detect date column
            date_col = None
            for c in df_plot.columns:
                if np.issubdtype(df_plot[c].dtype, np.datetime64):
                    date_col = c
                    break
            if date_col is None:
                for c in df_plot.columns:
                    if any(k in c.lower() for k in ['date', 'time', 'timestamp', 'index']):
                        date_col = c
                        break
            if date_col is None:
                # try to coerce first column to datetime
                col0 = df_plot.columns[0]
                try:
                    df_plot[col0] = pd.to_datetime(df_plot[col0])
                    date_col = col0
                except Exception:
                    st.error("Could not find or convert a date/time column for plotting.")
                    st.write("Columns and dtypes:", df_plot.dtypes.to_dict())
                    st.stop()

            # ensure date_col is datetime
            try:
                df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            except Exception:
                st.error(f"Failed to convert column '{date_col}' to datetime.")
                st.write("Column sample:", df_plot[date_col].head().astype(str).tolist())
                st.stop()

            # ensure close is numeric
            if 'close' not in df_plot.columns:
                st.error("'close' column not found for plotting.")
                st.stop()
            df_plot['close'] = pd.to_numeric(df_plot['close'], errors='coerce')

            # drop NA rows
            before_len = len(df_plot)
            df_plot = df_plot.dropna(subset=[date_col, 'close']).copy()
            after_len = len(df_plot)
            if after_len < before_len:
                st.warning(f"Dropped {before_len - after_len} rows due to non-datetime date or non-numeric close values.")

            # show debug info (helpful if Streamlit redacts logs)
            st.write("Plotting DataFrame columns:", df_plot.columns.tolist())
            st.write("Plotting DataFrame dtypes:", df_plot.dtypes.astype(str).to_dict())
            st.write("Plotting DataFrame sample:", df_plot[[date_col, 'close']].head().to_dict('records'))

            # create and show Plotly chart
            fig = px.line(df_plot, x=date_col, y='close', title='Recent Close Prices')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            # catch-all: show friendly error plus stacktrace for debugging (Streamlit will redact full trace in logs)
            st.error(f"Unexpected error during prediction: {e}")
            st.exception(e)


