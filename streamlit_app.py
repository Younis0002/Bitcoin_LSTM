# Save below as streamlit_app.py
import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Settings
LOOKBACK = 60
MODEL_PATH = "final_lstm_model.h5"
SCALER_X = "scaler_x.save"
SCALER_Y = "scaler_y.save"

# Load artifacts
model = load_model(MODEL_PATH)
scaler_x = joblib.load(SCALER_X)
scaler_y = joblib.load(SCALER_Y)

st.title("Bitcoin Price Prediction - LSTM (Demo)")

symbol = st.selectbox("Select coin", ["BTC-USD"], index=0)
if st.button("Get latest prediction"):
    # fetch latest 200 daily candles to be safe
    df = yf.download(symbol, period='400d', interval='1d', progress=False)
    df = df[['Open','High','Low','Close','Volume']]
    df.columns = ['open','high','low','close','volume']
    df.index = pd.to_datetime(df.index)

    # compute indicators (same as training)
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

    features = ['close','volume','returns','rsi','macd','macd_signal','bb_h','bb_l','atr','ema20','sma50']
    X = scaler_x.transform(df[features].values)
    # build sequence for latest point
    X_seq = np.array([X[-LOOKBACK:]])
    y_pred_scaled = model.predict(X_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

    st.metric("Predicted next-day close (USD)", f"{y_pred:,.2f}")

    # show last N actual vs predicted (we can show last close)
    # Show last 120 days close price chart
df_plot = df.reset_index().tail(120)   # reset index so 'date' becomes a column
fig = px.line(df_plot, x='date', y='close', title='Recent Close Prices')
st.plotly_chart(fig, use_container_width=True)

# To run: `streamlit run streamlit_app.py`


