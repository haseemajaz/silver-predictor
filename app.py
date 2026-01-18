import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Silver Price Forecaster", layout="centered")

st.title("🥈 Silver Price AI Forecaster")
st.write("This tool uses Linear Regression to predict the next closing price of Silver Futures (SI=F).")

# --- FUNCTION TO GET DATA ---
@st.cache_data(ttl=3600)
def get_silver_data():
    symbol = "SI=F"
    # Ensure we get enough data to handle missing weekends/holidays
    df = yf.download(symbol, period="2y", interval="1d", progress=False)
    return df

# --- MAIN LOGIC ---
if st.button("Generate Prediction"):
    with st.spinner('Fetching data from Yahoo Finance...'):
        try:
            df = get_silver_data()
            
            if df.empty:
                st.error("Could not fetch data.")
                st.stop()

            # --- FEATURE ENGINEERING ---
            # Create Moving Averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            # Create Lag (Yesterday's Price)
            df['Lag_1'] = df['Close'].shift(1)
            
            # Create RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Drop NaN values
            df.dropna(inplace=True)

            # --- ML SETUP ---
            features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'Lag_1', 'RSI']
            X = df[features]
            y = df['Close']

            # Split for Graph (Last 15% of data)
            split = int(len(X) * 0.85)
            X_train_viz, X_test_viz = X.iloc[:split], X.iloc[split:]
            y_train_viz, y_test_viz = y.iloc[:split], y.iloc[split:]
            dates = df.index[split:]

            # Train Viz Model (Linear Regression)
            model_viz = LinearRegression()
            model_viz.fit(X_train_viz, y_train_viz.values.ravel())
            preds_viz = model_viz.predict(X_test_viz)
            mae = mean_absolute_error(y_test_viz, preds_viz)

            # Train Final Model (Full Data)
            final_model = LinearRegression()
            final_model.fit(X, y.values.ravel())

            # Predict Tomorrow
            last_row = df.iloc[-1][features].values.reshape(1, -1)
            next_price = final_model.predict(last_row)[0]
            current_price = df['Close'].iloc[-1].item()

            # --- DISPLAY RESULTS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Close", f"${current_price:.2f}")
            col2.metric("Next Day Forecast", f"${next_price:.2f}", delta=f"{next_price-current_price:.2f}")
            col3.metric("Model Error (MAE)", f"${mae:.2f}")

            # --- PLOT ---
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dates, y_test_viz.values, label='Actual Price', color='blue')
            ax.plot(dates, preds_viz, label='AI Trend Prediction', color='orange', linestyle='--')
            ax.set_title("Silver Price Trend Analysis")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
