import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Silver Price Pro Forecaster", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🥈 Silver Price AI Analyst & Forecaster")
st.markdown("### Institutional-Grade Analysis for Silver Futures (SI=F)")
st.write("---")

# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def get_data():
    symbol = "SI=F"
    # Get 5 years to establish long-term trends
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    return df

# --- TECHNICAL INDICATORS FUNCTION ---
def add_indicators(df):
    df = df.copy()
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean() # Short term
    df['SMA_50'] = df['Close'].rolling(window=50).mean() # Medium term
    df['SMA_200'] = df['Close'].rolling(window=200).mean() # Long term trend
    
    # RSI (Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility (Standard Deviation)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    df.dropna(inplace=True)
    return df

# --- LONG TERM PROJECTION ALGORITHM ---
def project_future_prices(df, days_ahead):
    """
    Uses a simple Linear Trend Extrapolation based on the last 60 days 
    to project medium/long term. ML is great for tomorrow, 
    but Trend Lines are safer for 1 Year forecasts.
    """
    # Use the last 60 days to determine the current trend trajectory
    recent_df = df.iloc[-60:].copy()
    recent_df['Day_Index'] = np.arange(len(recent_df))
    
    X = recent_df[['Day_Index']]
    y = recent_df['Close'].values
    
    reg = LinearRegression().fit(X, y)
    
    # Project forward
    last_index = recent_df['Day_Index'].iloc[-1]
    future_index = last_index + days_ahead
  


future_price = reg.predict([[future_index]])[0]
    
    return float(future_price) # <--- This converts the array to a simple number



# --- MAIN APP LOGIC ---
if st.button("Run AI Analysis"):
    with st.spinner('Fetching market data and calculating models...'):
        try:
            # 1. Get Data
            raw_df = get_data()
            if raw_df.empty:
                st.error("Market data unavailable.")
                st.stop()
                
            df = add_indicators(raw_df)
            current_price = df['Close'].iloc[-1].item()
            current_date = df.index[-1].date()
            
            # 2. Calculate Forecasts
            # We use Trend Projection for stability over long timeframes
            pred_1d = project_future_prices(df, 1)
            pred_1w = project_future_prices(df, 5)   # 1 Week (5 trading days)
            pred_1m = project_future_prices(df, 20)  # 1 Month (20 trading days)
            pred_1y = project_future_prices(df, 252) # 1 Year (252 trading days)
            
            # 3. Insights Generation
            rsi = df['RSI'].iloc[-1].item()
            sma_50 = df['SMA_50'].iloc[-1].item()
            sma_200 = df['SMA_200'].iloc[-1].item()
            
            sentiment = "Neutral"
            if current_price > sma_50 and current_price > sma_200:
                sentiment = "Strongly Bullish 🐂"
            elif current_price < sma_50 and current_price < sma_200:
                sentiment = "Bearish 🐻"
            elif current_price > sma_200:
                sentiment = "Long-term Bullish (Correction Phase)"

            rsi_status = "Neutral"
            if rsi > 70: rsi_status = "Overbought (Risk of Pullback)"
            elif rsi < 30: rsi_status = "Oversold (Buying Opportunity)"

            # --- DISPLAY DASHBOARD ---
            
            # Top Section: Current Stats
            st.subheader(f"Market Status: {current_date}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Latest Close", f"${current_price:.2f}")
            c2.metric("Market Sentiment", sentiment)
            c3.metric("RSI Momentum", f"{rsi:.1f} ({rsi_status})")
            
            st.write("---")
            
            # Forecast Section
            st.subheader("🔮 AI Price Projections")
            st.info("Note: These projections assume the current 60-day trend continues linearly. Markets are volatile.")
            
            f1, f2, f3, f4 = st.columns(4)
            
            with f1:
                change = ((pred_1d - current_price) / current_price) * 100
                st.metric("Tomorrow", f"${pred_1d:.2f}", f"{change:.2f}%")
            
            with f2:
                change = ((pred_1w - current_price) / current_price) * 100
                st.metric("1 Week", f"${pred_1w:.2f}", f"{change:.2f}%")
                
            with f3:
                change = ((pred_1m - current_price) / current_price) * 100
                st.metric("1 Month", f"${pred_1m:.2f}", f"{change:.2f}%")
                
            with f4:
                change = ((pred_1y - current_price) / current_price) * 100
                st.metric("1 Year", f"${pred_1y:.2f}", f"{change:.2f}%")

            st.write("---")

            # --- DETAILED ANALYSIS TEXT ---
            st.subheader("📝 Deep Dive Analysis")
            
            analysis_text = f"""
            **1. Trend Analysis:**
            Silver is currently trading at **${current_price:.2f}**. 
            The 50-day Moving Average is currently ${sma_50:.2f}. 
            {"Since the price is ABOVE the 50-day average, the medium-term trend is upward." if current_price > sma_50 else "Since the price is BELOW the 50-day average, the medium-term trend is downward."}
            
            **2. Momentum (RSI):**
            The Relative Strength Index is at **{rsi:.1f}**. 
            {
                "This indicates the asset is currently Overbought. Traders typically exercise caution here as a price correction often follows." if rsi > 70 else 
                "This indicates the asset is Oversold. This is often viewed as a value buying opportunity by institutional investors." if rsi < 30 else 
                "The RSI is in neutral territory, indicating steady momentum without extreme buying or selling pressure."
            }
            
            **3. Future Outlook:**
            Based on the linear regression of the last 60 trading days, the algorithm projects a target of **${pred_1m:.2f}** within the next month. 
            If the current trend maintains its velocity, the 1-year extrapolation suggests a price of **${pred_1y:.2f}**, though global economic factors will likely alter this trajectory.
            """
            st.markdown(analysis_text)

            # --- CHARTING ---
            st.subheader("📊 Trend Visualization")
            
            # Prepare data for plotting
            plot_df = df.iloc[-180:].copy() # Show last 6 months
            dates = plot_df.index
            prices = plot_df['Close']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, prices, label='Historical Price', color='#0066cc', linewidth=2)
            
            # Plot SMA
            ax.plot(dates, plot_df['SMA_50'], label='50-Day SMA', color='orange', linestyle='--', alpha=0.7)
            
            # Plot Forecast Line (Visual aid only)
            # Create a line from today to 1 month out
            future_date = current_date + timedelta(days=30)
            ax.plot([dates[-1], pd.Timestamp(future_date)], [current_price, pred_1m], 
                    label='Trend Projection', color='green', linestyle=':', linewidth=3)

            ax.set_title("Silver Price History + AI Trend Path")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Analysis Error: {e}")
