import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Silver Price Pro Forecaster", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stDataFrame { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🥈 Silver Price AI Analyst & Forecaster")
st.markdown("### Institutional-Grade Analysis for Silver Futures (SI=F)")
st.write("---")

# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def get_data():
    symbol = "SI=F"
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    return df

# --- TECHNICAL INDICATORS ---
def add_indicators(df):
    df = df.copy()
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df

# --- PROJECTION ALGORITHM ---
def project_future_prices(df, days_ahead):
    recent_df = df.iloc[-60:].copy()
    recent_df['Day_Index'] = np.arange(len(recent_df))
    
    X = recent_df[['Day_Index']]
    y = recent_df['Close'].values.ravel()
    
    reg = LinearRegression().fit(X, y)
    
    last_index = recent_df['Day_Index'].iloc[-1]
    future_index = last_index + days_ahead
    
    future_price = reg.predict([[future_index]])[0]
    return float(future_price)

# --- MAIN LOGIC ---
if st.button("Run AI Analysis"):
    with st.spinner('Fetching market data, calculating models, and generating investment table...'):
        try:
            # 1. Get Data
            raw_df = get_data()
            if raw_df.empty:
                st.error("Market data unavailable.")
                st.stop()
                
            df = add_indicators(raw_df)
            
            # Use the most recent close as current price
            current_price = float(df['Close'].iloc[-1].item())
            current_date = df.index[-1].date()
            
            # 2. Calculate Forecasts
            pred_1d = project_future_prices(df, 1)
            pred_1w = project_future_prices(df, 5)   # 1 Week
            pred_1m = project_future_prices(df, 20)  # 1 Month
            pred_1y = project_future_prices(df, 252) # 1 Year
            
            # 3. Insights
            rsi = float(df['RSI'].iloc[-1].item())
            sma_50 = float(df['SMA_50'].iloc[-1].item())
            
            sentiment = "Neutral"
            if current_price > sma_50:
                sentiment = "Bullish 🐂"
            else:
                sentiment = "Bearish 🐻"

            # --- DISPLAY DASHBOARD ---
            st.subheader(f"Market Status: {current_date}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Spot Price (per oz)", f"${current_price:.2f}")
            c2.metric("Market Sentiment", sentiment)
            c3.metric("RSI Momentum", f"{rsi:.1f}")
            
            st.write("---")

            # --- FORECAST METRICS ---
            st.subheader("🔮 AI Price Projections (Per Ounce)")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Tomorrow", f"${pred_1d:.2f}", f"{((pred_1d-current_price)/current_price)*100:.2f}%")
            f2.metric("1 Week", f"${pred_1w:.2f}", f"{((pred_1w-current_price)/current_price)*100:.2f}%")
            f3.metric("1 Month", f"${pred_1m:.2f}", f"{((pred_1m-current_price)/current_price)*100:.2f}%")
            f4.metric("1 Year", f"${pred_1y:.2f}", f"{((pred_1y-current_price)/current_price)*100:.2f}%")

            st.write("---")

            # ==========================================
            # 💰 NEW SECTION: INVESTMENT CALCULATOR
            # ==========================================
            st.subheader("💰 ROI Calculator: How much will you get?")
            st.info("Calculate the future value of your holding based on the AI predictions above.")

            # Create Tabs for different input methods
            tab1, tab2 = st.tabs(["💵 Calculate by Budget ($)", "⚖️ Calculate by Weight"])

            # --- TAB 1: BY BUDGET ---
            with tab1:
                col_input, col_type = st.columns(2)
                with col_input:
                    invest_amount = st.number_input("I want to invest ($ USD):", min_value=10.0, value=1000.0, step=50.0)
                with col_type:
                    inv_type = st.selectbox("Investment Type:", ["ETF / Paper Silver (No Premium)", "Physical Silver (Coins/Bars)"])
                
                # Logic: Physical silver usually costs MORE than spot price (Premium)
                premium_pct = 0.0
                if "Physical" in inv_type:
                    premium_pct = 0.10 # Assume 10% premium for physical
                    st.caption("⚠️ Assuming ~10% premium/markup for Physical Silver.")
                
                # Calculate how many ounces they get
                acquisition_cost = current_price * (1 + premium_pct)
                ounces_owned = invest_amount / acquisition_cost
                
                st.success(f"With **${invest_amount:,.2f}**, you can buy approximately **{ounces_owned:.2f} Troy Ounces**.")

            # --- TAB 2: BY WEIGHT ---
            with tab2:
                col_w, col_u = st.columns(2)
                with col_w:
                    weight_input = st.number_input("I currently own:", min_value=0.1, value=1.0, step=0.1)
                with col_u:
                    unit_select = st.selectbox("Unit:", ["Troy Ounces (oz)", "Kilograms (kg)", "Grams (g)"])
                
                # Convert everything to Ounces
                if unit_select == "Kilograms (kg)":
                    ounces_owned = weight_input * 32.1507
                elif unit_select == "Grams (g)":
                    ounces_owned = weight_input * 0.0321507
                else:
                    ounces_owned = weight_input
                
                current_val = ounces_owned * current_price
                st.success(f"Your **{weight_input} {unit_select}** is currently worth **${current_val:,.2f}**.")
                invest_amount = current_val # For ROI calc

            # --- GENERATE PREDICTION TABLE ---
            # Create data for table
            data = {
                "Timeframe": ["1 Day", "1 Week", "1 Month", "1 Year"],
                "Predicted Spot Price": [f"${pred_1d:.2f}", f"${pred_1w:.2f}", f"${pred_1m:.2f}", f"${pred_1y:.2f}"],
                "Your Portfolio Value": [
                    f"${(ounces_owned * pred_1d):,.2f}",
                    f"${(ounces_owned * pred_1w):,.2f}",
                    f"${(ounces_owned * pred_1m):,.2f}",
                    f"${(ounces_owned * pred_1y):,.2f}"
                ],
                "Profit / Loss": [
                    f"${(ounces_owned * pred_1d) - invest_amount:,.2f}",
                    f"${(ounces_owned * pred_1w) - invest_amount:,.2f}",
                    f"${(ounces_owned * pred_1m) - invest_amount:,.2f}",
                    f"${(ounces_owned * pred_1y) - invest_amount:,.2f}"
                ]
            }
            
            df_calc = pd.DataFrame(data)
            st.table(df_calc)

            st.write("---")

            # --- CHARTS ---
            st.subheader("📊 Trend Visualization")
            plot_df = df.iloc[-180:].copy()
            dates = plot_df.index
            prices = plot_df['Close']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, prices, label='Historical Price', color='#0066cc', linewidth=2)
            
            # Forecast Line
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
