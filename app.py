import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Silver Price Pro Forecaster", layout="wide")

# --- CUSTOM CSS (Hide Scrollbars & Styling) ---
st.markdown("""
    <style>
    /* Hide the default Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide scrollbar for Chrome, Safari and Opera */
    ::-webkit-scrollbar {
        display: none;
    }
    /* Hide scrollbar for IE, Edge and Firefox */
    html {
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;  /* Firefox */
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🥈 Silver Price AI Analyst & Forecaster")
st.markdown("### Institutional-Grade Analysis for Silver Futures (SI=F)")
st.write("---")

# --- DATA LOADING (Auto-Update every 1 hour) ---
# ttl=3600 means it refreshes data from Yahoo every 1 hour automatically
@st.cache_data(ttl=3600)
def get_data_and_model():
    # 1. Fetch Data
    symbol = "SI=F"
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    
    if df.empty:
        return None, None, None

    # 2. Add Indicators
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)

    # 3. Train Model (Linear Regression on Trend)
    recent_df = df.iloc[-60:].copy()
    recent_df['Day_Index'] = np.arange(len(recent_df))
    
    X = recent_df[['Day_Index']]
    y = recent_df['Close'].values.ravel()
    
    model = LinearRegression().fit(X, y)
    
    # Return everything we need
    return df, model, recent_df

# --- PREDICTION FUNCTION ---
def get_prediction(model, last_index, days_ahead):
    future_index = last_index + days_ahead
    pred = model.predict([[future_index]])[0]
    return float(pred)

# ==========================================
# MAIN EXECUTION (Runs Automatically on Load)
# ==========================================

# Load Data Immediately
with st.spinner('Analyzing live market data...'):
    df, model, recent_df = get_data_and_model()

if df is None:
    st.error("Could not fetch market data. Please reload.")
    st.stop()

# Prepare Variables
current_price = float(df['Close'].iloc[-1].item())
current_date = df.index[-1].date()
last_idx = recent_df['Day_Index'].iloc[-1]

# Calculate Predictions
pred_1d = get_prediction(model, last_idx, 1)
pred_1w = get_prediction(model, last_idx, 5)
pred_1m = get_prediction(model, last_idx, 20)
pred_1y = get_prediction(model, last_idx, 252)

# Calculate Indicators
rsi = float(df['RSI'].iloc[-1].item())
sma_50 = float(df['SMA_50'].iloc[-1].item())

# Determine Sentiment
sentiment = "Neutral"
if current_price > sma_50:
    sentiment = "Bullish 🐂"
else:
    sentiment = "Bearish 🐻"

# --- DISPLAY TOP DASHBOARD ---
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
# 💰 ROI CALCULATOR (Interactive)
# ==========================================
st.subheader("💰 ROI Calculator: How much will you get?")
st.info("Calculate the future value of your holding based on the AI predictions above.")

# Create Tabs
tab1, tab2 = st.tabs(["💵 Calculate by Budget ($)", "⚖️ Calculate by Weight"])

invest_amount = 0.0
ounces_owned = 0.0

# --- TAB 1: BY BUDGET ---
with tab1:
    col_input, col_type = st.columns(2)
    with col_input:
        budget_input = st.number_input("I want to invest ($ USD):", min_value=10.0, value=1000.0, step=50.0)
    with col_type:
        inv_type = st.selectbox("Investment Type:", ["ETF / Paper Silver (No Premium)", "Physical Silver (Coins/Bars)"])
    
    premium_pct = 0.0
    if "Physical" in inv_type:
        premium_pct = 0.10 # 10% premium
        st.caption("⚠️ Assuming ~10% premium/markup for Physical Silver.")
    
    acquisition_cost = current_price * (1 + premium_pct)
    ounces_owned = budget_input / acquisition_cost
    invest_amount = budget_input
    
    st.success(f"With **${budget_input:,.2f}**, you can buy approximately **{ounces_owned:.2f} Troy Ounces**.")

# --- TAB 2: BY WEIGHT ---
with tab2:
    col_w, col_u = st.columns(2)
    with col_w:
        weight_input = st.number_input("I currently own:", min_value=0.1, value=1.0, step=0.1)
    with col_u:
        unit_select = st.selectbox("Unit:", ["Troy Ounces (oz)", "Kilograms (kg)", "Grams (g)"])
    
    # Calculate Ounces based on selection
    if unit_select == "Kilograms (kg)":
        ounces_owned_w = weight_input * 32.1507
    elif unit_select == "Grams (g)":
        ounces_owned_w = weight_input * 0.0321507
    else:
        ounces_owned_w = weight_input
    
    current_val = ounces_owned_w * current_price
    st.success(f"Your **{weight_input} {unit_select}** is currently worth **${current_val:,.2f}**.")
    
    # Logic to prioritize inputs
    if budget_input == 1000.0 and weight_input != 1.0:
        ounces_owned = ounces_owned_w
        invest_amount = current_val

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

# Override display logic if using weight
if weight_input != 1.0 and budget_input == 1000.0:
     st.caption("Showing forecast for **Weight Input**.")
else:
    st.caption("Showing forecast for **Budget Input** (Default). To use Weight, change the Weight value and leave Budget at 1000.")

df_calc = pd.DataFrame(data)
st.table(df_calc)

# --- CHARTS ---
st.write("---")
st.subheader("📊 Trend Visualization")
plot_df = df.iloc[-180:].copy()
dates = plot_df.index
prices = plot_df['Close']

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, prices, label='Historical Price', color='#0066cc', linewidth=2)

future_date = current_date + timedelta(days=30)
ax.plot([dates[-1], pd.Timestamp(future_date)], [current_price, pred_1m], 
        label='Trend Projection', color='green', linestyle=':', linewidth=3)

ax.set_title("Silver Price History + AI Trend Path")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
