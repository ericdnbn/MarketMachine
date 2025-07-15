import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Function to fetch data with error handling
@st.cache
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='1y', interval='1d')
        if df.empty:
            raise ValueError("Empty data fetched.")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        return None

# Function to compute indicators
def compute_indicators(df):
    if df is None:
        return None

    # Moving Averages for 50 and 200 days
    for window in [50, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
    df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()

    # Percent Off Recent High (52-week)
    rolling_high = df['High'].rolling(window=252).max()
    df['Percent_Off_High'] = (rolling_high - df['Close']) / rolling_high * 100

    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.clip(lower=0)).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands with fixed window 20
    window = 20
    df['Bollinger_Middle'] = df['Close'].rolling(window).mean()
    df['Bollinger_Std'] = df['Close'].rolling(window).std()
    df['Upper_Band'] = df['Bollinger_Middle'] + 2 * df['Bollinger_Std']
    df['Lower_Band'] = df['Bollinger_Middle'] - 2 * df['Bollinger_Std']

    # Percent change over last 3, 5, 10 days
    for n in [3, 5, 10]:
        df[f'Pct_Change_{n}D'] = df['Close'].pct_change(n) * 100

    return df

# Streamlit UI
st.title("Stock Market Dashboard")
ticker = st.text_input("Enter stock ticker:", value='AAPL')

if ticker:
    data = get_stock_data(ticker)
    if data is None:
        st.error("Failed to fetch data. Please check the ticker symbol.")
    else:
        data = compute_indicators(data)
        if data is None:
            st.error("Error processing data.")
        else:
        data = compute_indicators(data)
    if data is None:
        st.error("Error processing data.")
    else:
        # Plot price with moving averages (50 & 200)
        st.subheader(f"{ticker} Price & Moving Averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
        for window in [50, 200]:
            if f'SMA_{window}' in data:
                fig.add_trace(go.Scatter(x=data['Date'], y=data[f'SMA_{window}'], mode='lines', name=f'SMA {window}'))
        st.plotly_chart(fig, use_container_width=True)
        
        # MACD Plot
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD'))
        fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], name='Signal Line'))
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # ATR
        st.subheader("Average True Range (ATR)")
        st.line_chart(data.set_index('Date')['ATR_14'])
        
        # Percent Off Recent High
        st.subheader("Percent Off Recent High")
        st.line_chart(data.set_index('Date')['Percent_Off_High'])
        
        # RSI
        st.subheader("RSI (14)")
        st.line_chart(data.set_index('Date')['RSI_14'])
        
        # Bollinger Bands
        st.subheader("Bollinger Bands (20 window)")
        fig_bollinger = go.Figure()
        fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
        fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Bollinger_Middle'], mode='lines', name='Bollinger Middle'))
        fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Band'], mode='lines', name='Upper Band'))
        fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Band'], mode='lines', name='Lower Band'))
        st.plotly_chart(fig_bollinger, use_container_width=True)
        
        # Percent Change over 3,5,10 days
        st.subheader("Percent Change Over N Days")
        cols = ['Pct_Change_3D', 'Pct_Change_5D', 'Pct_Change_10D']
        for col in cols:
            if col in data:
                st.line_chart(data.set_index('Date')[col])

