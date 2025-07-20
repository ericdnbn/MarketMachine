import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

@st.cache_data
def get_stock_data(ticker, period='1y'):
    try:
        df = yf.download(ticker, period=period, interval='1d')
        if df.empty:
            return None
        df.reset_index(inplace=True)
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        return None

def compute_indicators(df, sma_windows=[50, 200], rsi_window=14, bb_window=20):
    if df is None:
        return None
    # Moving Averages
    for window in sma_windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.clip(lower=0)).rolling(window=rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['Bollinger_Middle'] = df['Close'].rolling(bb_window).mean()
    df['Bollinger_Std'] = df['Close'].rolling(bb_window).std()
    df['Upper_Band'] = df['Bollinger_Middle'] + 2 * df['Bollinger_Std']
    df['Lower_Band'] = df['Bollinger_Middle'] - 2 * df['Bollinger_Std']

    # ATR
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
    df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
    df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()

    # Percent Off Recent High (52 weeks approx)
    rolling_high = df['High'].rolling(window=252).max()
    df['Percent_Off_High'] = (rolling_high - df['Close']) / rolling_high * 100

    # Percent change over last 3,5,10 days
    for n in [3, 5, 10]:
        df[f'Pct_Change_{n}D'] = df['Close'].pct_change(n) * 100

    return df

def generate_ma_crossover_signals(df, short_window=50, long_window=200):
    ma_signals = []
    for i in range(1, len(df)):
        if pd.isna(df[f'SMA_{short_window}'].iloc[i]) or pd.isna(df[f'SMA_{long_window}'].iloc[i]):
            continue
        prev_short = df[f'SMA_{short_window}'].iloc[i-1]
        prev_long = df[f'SMA_{long_window}'].iloc[i-1]
        curr_short = df[f'SMA_{short_window}'].iloc[i]
        curr_long = df[f'SMA_{long_window}'].iloc[i]
        # Cross upward
        if prev_short < prev_long and curr_short > curr_long:
            ma_signals.append(('buy', df['Date'].iloc[i]))
        # Cross downward
        elif prev_short > prev_long and curr_short < curr_long:
            ma_signals.append(('sell', df['Date'].iloc[i]))
    return ma_signals


def generate_macd_signals(df):
    macd_signals = []
    for i in range(1, len(df)):
        if pd.isna(df['MACD'].iloc[i]) or pd.isna(df['Signal_Line'].iloc[i]):
            continue
        prev_macd = df['MACD'].iloc[i - 1]
        prev_signal = df['Signal_Line'].iloc[i - 1]
        curr_macd = df['MACD'].iloc[i]
        curr_signal = df['Signal_Line'].iloc[i]
        # Buy
        if prev_macd < prev_signal and curr_macd > curr_signal:
            macd_signals.append(('buy', df['Date'].iloc[i]))
        # Sell
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            macd_signals.append(('sell', df['Date'].iloc[i]))
    return macd_signals

def generate_rsi_signals(df, rsi_lower=30, rsi_upper=70):
    rsi_signals = []
    for i in range(1, len(df)):
        if pd.isna(df['RSI_14'].iloc[i]):
            continue
        if df['RSI_14'].iloc[i-1] > rsi_upper and df['RSI_14'].iloc[i] < rsi_upper:
            # RSI crosses below overbought -> potential sell
            rsi_signals.append(('sell', df['Date'].iloc[i]))
        elif df['RSI_14'].iloc[i-1] < rsi_lower and df['RSI_14'].iloc[i] > rsi_lower:
            # RSI crosses above oversold -> potential buy
            rsi_signals.append(('buy', df['Date'].iloc[i]))
    return rsi_signals


def generate_bollinger_reversal_signals(df):
    br_signals = []
    for i in range(1, len(df)):
        lower = df['Lower_Band'].iloc[i]
        upper = df['Upper_Band'].iloc[i]
        # Check for NaN in either band before comparing
        if pd.isna(lower) or pd.isna(upper):
            continue
        price = df['Close'].iloc[i]
        date = df['Date'].iloc[i]
        # Buy signal: price touches or below lower band
        if price <= lower:
            br_signals.append(('buy', date))
        # Sell signal: price touches or above upper band
        elif price >= upper:
            br_signals.append(('sell', date))
    return br_signals



def generate_breakout_signals(df, window=20):
    break_signals = []
    max_last = df['High'].rolling(window).max()
    min_last = df['Low'].rolling(window).min()
    for i in range(window, len(df)):
        date = df['Date'].iloc[i]
        price = df['Close'].iloc[i]
        high = max_last.iloc[i]
        low = min_last.iloc[i]
        # Break above recent high
        if price > high:
            break_signals.append(('buy', date))
        # Break below recent low
        elif price < low:
            break_signals.append(('sell', date))
    return break_signals


st.title("Stock Market Dashboard with Indicators & Signals")

ticker = st.text_input("Enter stock ticker:", value='AAPL')
period = st.selectbox("Select period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)

if ticker:
    # Fetch stock data
    data = get_stock_data(ticker, period)
    if data is None:
        st.error("Failed to fetch data. Check the ticker symbol.")
    else:
        # Compute indicators and signals
        data = compute_indicators(data)
        if data is None:
            st.error("Error processing data.")
        else:
            # Generate buy and sell signals based on MACD crossover
            macd_signals = generate_macd_signals(data)
            # Generate buy and sell signals based on RSI oversold or overbought
            rsi_signals = generate_rsi_signals(data)
            # Generate buy and sell signals based on MA crossover
            ma_signals = generate_ma_crossover_signals(data)
            # Generate buy and sell signals based on Bollinger reversal 
            br_signals = generate_bollinger_reversal_signals(data)
            # Generate buy and sell signals based on price breakout
            break_signals = generate_breakout_signals(data)

            # Utility to plot signals based on metric
            def plot_signals(fig, signals, data, marker_symbol, color, label):
                """Plot signals filtered by type."""
                if not signals:
                    return
                dates = [d for s, d in signals]
                prices = data.loc[data['Date'].isin(dates), 'Close']
                if not dates:
                    return
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    mode='markers',
                    marker=dict(symbol=marker_symbol, color=color, size=12),
                    name=label
                ))


            ###############################
            # Plot Price, Moving Averages, MACD, RSI, Bollinger Bands, Price Breakouts, Buy/Sell Signals
            ###############################
 
            # --- Plot Stock Closing Price --- #
            fig_close = go.Figure()
            fig_close.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
            st.subheader("Stock Closing Price")
            st.plotly_chart(fig_close, use_container_width=True)

            # --- Plot MA crossover signals and breakouts--- #
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
            if 'SMA_50' in data:
                fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], mode='lines', name='SMA 50'))
            if 'SMA_200' in data:
                fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], mode='lines', name='SMA 200'))
            # Example for MA signals
            # Plot Buy signals
            plot_signals(fig_ma, [('buy', d) for s, d in ma_signals if s=='buy'], data, 'arrow-up', 'green', 'MA Buy')
            # Plot Sell signals
            plot_signals(fig_ma, [('sell', d) for s, d in ma_signals if s=='sell'], data, 'arrow-down', 'red', 'MA Sell')
            # Plot Breakout Buy signals
            plot_signals(fig_ma, [('buy', d) for s, d in break_signals if s=='buy'], data, 'arrow-up', 'green', 'Breakout (20) Buy')
            # Plot Breakout Sell signals
            plot_signals(fig_ma, [('sell', d) for s, d in break_signals if s=='sell'], data, 'arrow-down', 'red', 'Breakout (20) Sell')
            st.subheader("Price, MA Crossovers, Breakouts (20 Day)")
            st.plotly_chart(fig_ma, use_container_width=True)


            # --- Plot MACD --- #
            fig_macd = go.Figure()
            fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', name='Signal Line'))
            #Plots MACD Signals
            plot_signals(fig_macd, [('buy', d) for s, d in macd_signals if s=='buy'], data, 'arrow-up', 'green', 'MACD Buy')
            plot_signals(fig_macd, [('sell', d) for s, d in macd_signals if s=='sell'], data, 'arrow-down', 'red', 'MACD Sell')
            st.subheader("Moving Average Convergence Divergence MACD")
            st.plotly_chart(fig_macd, use_container_width=True)


            # --- Plot RSI --- #
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data['Date'], y=data['RSI_14'], mode='lines', name='RSI'))
            # Add overbought/oversold lines
            fig_rsi.add_shape(type='line', x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=30, y1=30, line=dict(color='gray', dash='dash'))
            fig_rsi.add_shape(type='line', x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=70, y1=70, line=dict(color='gray', dash='dash'))
            # Plot RSI signals
            plot_signals(fig_rsi, [('buy', d) for s, d in rsi_signals if s=='buy'], data, 'circle', 'green', 'RSI Buy')
            plot_signals(fig_rsi, [('sell', d) for s, d in rsi_signals if s=='sell'], data, 'circle', 'red', 'RSI Sell')
            st.subheader("Relative Strength Index (RSI)")
            st.plotly_chart(fig_rsi, use_container_width=True)


            # --- Plot Bollinger Bands with Reversal Signals --- #
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
            fig_br.add_trace(go.Scatter(x=data['Date'], y=data['Bollinger_Middle'], mode='lines', name='Middle Band'))
            fig_br.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Band'], mode='lines', name='Upper Band'))
            fig_br.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Band'], mode='lines', name='Lower Band'))
            # Plot buy and sell signals on Bollinger Bands
            plot_signals(fig_br, [('buy', d) for s, d in br_signals if s=='buy'], data, 'triangle-up', 'blue', 'BB Buy')
            plot_signals(fig_br, [('sell', d) for s, d in br_signals if s=='sell'], data, 'triangle-down', 'orange', 'BB Sell')
            st.subheader("Bollinger Bands & Reversal Signals")
            st.plotly_chart(fig_br, use_container_width=True)

            # Plot ATR
            fig_atr = go.Figure()
            fig_atr.add_trace(go.Scatter(x=data['Date'], y=data['ATR_14'], mode='lines', name='ATR'))
            st.subheader("Average True Range (ATR)")
            st.plotly_chart(fig_br, use_container_width=True)

            # Plot Percent Off High
            fig_atr = go.Figure()
            fig_atr.add_trace(go.Scatter(x=data['Date'], y=data['Percent_Off_High'], mode='lines', name='Percent Off High'))
            st.subheader("Percent Off 52 Week High")
            st.plotly_chart(fig_br, use_container_width=True)

            # --- Plot PCT Change --- #
            fig_ma = go.Figure()
            if 'Pct_Change_3D' in data:
                fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Pct_Change_3D'], mode='lines', name='Pct Change 3 Days'))
            if 'Pct_Change_5D' in data:
                fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Pct_Change_5D'], mode='lines', name='Pct Change 5 Days'))
            if 'Pct_Change_10D' in data:
                fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Pct_Change_10D'], mode='lines', name='Pct Change 10 Days'))
            st.subheader("Percent Change In Price Over 3, 5, and 10 Days")
            st.plotly_chart(fig_ma, use_container_width=True)
            

            ###############################
            # Download Data Button
            ###############################
            # After all plots and tables
            st.markdown("---")
            st.subheader("Download Data")
            st.download_button(
            label="Download Data as CSV",
            data=data.to_csv(index=False),
            file_name=f"{ticker}_data.csv",
            mime="text/csv"
            )