import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback

@st.cache_data
def get_stock_data(ticker, period):
    try:
        df = yf.download(ticker, period=period)
        print(f"Attempted fetch for {ticker} with period {period}")
        print(f"Type of fetched object: {type(df)}")
        if not isinstance(df, pd.DataFrame):
            print("Fetched data is NOT a DataFrame. Check the API response.")
            return None
        print(f"Data shape: {df.shape}")

        if df.empty:
            print("Data is empty (shape[0]==0).")
            return None

        # Reset index so Date is a column
        ###df.reset_index(inplace=True)

        # Confirm 'Close' exists
        if 'Close' not in df.columns:
            print("Column 'Close' not found in data.")
            return None

        # Convert 'Close' to numeric, coercing errors
        ###df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        # Drop NaN in 'Close'
        ###df.dropna(subset=['Close'], inplace=True)

        # Check if 'Date' column exists, then convert to datetime
        ###if 'Date' not in df.columns:
            ###print("No 'Date' column found after resetting index.")
            ###return None
        ###if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            ###df['Date'] = pd.to_datetime(df['Date'])

        print(f"Fetched {len(df)} rows after cleaning.")
        return df
    except Exception:
        print(f"Error fetching data for {ticker}")
        traceback.print_exc()  # This prints the full traceback
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
    # Check column names before dropna
    print(df.columns)
    # You might also want to drop NaNs after Bollinger Band calculations
    df.dropna(subset=[('Bollinger_Middle', ''), ('Upper_Band', ''), ('Lower_Band', '')], inplace=True)

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
        # Get the date from the DataFrame's index
        date = df.index[i]  # Corrected line

        if pd.isna(df[f'SMA_{short_window}'].iloc[i]) or pd.isna(df[f'SMA_{long_window}'].iloc[i]):
            continue
        prev_short = df[f'SMA_{short_window}'].iloc[i-1]
        prev_long = df[f'SMA_{long_window}'].iloc[i-1]
        curr_short = df[f'SMA_{short_window}'].iloc[i]
        curr_long = df[f'SMA_{long_window}'].iloc[i]
        # Cross upward
        if prev_short < prev_long and curr_short > curr_long:
            ma_signals.append(('buy', date))
        # Cross downward
        elif prev_short > prev_long and curr_short < curr_long:
            ma_signals.append(('sell', date))
    return ma_signals


def generate_macd_signals(df):
    macd_signals = []
    # Access MACD and Signal Line Series based on MultiIndex or flattened names
    if isinstance(df.columns, pd.MultiIndex):
        try:
            macd_series = df[('MACD', '')]
            signal_series = df[('Signal_Line', '')]
        except KeyError:
            # Fallback if specific MultiIndex tuple isn't found
            macd_series = df['MACD_']  # Assuming flattened name
            signal_series = df['Signal_Line_'] # Assuming flattened name
    else:
        macd_series = df['MACD']
        signal_series = df['Signal_Line']

    for i in range(1, len(df)):
        # Get the date from the DataFrame's index
        date = df.index[i]  # Corrected line

        if pd.isna(macd_series.iloc[i]) or pd.isna(signal_series.iloc[i]):
            continue
        prev_macd = macd_series.iloc[i - 1]
        prev_signal = signal_series.iloc[i - 1]
        curr_macd = macd_series.iloc[i]
        curr_signal = signal_series.iloc[i]

        # Buy
        if prev_macd < prev_signal and curr_macd > curr_signal:
            macd_signals.append(('buy', date))
        # Sell
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            macd_signals.append(('sell', date))
    return macd_signals


def generate_rsi_signals(df, rsi_lower=30, rsi_upper=70):
    rsi_signals = []
    for i in range(1, len(df)):
        # Get the date from the DataFrame's index
        date = df.index[i]  # Corrected line

        if pd.isna(df['RSI_14'].iloc[i]):
            continue
        if df['RSI_14'].iloc[i-1] > rsi_upper and df['RSI_14'].iloc[i] < rsi_upper:
            # RSI crosses below overbought -> potential sell
            rsi_signals.append(('sell', date))
        elif df['RSI_14'].iloc[i-1] < rsi_lower and df['RSI_14'].iloc[i] > rsi_lower:
            # RSI crosses above oversold -> potential buy
            rsi_signals.append(('buy', date))
    return rsi_signals


def generate_bollinger_reversal_signals(df, bb_window=20):
    br_signals = []

    # --- Pre-select Series to avoid repeated MultiIndex lookup in loop ---
    # This assumes your DataFrame's columns are structured like:
    # ('Close', 'AAPL'), ('Lower_Band', ''), ('Upper_Band', '')
    # Adjust the second element of the tuple if your MultiIndex is different
    # (e.g., if you've flattened the names, use 'Lower_Band_', 'Upper_Band_', 'Close_AAPL')
    try:
        lower_band_series = df[('Lower_Band', '')]
        upper_band_series = df[('Upper_Band', '')]
        close_series = df[('Close', 'AAPL')]
    except KeyError as e:
        print(f"Error accessing MultiIndex columns: {e}. Check column names and structure.")
        print(f"Available columns: {df.columns}")
        return br_signals # Exit if columns can't be found


    # Start from index = bb_window to avoid NaNs from rolling calculations and ensure window size is met
    for i in range(bb_window, len(df)):
        # Get the date directly from the DataFrame's index
        date = df.index[i] 

        # Access values as scalars. Use .iloc[i] followed by [0] to extract the value
        # from a single-element Series if that's what .iloc[i] returns.
        lower = lower_band_series.iloc[i][0]
        upper = upper_band_series.iloc[i][0]
        price = close_series.iloc[i][0]
        
        # Check for NaN in values to prevent comparison errors
        if pd.isna(price) or pd.isna(lower) or pd.isna(upper):
            continue

        # This isinstance check is good for debugging, but should ideally not be hit now.
        if not isinstance(price, (int, float)) or not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            print(f"ERROR: Still non-scalar after .iloc[i][0] at index {i}. Price: {price}, Lower: {lower}, Upper: {upper}")
            continue

        # Generate buy signal if price touches or dips below lower band
        if price <= lower:
            br_signals.append(('buy', date))
        # Generate sell signal if price touches or exceeds upper band
        elif price >= upper:
            br_signals.append(('sell', date))
            
    return br_signals

def generate_breakout_signals(df, window=20):
    break_signals = []

    if isinstance(df.columns, pd.MultiIndex):
        # Select the relevant columns as single-level Series for calculations
        high_series = df[('High', 'AAPL')] 
        low_series = df[('Low', 'AAPL')]
        close_series = df[('Close', 'AAPL')] # Need to extract 'Close' similarly

    else:
        high_series = df['High']
        low_series = df['Low']
        close_series = df['Close']


    max_last = high_series.rolling(window).max()
    min_last = low_series.rolling(window).min()

    for i in range(window, len(df)):
        # Get the date from the DataFrame's index
        date = df.index[i]  # Corrected line
        price = close_series.iloc[i].item() 
        high = max_last.iloc[i].item()
        low = min_last.iloc[i].item()

        # Break above recent high
        if price > high:
            break_signals.append(('buy', date))
        # Break below recent low
        elif price < low:
            break_signals.append(('sell', date))
    return break_signals


# --- Main App ---
st.title("Stock Market Dashboard with Indicators & Signals")

# Input ticker and wait for fetch button
ticker_input = st.text_input("Enter stock ticker:", value='')  # no default
period = st.selectbox("Select period", ['1y', '2y', '5y'], index=0)

if st.button("Fetch Data") and ticker_input.strip():
    # Fetch data
    data = get_stock_data(ticker_input.strip(), period=period)
    if data is None:
        st.error("Failed to fetch data. Check the ticker symbol.")
    else:
        # Compute indicators and signals
        data = compute_indicators(data)
        if data is None:
            st.error("Error processing data.")
        else:
            # Generate signals
            macd_signals = generate_macd_signals(data)
            rsi_signals = generate_rsi_signals(data)
            ma_signals = generate_ma_crossover_signals(data)
            br_signals = generate_bollinger_reversal_signals(data)
            break_signals = generate_breakout_signals(data)

            def plot_signals(fig, signals, data, marker_symbol, color, label):
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

            # --- Plot Closing Price --- #
            fig_close = go.Figure()
            fig_close.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
            st.subheader("Stock Closing Price")
            st.plotly_chart(fig_close, use_container_width=True)

            # --- Plot Price, MA Crossover, Breakouts --- #
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
            if 'SMA_50' in data:
                fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], mode='lines', name='SMA 50'))
            if 'SMA_200' in data:
                fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], mode='lines', name='SMA 200'))
            # Plot MA crossover signals
            plot_signals(fig_ma, [('buy', d) for s, d in ma_signals if s=='buy'], data, 'arrow-up', 'green', 'MA Buy')
            plot_signals(fig_ma, [('sell', d) for s, d in ma_signals if s=='sell'], data, 'arrow-down', 'red', 'MA Sell')
            # Plot breakout signals
            plot_signals(fig_ma, [('buy', d) for s, d in break_signals if s=='buy'], data, 'arrow-up', 'green', 'Breakout Buy')
            plot_signals(fig_ma, [('sell', d) for s, d in break_signals if s=='sell'], data, 'arrow-down', 'red', 'Breakout Sell')
            st.subheader("Price, MA Crossovers & Breakouts (20 Day)")
            st.plotly_chart(fig_ma, use_container_width=True)

            # --- Plot MACD --- #
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', name='Signal Line'))
            plot_signals(fig_macd, [('buy', d) for s, d in macd_signals if s=='buy'], data, 'arrow-up', 'green', 'MACD Buy')
            plot_signals(fig_macd, [('sell', d) for s, d in macd_signals if s=='sell'], data, 'arrow-down', 'red', 'MACD Sell')
            st.subheader("MACD")
            st.plotly_chart(fig_macd, use_container_width=True)

            # --- Plot RSI --- #
            st.subheader("Relative Strength Index (RSI)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data['Date'], y=data['RSI_14'], mode='lines', name='RSI'))
            # Add overbought/oversold threshold lines
            fig_rsi.add_shape(type='line', x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=30, y1=30, line=dict(color='gray', dash='dash'))
            fig_rsi.add_shape(type='line', x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1], y0=70, y1=70, line=dict(color='gray', dash='dash'))
            # Plot RSI buy signals (oversold crossing above 30)
            plot_signals(fig_rsi, [('buy', d) for s, d in rsi_signals if s=='buy'], data, 'circle', 'green', 'RSI Buy')
            # Plot RSI sell signals (overbought crossing below 70)
            plot_signals(fig_rsi, [('sell', d) for s, d in rsi_signals if s=='sell'], data, 'circle', 'red', 'RSI Sell')
            st.plotly_chart(fig_rsi, use_container_width=True)

            # --- Plot Bollinger Bands & Reversal Signals --- #
            st.subheader("Bollinger Bands & Reversal Signals")
            fig_bband = go.Figure()
            fig_bband.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
            fig_bband.add_trace(go.Scatter(x=data['Date'], y=data['Bollinger_Middle'], mode='lines', name='Middle Band'))
            fig_bband.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Band'], mode='lines', name='Upper Band'))
            fig_bband.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Band'], mode='lines', name='Lower Band'))
            # Plot buy signals at or below lower band
            plot_signals(fig_bband, [('buy', d) for s, d in br_signals if s=='buy'], data, 'triangle-up', 'blue', 'BB Buy')
            # Plot sell signals at or above upper band
            plot_signals(fig_bband, [('sell', d) for s, d in br_signals if s=='sell'], data, 'triangle-down', 'orange', 'BB Sell')
            st.subheader("Bollinger Bands & Reversal Signals")
            st.plotly_chart(fig_bband, use_container_width=True)

            # --- Plot ATR --- #
            st.subheader("Average True Range (ATR)")
            fig_atr = go.Figure()
            fig_atr.add_trace(go.Scatter(x=data['Date'], y=data['ATR_14'], mode='lines', name='ATR'))
            st.plotly_chart(fig_atr, use_container_width=True)

            # --- Plot Percent Off High --- #           
            st.subheader("Percent Off 52 Week High")
            fig_off_high = go.Figure()
            fig_off_high.add_trace(go.Scatter(x=data['Date'], y=data['Percent_Off_High'], mode='lines', name='% Off High'))
            st.plotly_chart(fig_off_high, use_container_width=True)

            # --- Plot Percent Change over 3, 5, 10 Days --- #
            st.subheader("Percent Change Over 3, 5, 10 Days")
            fig_pct_change = go.Figure()
            if 'Pct_Change_3D' in data:
                fig_pct_change.add_trace(go.Scatter(x=data['Date'], y=data['Pct_Change_3D'], mode='lines', name='Pct Change 3D'))
            if 'Pct_Change_5D' in data:
                fig_pct_change.add_trace(go.Scatter(x=data['Date'], y=data['Pct_Change_5D'], mode='lines', name='Pct Change 5D'))
            if 'Pct_Change_10D' in data:
                fig_pct_change.add_trace(go.Scatter(x=data['Date'], y=data['Pct_Change_10D'], mode='lines', name='Pct Change 10D'))
            st.plotly_chart(fig_pct_change, use_container_width=True)

            # --- Download Data --- #
            st.markdown("---")
            st.subheader("Download Data")
            st.download_button(
                label="Download Data as CSV",
                data=data.to_csv(index=False),
                file_name=f"{ticker_input}_data.csv",
                mime="text/csv"
            )
