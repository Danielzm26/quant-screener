import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time

st.set_page_config(layout="wide")

stocks = ["MU","MSFT","CIEN","VST","NVDA","TSLA","PLTR","AMD","AMZN","AAPL","NFLX",
          "CRWD","NOW","NBIS","BE","ALAB","COIN","SOFI","HIMS","INTC","SNDK","HOOD",
          "CRM","TSM","ASTS","HPE","NU","DUOL","SOUN","UPST","FSLR","RGTI","DELL",
          "OXY","MSTR","ORCL","ARM","OSCR","CIFR","AAL","MRNA","WULF","RIOT","MARA",
          "SMCI","SNOW"]

# =========================
# DOWNLOAD
# =========================
@st.cache_data(ttl=3600)
def download_all(stocks):
    return yf.download(stocks, period="5y", group_by="ticker", progress=False)

# =========================
# MARKET REGIME
# =========================
def market_condition():
    spy = yf.download("SPY", period="1y", progress=False)

    if spy.empty or len(spy) < 200:
        return "UNKNOWN", None

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy["SMA50"] = spy["Close"].rolling(50).mean()
    spy["SMA200"] = spy["Close"].rolling(200).mean()

    last = spy.iloc[-1]

    high_50 = spy["Close"].rolling(50).max().iloc[-1]
    low_50 = spy["Close"].rolling(50).min().iloc[-1]

    if last["SMA50"] > last["SMA200"]:
        return "BULL", None
    elif last["SMA50"] < last["SMA200"]:
        return "BEAR", None

    return "SIDEWAYS", {"high": high_50, "low": low_50}

# =========================
# FACTORS
# =========================
def compute_factors(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Return_20d"] = df["Close"].pct_change(20)
    df["Volatility"] = df["Close"].pct_change().rolling(20).std()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["AvgVol"] = df["Volume"].rolling(20).mean()

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# MULTI-TIMEFRAME FILTER
# =========================
def higher_timeframe_filter(df):
    df = df.copy()
    df["SMA200"] = df["Close"].rolling(200).mean()

    if len(df) < 200:
        return True

    return df["Close"].iloc[-1] > df["SMA200"].iloc[-1]

# =========================
# SIGNALS (ATR SL/TP)
# =========================
def generate_chart_signals(df, market=None):
    signals = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        if market == "SIDEWAYS":
            if row["Close"] <= row["Low"] * 1.01 and row["RSI"] < 45:
                signals.append({
                    "type": "BUY",
                    "x": df.index[i],
                    "y": row["Low"],
                    "sl": row["Close"] - row["ATR"],
                    "tp": row["Close"] + row["ATR"] * 2
                })
            elif row["Close"] >= row["High"] * 0.99 and row["RSI"] > 55:
                signals.append({
                    "type": "SELL",
                    "x": df.index[i],
                    "y": row["High"]
                })

        else:
            if (
                row["Close"] > row["SMA50"]
                and row["RSI"] < 40
                and row["Volume"] > row["AvgVol"]
            ):
                signals.append({
                    "type": "BUY",
                    "x": df.index[i],
                    "y": row["Low"],
                    "sl": row["Close"] - row["ATR"],
                    "tp": row["Close"] + row["ATR"] * 2
                })

            elif (
                row["Close"] < row["SMA50"]
                and row["RSI"] > 65
            ):
                signals.append({
                    "type": "SELL",
                    "x": df.index[i],
                    "y": row["High"]
                })

    return signals

# =========================
# EQUITY CURVE
# =========================
def build_equity_curve(df, signals, initial_capital=10000):
    df = df.copy()

    equity = np.zeros(len(df), dtype=float)
    equity[0] = float(initial_capital)

    cash = float(initial_capital)
    position = 0.0

    signal_map = {s["x"]: s for s in signals}

    for i in range(1, len(df)):
        price = float(df["Close"].iloc[i])
        date = df.index[i]

        if date in signal_map:
            sig = signal_map[date]

            if sig["type"] == "BUY" and position == 0:
                position = cash / price
                cash = 0.0

            elif sig["type"] == "SELL" and position > 0:
                cash = position * price
                position = 0.0

        equity[i] = cash + position * price

    df["Equity"] = equity
    return df

# =========================
# PERFORMANCE METRICS
# =========================
def compute_performance_metrics(df):
    returns = df["Equity"].pct_change().dropna()

    if len(returns) == 0:
        return None

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    cumulative = df["Equity"]
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min() * 100

    win_rate = (returns > 0).sum() / len(returns) * 100

    return {
        "Sharpe": round(sharpe, 2),
        "MaxDD": round(max_dd, 2),
        "WinRate": round(win_rate, 2)
    }

# =========================
# ANALYZE
# =========================
def analyze_stock(ticker, market, all_data):
    try:
        df = all_data[ticker].copy()
    except:
        return None

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df["Volume"].iloc[-1] < 1_000_000:
        return None

    df = compute_factors(df)
    if df.empty:
        return None

    # 🔥 FILTRO MACRO
    if not higher_timeframe_filter(df):
        return None

    row = df.iloc[-1]

    # 🔥 SCORE PRO
    score = (
        row["Return_20d"] * 50 +
        (row["Close"] / row["SMA200"]) * 30 -
        row["Volatility"] * 120 +
        (row["RSI"] < 40) * 10
    )

    if market == "BULL":
        score += 10
    elif market == "BEAR":
        score -= 10

    score = int(score) if not pd.isna(score) else 0

    prob = min(
        95,
        50 +
        (15 if row["Return_20d"] > 0 else 0) +
        (15 if row["Close"] > row["SMA200"] else 0) +
        (10 if row["RSI"] < 40 else 0) +
        (10 if row["Volume"] > row["AvgVol"] else 0)
    )

    buy = row["Close"] - row["ATR"] * 0.5
    sell = row["Close"] + row["ATR"] * 1.5

    rating = "💎 STRONG BUY" if score > 80 else "🟢 BUY" if score > 60 else "🟡 HOLD" if score > 40 else "🔴 AVOID"

    return {
        "Ticker": ticker,
        "Price": round(row["Close"], 2),
        "Buy": round(buy, 2),
        "Sell": round(sell, 2),
        "Score": score,
        "Prob": prob,
        "Rating": rating,
        "Data": df
    }

# =========================
# UI
# =========================
st.title("📊 Quant Screener PRO MAX (HEDGE FUND)")

market, _ = market_condition()
st.metric("🌎 Market Regime", market)

all_data = download_all(stocks)

results = []
data_map = {}

for stock in stocks:
    data = analyze_stock(stock, market, all_data)
    if data:
        results.append({k: data[k] for k in ["Ticker","Price","Buy","Sell","Score","Prob","Rating"]})
        data_map[stock] = data["Data"]

df = pd.DataFrame(results).sort_values(["Score","Prob"], ascending=False)

st.subheader("🏆 Ranking")
st.dataframe(df.head(35), use_container_width=True)

# =========================
# CHART
# =========================
st.subheader("📈 Advanced Chart PRO MAX")

if len(df) > 0:

    selected = st.selectbox("Selecciona ticker", df["Ticker"])
    years = st.selectbox("Selecciona años", [1,2,3,5], index=3)

    chart_df = data_map[selected].copy()
    chart_df.index = pd.to_datetime(chart_df.index)

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    chart_df = chart_df[chart_df.index >= cutoff]

    signals = generate_chart_signals(chart_df, market)
    equity_df = build_equity_curve(chart_df, signals)

    # 🔥 METRICS
    metrics = compute_performance_metrics(equity_df)
    if metrics:
        c1,c2,c3 = st.columns(3)
        c1.metric("Sharpe", metrics["Sharpe"])
        c2.metric("MaxDD", f"{metrics['MaxDD']}%")
        c3.metric("WinRate", f"{metrics['WinRate']}%")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"]
    ))

    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50"], name="SMA50"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA200"], name="SMA200"))

    for s in signals:
        fig.add_trace(go.Scatter(
            x=[s["x"]],
            y=[s["y"]],
            mode="markers+text",
            text=[s["type"]],
            marker=dict(size=10, color="green" if s["type"]=="BUY" else "red"),
            showlegend=False
        ))

        if "sl" in s:
            fig.add_hline(y=s["sl"], line_dash="dot", line_color="red")
            fig.add_hline(y=s["tp"], line_dash="dot", line_color="green")

    fig.add_trace(go.Scatter(
        x=equity_df.index,
        y=equity_df["Equity"],
        name="Equity",
        yaxis="y2"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=750,
        yaxis2=dict(overlaying="y", side="right")
    )

    st.plotly_chart(fig, use_container_width=True)
