
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Backtest Demo", layout="wide")
st.title("📈 Market Backtesting Demo")
st.write("This demo runs immediately with the included sample data. Replace the sample CSVs with real historical data when ready.")

DATA = {
    "NIFTY 50": "sample_nifty.csv",
    "BANKNIFTY": "sample_banknifty.csv",
    "TMPV": "sample_tmpv.csv",
}

def load(path):
    df=pd.read_csv(path)
    df["Date"]=pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

instrument=st.sidebar.selectbox("Instrument",list(DATA))
uploaded=st.sidebar.file_uploader("Optional: upload your own OHLC CSV",type="csv")

df=pd.read_csv(uploaded) if uploaded else load(DATA[instrument])
df.columns=[str(c).strip().title() for c in df.columns]
df["Date"]=pd.to_datetime(df["Date"])
df=df.sort_values("Date").reset_index(drop=True)

st.sidebar.header("Strategy")
strategy=st.sidebar.selectbox("Strategy",["EMA Crossover","SMA Crossover","Buy & Hold"])
fast=st.sidebar.number_input("Fast period",2,200,20)
slow=st.sidebar.number_input("Slow period",3,500,50)
capital=st.sidebar.number_input("Initial capital ₹",10000,10000000,500000,50000)
cost=st.sidebar.number_input("Cost + slippage %",0.0,2.0,0.05,0.01)

if strategy=="Buy & Hold":
    df["position"]=1
else:
    if strategy=="EMA Crossover":
        df["fast"]=df.Close.ewm(span=fast,adjust=False).mean()
        df["slow"]=df.Close.ewm(span=slow,adjust=False).mean()
    else:
        df["fast"]=df.Close.rolling(fast).mean()
        df["slow"]=df.Close.rolling(slow).mean()
    df["position"]=(df.fast>df.slow).astype(int)

df["ret"]=df.Close.pct_change().fillna(0)
df["trade"]=df.position.diff().abs().fillna(0)
df["strategy_ret"]=df.ret*df.position.shift(1).fillna(0)-df.trade*cost/100
df["equity"]=capital*(1+df.strategy_ret).cumprod()
df["buy_hold"]=capital*(1+df.ret).cumprod()
dd=df.equity/df.equity.cummax()-1
years=max((df.Date.iloc[-1]-df.Date.iloc[0]).days/365.25,.01)
cagr=(df.equity.iloc[-1]/capital)**(1/years)-1

a,b,c,d=st.columns(4)
a.metric("Final Capital",f"₹{df.equity.iloc[-1]:,.0f}")
b.metric("Strategy CAGR",f"{cagr*100:.2f}%")
c.metric("Max Drawdown",f"{dd.min()*100:.2f}%")
d.metric("Round Trips",int(df.trade.sum()/2))

fig,ax=plt.subplots(figsize=(12,4))
ax.plot(df.Date,df.equity,label="Strategy")
ax.plot(df.Date,df.buy_hold,label="Buy & Hold")
ax.set_title(instrument+" — Equity Curve")
ax.legend(); ax.grid(alpha=.25)
st.pyplot(fig)

st.subheader("Latest data")
st.dataframe(df.tail(20),use_container_width=True)
st.download_button("Download results CSV",df.to_csv(index=False).encode(),"backtest_results.csv","text/csv")

st.info("⚠️ The included data is synthetic demo data for testing the app only. Do not use its returns for investment decisions.")
