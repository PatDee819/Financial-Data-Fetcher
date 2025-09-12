import yfinance as yf
import pandas as pd
from datetime import datetime
import base64
import os

def fetch_financial_data(ticker, period="5d"):
    try:
        asset = yf.Ticker(ticker)
        current_price = asset.history(period="1d")["Close"].iloc[-1]
        hist_data = asset.history(period=period)
        if len(hist_data) >= 2:
            initial_price = hist_data["Close"].iloc[0]
            final_price = hist_data["Close"].iloc[-1]
            momentum = ((final_price - initial_price) / initial_price) * 100
        else:
            momentum = None
        return current_price, momentum
    except Exception as e:
        print(f"Error for {ticker}: {e}")
        return None, None

if __name__ == "__main__":
    tickers = {"VIX": "^VIX", "GVZ": "^GVZ", "DXY": "DX-Y.NYB"}
    momentum_period = "5d"
    results = {}

    for key, ticker in tickers.items():
        current, momentum = fetch_financial_data(ticker, momentum_period)
        results[f"{key}_Current"] = current
        results[f"{key}_Momentum"] = momentum

    # Save to CSV in repo (via env var for GitHub token if needed)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f"financial_data_{timestamp}.csv"
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)

    print(f"Data saved to {filename}")
    print(f"Fetched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for key, value in results.items():
        if value is not None:
            if "Momentum" in key:
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: Data not available")
