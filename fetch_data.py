import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_financial_data(ticker, period="2d", interval="1h"):
    try:
        asset = yf.Ticker(ticker)
        hist_data = asset.history(period=period, interval=interval)
        if hist_data.empty or len(hist_data) < 5:
            print(f"Insufficient data for {ticker}: {len(hist_data)} hours")
            return None, None, None
        current_price = hist_data["Close"].iloc[-1]
        # 5-hour momentum
        initial_price = hist_data["Close"].iloc[-5]
        momentum = ((current_price - initial_price) / initial_price) * 100 if initial_price != 0 else None
        print(f"{ticker}: {len(hist_data)} hours available, momentum period: 5-hour, momentum: {momentum:.2f}%")
        # Volatility: annualized std dev of hourly returns
        returns = hist_data["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else None
        print(f"{ticker}: Volatility: {volatility:.2f}%")
        return current_price, momentum, volatility
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None, None, None

def normalize_value(value, min_val, max_val):
    """Normalize to 0-100 scale, clamping to avoid outliers."""
    if value is None or min_val is None or max_val is None:
        return None
    try:
        normalized = ((value - min_val) / (max_val - min_val)) * 100 if max_val != min_val else 50
        return max(0, min(100, normalized))
    except Exception as e:
        print(f"Normalization error: {e}")
        return None

def calculate_composite_score(results):
    try:
        # Input values with normalization ranges and weights
        inputs = {
            "VIX_Level": (results.get("VIX_Current"), 10, 50, 0.10),
            "VIX_Momentum": (results.get("VIX_Momentum"), -5, 5, 0.05),
            "VIX_Volatility": (results.get("VIX_Volatility"), 2, 20, 0.05),
            "GVZ_Level": (results.get("GVZ_Current"), 10, 40, 0.15),
            "GVZ_Momentum": (results.get("GVZ_Momentum"), -5, 5, 0.10),
            "GVZ_Volatility": (results.get("GVZ_Volatility"), 2, 20, 0.05),
            "DXY_Level": (results.get("DXY_Current"), 80, 120, 0.10),
            "DXY_Momentum": (results.get("DXY_Momentum"), -5, 5, 0.05),
            "DXY_Volatility": (results.get("DXY_Volatility"), 2, 20, 0.05),
            "GOLD_Level": (results.get("GOLD_Current"), 2000, 4000, 0.10),
            "GOLD_Momentum": (results.get("GOLD_Momentum"), -3, 3, 0.20),
            "GOLD_Volatility": (results.get("GOLD_Volatility"), 2, 20, 0.05),
        }
        if None in [v[0] for v in inputs.values()]:
            print("Cannot calculate Composite Score: Missing data")
            return None
        # Compute weighted sum of normalized values
        composite_score = sum(
            normalize_value(value, min_val, max_val) * weight
            for name, (value, min_val, max_val, weight) in inputs.items()
        )
        return composite_score
    except Exception as e:
        print(f"Error calculating Composite Score: {e}")
        return None

if __name__ == "__main__":
    # Tickers for yFinance
    tickers = {"VIX": "^VIX", "GVZ": "^GVZ", "DXY": "DX-Y.NYB", "GOLD": "GC=F"}
    results = {}

    # Fetch data for each ticker
    for key, ticker in tickers.items():
        current, momentum, volatility = fetch_financial_data(ticker)
        results[f"{key}_Current"] = current
        results[f"{key}_Momentum"] = momentum
        results[f"{key}_Volatility"] = volatility

    # Calculate composite score
    composite_score = calculate_composite_score(results)
    results["Composite_Score"] = composite_score

    # Save to CSV
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f"financial_data_{timestamp}.csv"
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)

    # Print results
    print(f"Data saved to {filename}")
    print(f"Fetched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for key, value in results.items():
        if value is not None:
            if "Momentum" in key or "Volatility" in key:
                print(f"{key}: {value:.2f}%")
            elif key == "Composite_Score":
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: Data not available")
