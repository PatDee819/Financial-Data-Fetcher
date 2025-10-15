import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

def fetch_financial_data(ticker, period="5d", interval="1h", retries=3):
    for attempt in range(retries):
        try:
            asset = yf.Ticker(ticker)
            hist_data = asset.history(period=period, interval=interval)
            if hist_data.empty or len(hist_data) < 5:
                print(f"Insufficient data for {ticker}: {len(hist_data)} hours, Data: {hist_data}")
                return None, None, None
            current_price = hist_data["Close"].iloc[-1]
            initial_price = hist_data["Close"].iloc[-5]
            momentum = ((current_price - initial_price) / initial_price) * 100 if initial_price != 0 else None
            print(f"{ticker}: {len(hist_data)} hours available, momentum period: 5-hour, momentum: {momentum:.2f}%")
            returns = hist_data["Close"].pct_change().dropna()
            if len(returns) < 5:
                print(f"Unreliable data for {ticker}: Only {len(returns)} returns available")
                return None, None, None
            if any(abs(r) > 0.5 for r in returns):
                print(f"Unreliable data for {ticker}: Extreme returns detected {returns}")
                return None, None, None
            volatility = returns.std() * np.sqrt(252 * 6.5) * 100 if len(returns) > 1 else None
            print(f"{ticker}: Volatility: {volatility:.2f}%")
            return current_price, momentum, volatility
        except yf.utils.YFRateLimitError as e:
            wait_time = 2 ** attempt
            print(f"Rate limit hit for {ticker}: {e}. Retrying after {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error fetching {ticker} (attempt {attempt+1}/{retries}): {e}, Type: {type(e).__name__}")
            time.sleep(1)
    print(f"Failed to fetch {ticker} after {retries} attempts.")
    return None, None, None

def normalize_value(value, min_val, max_val):
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
        inputs = {
            "VIX_Level": (results.get("VIX_Current"), 10, 50, 0.10),
            "VIX_Momentum": (results.get("VIX_Momentum"), -10, 10, 0.05),
            "VIX_Volatility": (results.get("VIX_Volatility"), 0, 40, 0.05),
            "GVZ_Level": (results.get("GVZ_Current"), 10, 40, 0.15),
            "GVZ_Momentum": (results.get("GVZ_Momentum"), -10, 10, 0.10),
            "GVZ_Volatility": (results.get("GVZ_Volatility"), 0, 40, 0.05),
            "DXY_Level": (results.get("DXY_Current"), 80, 120, 0.10),
            "DXY_Momentum": (results.get("DXY_Momentum"), -5, 5, 0.05),
            "DXY_Volatility": (results.get("DXY_Volatility"), 0, 30, 0.05),
            "GOLD_Level": (results.get("GOLD_Current"), 2000, 4000, 0.10),
            "GOLD_Momentum": (results.get("GOLD_Momentum"), -3, 3, 0.20),
            "GOLD_Volatility": (results.get("GOLD_Volatility"), 0, 25, 0.05),
        }
        missing = [k for k, v in inputs.items() if v[0] is None]
        if len(missing) > 4:
            print(f"Cannot calculate Composite Score: Too many missing data for {', '.join(missing)}")
            return None
        for name, (value, min_val, max_val, weight) in inputs.items():
            normalized = normalize_value(value, min_val, max_val)
            print(f"{name}: Raw={value:.2f}, Normalized={normalized:.2f}")
        composite_score = sum(
            normalize_value(value, min_val, max_val) * weight if value is not None else 0
            for name, (value, min_val, max_val, weight) in inputs.items()
        )
        return composite_score
    except Exception as e:
        print(f"Error calculating Composite Score: {e}")
        return None

if __name__ == "__main__":
    tickers = {"VIX": "^VIX", "GVZ": "^GVZ", "DXY": "DX-Y.NYB", "GOLD": "GC=F"}
    results = {}
    for key, ticker in tickers.items():
        current, momentum, volatility = fetch_financial_data(ticker)
        if key == "VIX" and current is None:
            print(f"Retrying VIX with fallback ticker VXX")
            current, momentum, volatility = fetch_financial_data("VXX")
        if key == "GOLD" and current is None:
            print(f"Retrying GOLD with fallback ticker GLD")
            current, momentum, volatility = fetch_financial_data("GLD")
        results[f"{key}_Current"] = current
        results[f"{key}_Momentum"] = momentum
        results[f"{key}_Volatility"] = volatility
        time.sleep(5)

    composite_score = calculate_composite_score(results)
    results["Composite_Score"] = composite_score

    filename = "scores.csv"
    for key in results:
        if results[key] is None:
            results[key] = np.nan
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False, header=True, na_rep='')
    print(f"Data saved to {filename}")
    print(f"Fetched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for key, value in results.items():
        if pd.notna(value):
            if "Momentum" in key or "Volatility" in key:
                print(f"{key}: {value:.2f}%")
            elif key == "Composite_Score":
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: Data not available")
