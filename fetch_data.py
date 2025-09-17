import yfinance as yf
import pandas as pd
from datetime import datetime

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

def calculate_composite_score(results):
    try:
        vix_level = results.get("VIX_Current")
        vix_momentum = results.get("VIX_Momentum")
        gvz_level = results.get("GVZ_Current")
        gvz_momentum = results.get("GVZ_Momentum")
        dxy_level = results.get("DXY_Current")
        dxy_momentum = results.get("DXY_Momentum")
        if None in [vix_level, vix_momentum, gvz_level, gvz_momentum, dxy_level, dxy_momentum]:
            print("Cannot calculate Composite Score: Missing data")
            return None
        composite_score = (
            (vix_level * 0.35) +
            (vix_momentum * 0.25) +
            (gvz_level * 0.20) +
            (gvz_momentum * 0.10) +
            (dxy_level * 0.05) +
            (dxy_momentum * 0.05)
        )
        return composite_score
    except Exception as e:
        print(f"Error calculating Composite Score: {e}")
        return None

if __name__ == "__main__":
    tickers = {"VIX": "^VIX", "GVZ": "^GVZ", "DXY": "DX-Y.NYB"}
    momentum_period = "5d"
    results = {}

    for key, ticker in tickers.items():
        current, momentum = fetch_financial_data(ticker, momentum_period)
        results[f"{key}_Current"] = current
        results[f"{key}_Momentum"] = momentum

    composite_score = calculate_composite_score(results)
    results["Composite_Score"] = composite_score

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
            elif key == "Composite_Score":
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: Data not available")
