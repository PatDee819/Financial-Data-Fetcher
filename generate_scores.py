import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import requests
from io import StringIO
import base64
import json
import os # NEW IMPORT

# ==============================
# GITHUB CONFIG (EDIT THESE)
# ==============================
# 🛑 CRITICAL CHANGE: Token is now read from an environment variable (GH_TOKEN)
# This variable will be set securely by GitHub Actions and is NOT visible in the code.
GITHUB_TOKEN = os.getenv("GH_TOKEN") 
REPO_OWNER = "PatDee819"
REPO_NAME = "Financial-Data-Fetcher"
BRANCH = "main"

# ==============================
# 1. FETCH & COMPOSITE (UNCHANGED)
# ==============================
def fetch_financial_data(ticker, period="5d", interval="1h", retries=3):
    for attempt in range(retries):
        try:
            asset = yf.Ticker(ticker)
            hist_data = asset.history(period=period, interval=interval)
            if hist_data.empty or len(hist_data) < 5:
                return None, None, None
            current_price = hist_data["Close"].iloc[-1]
            initial_price = hist_data["Close"].iloc[-5]
            momentum = ((current_price - initial_price) / initial_price) * 100
            returns = hist_data["Close"].pct_change().dropna()
            if len(returns) < 5 or any(abs(r) > 0.5 for r in returns):
                return None, None, None
            volatility = returns.std() * np.sqrt(252 * 6.5) * 100
            return current_price, momentum, volatility
        except Exception as e:
            time.sleep(2 ** attempt)
    return None, None, None

def normalize_value(value, min_val, max_val):
    if value is None or min_val == max_val: return 50
    return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))

def calculate_composite_score(results):
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
    score = sum(normalize_value(v, minv, maxv) * w for v, minv, maxv, w in inputs.values() if v is not None)
    return round(score, 2)

# ==============================
# 2. UPLOAD TO GITHUB VIA API
# ==============================
def upload_file_to_github(file_path, file_content, commit_message):
    # Added a check here in case the token isn't loaded (e.g., local run without setting env var)
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN is missing. Cannot upload to GitHub.")
        return False
        
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Get current file SHA (for updates)
    response = requests.get(url, headers=headers)
    sha = response.json().get("sha") if response.status_code == 200 else None
    
    data = {
        "message": commit_message,
        "content": base64.b64encode(file_content.encode("utf-8")).decode("utf-8"),
        "branch": BRANCH
    }
    if sha:
        data["sha"] = sha
    
    response = requests.put(url, headers=headers, data=json.dumps(data))
    return response.status_code == 200

# ==============================
# 3. MAIN: GENERATE & UPLOAD
# ==============================
def main():
    tickers = {"VIX": "^VIX", "GVZ": "^GVZ", "DXY": "DX-Y.NYB", "GOLD": "GC=F"}
    results = {}
    for key, ticker in tickers.items():
        current, mom, vol = fetch_financial_data(ticker)
        if key == "VIX" and current is None:
            current, mom, vol = fetch_financial_data("VXX")
        if key == "GOLD" and current is None:
            current, mom, vol = fetch_financial_data("GLD")
        results[f"{key}_Current"] = current
        results[f"{key}_Momentum"] = mom
        results[f"{key}_Volatility"] = vol
        time.sleep(3)

    composite = calculate_composite_score(results)
    results["Composite_Score"] = composite

    # ==============================
    # 4. GENERATE LEGACY scores.csv
    # ==============================
    df_legacy = pd.DataFrame([results])
    df_legacy = df_legacy[[
        'VIX_Current','VIX_Momentum','VIX_Volatility',
        'GVZ_Current','GVZ_Momentum','GVZ_Volatility',
        'DXY_Current','DXY_Momentum','DXY_Volatility',
        'GOLD_Current','GOLD_Momentum','GOLD_Volatility',
        'Composite_Score'
    ]]
    scores_content = df_legacy.to_csv(index=False, header=True, na_rep='')

    # Upload scores.csv
    if upload_file_to_github("scores.csv", scores_content, f"Update scores.csv - {datetime.now().strftime('%Y-%m-%d %H:%M:%S') }"):
        print("scores.csv → Uploaded to GitHub")

    # ==============================
    # 5. GENERATE NEW bias_signal.csv
    # ==============================
    signal = generate_predictive_bias(results, composite)
    signal_df = pd.DataFrame([signal])
    bias_content = signal_df.to_csv(index=False)
    
    # Upload bias_signal.csv
    if upload_file_to_github("bias_signal.csv", bias_content, f"Update bias_signal.csv - {datetime.now().strftime('%Y-%m-%d %H:%M:%S') }"):
        print(f"bias_signal.csv → Uploaded | Action: {signal['action']} | Bias: {signal['bias']} | Strength: {signal['strength']:+.1f}")

# ==============================
# 6. PREDICTIVE BIAS (SAME AS BEFORE)
# ==============================
def generate_predictive_bias(results, current_score):
    try:
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/scores.csv"
        hist = pd.read_csv(StringIO(requests.get(url, timeout=10).text))
        scores = hist['Composite_Score'].dropna().tail(12).tolist()

        if len(scores) >= 6:
            recent = np.mean(scores[-3:])
            prior = np.mean(scores[-6:-3])
            slope = recent - prior
        else:
            slope = 0

        vix_mom = results.get("VIX_Momentum", 0)
        gvz_mom = results.get("GVZ_Momentum", 0)
        dxy_mom = results.get("DXY_Momentum", 0)
        gold_mom = results.get("GOLD_Momentum", 0)

        boost1 = 3.0 if gvz_mom > 1.5 and vix_mom < -1 else 0
        boost2 = 2.5 if dxy_mom < -0.5 and results.get("GOLD_Volatility", 100) < 15 else 0
        boost3 = 1.5 if len(scores) >= 3 and (scores[-1] - scores[-2]) > (scores[-2] - scores[-3]) else 0

        projected = current_score + slope * 2.0 + boost1 + boost2 + boost3
        bias = projected - current_score

        if bias > 6:
            action = "LONG"
        elif bias < -6:
            action = "SHORT"
        else:
            action = "FLAT"

        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "bias": "BULL" if bias > 0 else "BEAR" if bias < 0 else "NEUTRAL",
            "strength": round(bias, 1),
            "confidence": min(95, 60 + abs(bias) * 3),
            "action": action,
            "projected_move_pct": round(bias * 0.22, 1)
        }
    except:
        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "bias": "NEUTRAL", "strength": 0.0, "confidence": 0,
            "action": "FLAT", "projected_move_pct": 0.0
        }

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
