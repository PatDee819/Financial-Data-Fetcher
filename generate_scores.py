import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import requests
from io import StringIO
import base64
import json
import os

# =========================================================
# GITHUB CONFIG (CRITICAL: ACTION REQUIRED)
# =========================================================
# ⚠️ ACTION REQUIRED: REPLACE PLACEHOLDERS
# 1. Ensure your GITHUB_TOKEN is set as an environment variable (GH_TOKEN) 
#    OR replace the line below with your actual token (not recommended for security).
GITHUB_TOKEN = os.getenv("GH_TOKEN")# Reads from environment variable (RECOMMENDED)
# GITHUB_TOKEN = "YOUR_PERSONAL_ACCESS_TOKEN_HERE"  # UNCOMMENT and use if not using env var
REPO_OWNER = "PatrickDemba" # <<< REPLACED WITH YOUR USERNAME
REPO_NAME = "Financial-Data-Fetcher" # <<< REPLACED WITH YOUR REPO NAME
BRANCH = "main"

TICKERS = {
    "GOLD": "GC=F",     
    "VIX": "^VIX",      
    "GVZ": "^GVZ",      
    "DXY": "DX-Y.NYB"   
}
SCORES_FILE = "scores.csv"

# ==============================
# 1. FETCH & COMPOSITE FUNCTIONS
# ==============================

def normalize_value(value, min_val, max_val):
    """Normalize value to 0-100 scale."""
    if value is None or min_val == max_val: 
        return 50
    return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))

def fetch_financial_data(ticker, period="5d", interval="30m", retries=3):
    """Fetch data and calculate momentum, volatility, and RSI (for GOLD)."""
    for attempt in range(retries):
        try:
            hist_data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if hist_data.empty or len(hist_data) < 5:
                continue

            current_price = hist_data["Close"].iloc[-1]
            initial_price = hist_data["Close"].iloc[-5]
            
            # Momentum (% change over 5 periods)
            momentum = ((current_price - initial_price) / initial_price) * 100
            
            # Volatility (Annualized Standard Deviation of Log Returns)
            log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            daily_volatility = log_returns.std() * np.sqrt(252 * 13)
            volatility = daily_volatility * 100 # Convert to percentage

            # --- RSI CALCULATION (for GOLD only) ---
            current_rsi = None
            if ticker == TICKERS["GOLD"] or ticker == "GLD": 
                period_rsi = 14
                delta = hist_data["Close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=period_rsi - 1, adjust=False).mean()
                avg_loss = loss.ewm(com=period_rsi - 1, adjust=False).mean()
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))
                current_rsi = rsi_value.iloc[-1]

            # Return scalar values
            return current_price, momentum, volatility, current_rsi
            
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)
            
    return None, None, None, None

def calculate_composite_score(results):
    """Calculate weighted composite score with tighter momentum scaling."""
    # Define (Value, Min Range, Max Range, Weight)
    inputs = {
        # Levels and Volatility 
        "VIX_Level": (results.get("VIX_Current"), 10, 50, 0.10),
        "VIX_Volatility": (results.get("VIX_Volatility"), 0, 40, 0.05),
        "GVZ_Level": (results.get("GVZ_Current"), 10, 40, 0.15),
        "GVZ_Volatility": (results.get("GVZ_Volatility"), 0, 40, 0.05),
        "DXY_Level": (results.get("DXY_Current"), 80, 120, 0.10),
        "DXY_Volatility": (results.get("DXY_Volatility"), 0, 30, 0.05),
        "GOLD_Level": (results.get("GOLD_Current"), 1800, 3000, 0.02), 
        "GOLD_Volatility": (results.get("GOLD_Volatility"), 0, 25, 0.05),
        
        # MOMENTUM (Tighter Scaling)
        "VIX_Momentum": (results.get("VIX_Momentum"), -5.0, 5.0, 0.05),
        "GVZ_Momentum": (results.get("GVZ_Momentum"), -5.0, 5.0, 0.10),
        "DXY_Momentum": (results.get("DXY_Momentum"), -3.0, 3.0, 0.05),
        "GOLD_Momentum": (results.get("GOLD_Momentum"), -1.0, 1.0, 0.20),
    }
    
    score = sum(normalize_value(v, minv, maxv) * w for v, minv, maxv, w in inputs.values() if v is not None)
    return round(score, 2)

# ==============================
# 2. GITHUB API LOGIC
# ==============================

def upload_file_to_github(file_path, file_content, commit_message):
    """Upload or update a file in GitHub repository."""
    # FIX: Check both environment variable and placeholder for the token
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN is missing or empty. Cannot upload to GitHub.")
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
    
    if response.status_code in [200, 201]:
        return True
    else:
        # CRITICAL FIX: Print full error details for debugging
        try:
            error_message = response.json().get('message', response.text)
        except json.JSONDecodeError:
            error_message = response.text
        
        print(f"❌ GitHub upload failed: HTTP {response.status_code}")
        print(f"  API Response Error: {error_message}")
        return False

# ==============================
# 3. PREDICTIVE BIAS GENERATOR (ENHANCED)
# ==============================

def generate_predictive_bias(results, current_score):
    """Generate trading bias using dynamic momentum/mean-reversion and RSI filter."""
    scores = []
    slope = 0
    
    # --- Fetch Historical Scores for Slope Calculation ---
    # CRITICAL FIX: Use configured REPO_OWNER/REPO_NAME for fetching scores.csv
    try:
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{SCORES_FILE}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            hist = pd.read_csv(StringIO(response.text))
            scores = hist['Composite_Score'].dropna().tail(12).tolist()

            if len(scores) >= 6:
                recent = np.mean(scores[-3:])
                prior = np.mean(scores[-6:-3])
                slope = recent - prior
                print(f"  Slope calculated: {slope:.2f}")
            
    except Exception as e:
        # FIX: Added error printing to see why fetching scores fails
        print(f"⚠️ Warning: Failed to fetch scores.csv for slope calculation: {e}")
        pass

    # --- MEAN-REVERSION FACTOR ---
    if current_score > 80: 
        slope_multiplier = -2.0  # FADE the move
    elif current_score < 20: 
        slope_multiplier = -2.0  # FADE the move
    else:
        slope_multiplier = 2.0   # FOLLOW the trend
        
    
    # --- DYNAMIC BOOSTS ---
    vix_mom = results.get("VIX_Momentum", 0)
    gvz_mom = results.get("GVZ_Momentum", 0)
    dxy_mom = results.get("DXY_Momentum", 0)
    gold_vol = results.get("GOLD_Volatility", 100)
    
    boost1 = 0
    if gvz_mom > 1.0 and vix_mom < -1.0: boost1 = 3.0
    elif gvz_mom < -1.0 and vix_mom > 1.0: boost1 = -3.0
        
    boost2 = 0
    if dxy_mom < -0.5 and gold_vol < 15: boost2 = 2.5
    elif dxy_mom > 0.5 and gold_vol < 15: boost2 = -2.5

    boost3 = 0
    if len(scores) >= 3 and (scores[-1] - scores[-2]) > 0: boost3 = 1.5
    elif len(scores) >= 3 and (scores[-1] - scores[-2]) < 0: boost3 = -1.5
        
    # Calculate projected score and bias
    projected = current_score + (slope * slope_multiplier) + boost1 + boost2 + boost3
    bias = projected - current_score

    # Determine initial trading action
    if bias > 3.0:
        action = "LONG"
    elif bias < -3.0:
        action = "SHORT"
    else:
        action = "FLAT"

    # --- RSI Confirmation Filter ---
    gold_rsi = results.get("GOLD_RSI")
    rsi_threshold_long = 40 
    rsi_threshold_short = 60 

    if gold_rsi is not None:
        if action == "LONG" and gold_rsi > rsi_threshold_short:
            action = "FLAT" 
        elif action == "SHORT" and gold_rsi < rsi_threshold_long:
            action = "FLAT" 
    
    # Update bias if action was filtered to FLAT
    if action == "FLAT":
        bias = 0.0
    
    # --- Dynamic Stop-Loss (SL) Calculation ---
    gold_vol_annual = results.get("GOLD_Volatility", 20.0) 
    current_gold_price = results.get("GOLD_Current")
    
    if current_gold_price is None or current_gold_price == 0:
        suggested_sl_points = 15.0 # Fallback risk
    else:
        time_scale_factor = np.sqrt(252 * 13) 
        risk_points = current_gold_price * (gold_vol_annual / 100) / time_scale_factor * 4.0 
        suggested_sl_points = round(max(5.0, risk_points), 1) 
        
    print(f"  Final Action: {action} (Bias: {bias:+.1f} | SL: {suggested_sl_points:.1f} pts)")

    return {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "bias": "BULL" if bias > 0 else "BEAR" if bias < 0 else "NEUTRAL",
        "strength": round(abs(bias), 1),
        "confidence": min(95, 60 + abs(bias) * 3),
        "action": action,
        "projected_move_pct": round(bias * 0.22, 1),
        "suggested_sl_points": suggested_sl_points 
    }

# ==============================
# 4. MAIN EXECUTION
# ==============================
def main():
    print("=" * 60)
    print("🚀 FINANCIAL DATA FETCHER - STARTED")
    print("=" * 60)
    
    # CRITICAL: Check if GITHUB_TOKEN is available before proceeding
    if not GITHUB_TOKEN or GITHUB_TOKEN in ["YOUR_GITHUB_TOKEN_HERE", "YOUR_PERSONAL_ACCESS_TOKEN_HERE"]:
        print("!!! CRITICAL FAILURE !!!")
        print("GITHUB_TOKEN is NOT set or is still a placeholder.")
        print(f"Please set the GH_TOKEN environment variable or manually update the script.")
        print("EXECUTION HALTED.")
        print("=" * 60)
        return
    
    results = {}
    
    print("\n📊 Fetching market data...")
    for key, ticker in TICKERS.items():
        print(f"  → {key} ({ticker})...", end=" ")
        current, mom, vol, rsi = fetch_financial_data(ticker) 
        
        # Fallback tickers 
        if key == "VIX" and current is None:
            print("fallback to VXX...", end=" ")
            current, mom, vol, rsi = fetch_financial_data("VXX")
        if key == "GOLD" and current is None:
            print("fallback to GLD...", end=" ")
            current, mom, vol, rsi = fetch_financial_data("GLD")
            
        if current is not None:
            
            # --- FIX: SAFELY EXTRACT SCALAR VALUES ---
            # This ensures we handle any unexpected Series return from yfinance 
            # and prevents the TypeError.
            safe_current = np.array(current).item()
            safe_mom = np.array(mom).item()
            safe_vol = np.array(vol).item()
            safe_rsi = np.array(rsi).item() if rsi is not None else None
            
            # Store safe scalars in results dictionary
            results[f"{key}_Current"] = safe_current
            results[f"{key}_Momentum"] = safe_mom
            results[f"{key}_Volatility"] = safe_vol
            if safe_rsi is not None:
                 results[f"{key}_RSI"] = safe_rsi
                 
            print(f"✓ (Price: {safe_current:.2f}, Mom: {safe_mom:+.2f}%)")
        else:
            print("✗ FAILED")
        
        time.sleep(1) 

    # Calculate composite score
    composite = calculate_composite_score(results)
    print(f"\n📈 Composite Score: {composite:.2f}")

    # ==============================
    # APPEND TO scores.csv
    # ==============================
    print("\n💾 Updating scores.csv...")
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # CRITICAL FIX: Use configured REPO_OWNER/REPO_NAME for fetching scores.csv
    try:
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{SCORES_FILE}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            existing_df = pd.read_csv(StringIO(response.text))
        else:
            print(f"⚠️ Warning: Could not fetch existing scores.csv (HTTP {response.status_code}). Starting new file.")
            existing_df = pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Warning: Network error fetching scores.csv. Starting new file. Error: {e}")
        existing_df = pd.DataFrame()

    current_data = {
        'Timestamp': current_time_str,
        'VIX_Current': results.get('VIX_Current'),
        'VIX_Momentum': results.get('VIX_Momentum'),
        'VIX_Volatility': results.get('VIX_Volatility'),
        'GVZ_Current': results.get('GVZ_Current'),
        'GVZ_Momentum': results.get('GVZ_Momentum'),
        'GVZ_Volatility': results.get('GVZ_Volatility'),
        'DXY_Current': results.get('DXY_Current'),
        'DXY_Momentum': results.get('DXY_Momentum'),
        'DXY_Volatility': results.get('DXY_Volatility'),
        'GOLD_Current': results.get('GOLD_Current'),
        'GOLD_Momentum': results.get('GOLD_Momentum'),
        'GOLD_Volatility': results.get('GOLD_Volatility'),
        'Composite_Score': composite
    }
    current_reading = pd.DataFrame([current_data])

    df_legacy = pd.concat([existing_df, current_reading], ignore_index=True).tail(48)
    
    scores_content = df_legacy.to_csv(index=False)
    if upload_file_to_github(SCORES_FILE, scores_content, f"Score Update - {current_time_str}"):
        print(f"  ✅ {SCORES_FILE} uploaded successfully ({len(df_legacy)} readings)")

    # ==============================
    # GENERATE bias_signal.csv
    # ==============================
    print("\n🎯 Generating trading bias...")
    signal = generate_predictive_bias(results, composite)
    signal_df = pd.DataFrame([signal])
    bias_content = signal_df.to_csv(index=False)
    
    success = upload_file_to_github("bias_signal.csv", bias_content, f"Signal Update - {current_time_str}")
    
    if success:
        print(f"  ✅ bias_signal.csv uploaded")
        print(f"    → Action: {signal['action']}")
        print(f"    → SL Points: {signal['suggested_sl_points']:+.1f}")
    else:
        # FIX: This block will now print the full API error from the upload_file_to_github function
        print("  ❌ Failed to upload bias_signal.csv - SEE API ERROR ABOVE")

    print("\n" + "=" * 60)
    print("✅ EXECUTION COMPLETED")
    print("=" * 60)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
