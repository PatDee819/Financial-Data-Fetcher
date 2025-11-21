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
from numpy.polynomial.polynomial import polyfit

# =========================================================
# GITHUB CONFIG (CRITICAL: ACTION REQUIRED)
# =========================================================
GITHUB_TOKEN = os.getenv("GH_TOKEN")
REPO_OWNER = "PatDee819"
REPO_NAME = "Financial-Data-Fetcher"
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
            
            if hist_data.empty or len(hist_data) < 20:
                continue

            current_price = hist_data["Close"].iloc[-1]
            initial_price = hist_data["Close"].iloc[-15]
            
            momentum = ((current_price - initial_price) / initial_price) * 100
            
            log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            daily_volatility = log_returns.std() * np.sqrt(252 * 13)
            volatility = daily_volatility * 100

            current_rsi = None
            if ticker == TICKERS["GOLD"] or ticker == "GLD":
                period_rsi = 14
                delta = hist_data["Close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=period_rsi - 1, adjust=False).mean()
                avg_loss = loss.ewm(com=period_rsi - 1, adjust=False).mean()
                rs = avg_gain / avg_loss.replace(0, 1e-10) 
                rsi_value = 100 - (100 / (1 + rs))
                current_rsi = rsi_value.iloc[-1]

            return current_price, momentum, volatility, current_rsi
            
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)
            
    return None, None, None, None

def calculate_composite_score(results):
    """
    Calculate weighted composite score.
    🔧 FIX: Reduced GVZ weight to minimize circular logic (5% total, was 15%)
    """
    
    inputs = {
        # LEVELS AND VOLATILITY (Total Weight: 0.44)
        "VIX_Level": (results.get("VIX_Current"), 10, 50, 0.12),      # Increased
        "VIX_Volatility": (results.get("VIX_Volatility"), 0, 40, 0.05),
        "GVZ_Level": (results.get("GVZ_Current"), 10, 40, 0.03),      # 🔧 REDUCED from 0.10
        "GVZ_Volatility": (results.get("GVZ_Volatility"), 0, 40, 0.02), # 🔧 REDUCED from 0.05
        "DXY_Level": (results.get("DXY_Current"), 80, 120, 0.12),     # Increased
        "DXY_Volatility": (results.get("DXY_Volatility"), 0, 30, 0.05),
        "GOLD_Level": (results.get("GOLD_Current"), 1800, 3000, 0.02),
        "GOLD_Volatility": (results.get("GOLD_Volatility"), 0, 25, 0.03),
        
        # MOMENTUM (Total Weight: 0.56)
        "VIX_Momentum": (results.get("VIX_Momentum"), -5.0, 5.0, 0.10), # Increased
        "GVZ_Momentum": (results.get("GVZ_Momentum"), -5.0, 5.0, 0.02), # 🔧 REDUCED from 0.10
        "DXY_Momentum": (results.get("DXY_Momentum"), -3.0, 3.0, 0.08), # Increased
        "GOLD_Momentum": (results.get("GOLD_Momentum"), -1.0, 1.0, 0.36), # Slightly increased
    }
    
    score = sum(normalize_value(v, minv, maxv) * w for v, minv, maxv, w in inputs.values() if v is not None)
    return round(score, 2)

# ==============================
# 2. GITHUB API LOGIC
# ==============================

def upload_file_to_github(file_path, file_content, commit_message):
    """Upload or update a file in GitHub repository."""
    
    if not GITHUB_TOKEN or GITHUB_TOKEN in ["YOUR_GITHUB_TOKEN_HERE", "YOUR_PERSONAL_ACCESS_TOKEN_HERE"]:
        print("❌ Error: GITHUB_TOKEN is missing or empty. Cannot upload to GitHub.")
        return False
        
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
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
        try:
            error_message = response.json().get('message', response.text)
        except json.JSONDecodeError:
            error_message = response.text
            
        print(f"❌ GitHub upload failed: HTTP {response.status_code}")
        print(f"  API Response Error: {error_message}")
        return False

# ==============================
# 3. PREDICTIVE BIAS GENERATOR (ENHANCED v2.1)
# ==============================

def generate_predictive_bias(results, current_score):
    """
    Generate trading bias with signal invalidation logic.
    🔧 FIX: Added trade invalidation when strength collapses after entry
    """
    scores = []
    slope = 0.0
    last_action = None
    last_strength = 0.0
    time_since_signal = 999999  # Large number = no recent signal
    
    # --- Fetch Historical Scores for Slope Calculation & Last Signal ---
    try:
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{SCORES_FILE}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            hist = pd.read_csv(StringIO(response.text))
            scores = hist['Composite_Score'].dropna().tail(10).tolist()

            if len(scores) >= 5:
                x = np.arange(len(scores))
                coefficients = polyfit(x, scores, 1) 
                slope = coefficients[1]
                print(f"  Linear Regression Slope calculated: {slope:.3f}")
                
        # 🔧 NEW: Fetch last signal from bias_signal.csv for invalidation check
        signal_url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/bias_signal.csv"
        signal_response = requests.get(signal_url, timeout=10)
        
        if signal_response.status_code == 200:
            signal_data = pd.read_csv(StringIO(signal_response.text))
            if not signal_data.empty:
                last_row = signal_data.iloc[-1]
                last_action = last_row.get('action', None)
                last_strength = abs(float(last_row.get('strength', 0)))
                
                # Calculate time since last signal
                last_timestamp = pd.to_datetime(last_row.get('timestamp'))
                time_since_signal = (datetime.now() - last_timestamp).total_seconds()
                
                print(f"  Last Signal: {last_action} | Strength: {last_strength:.1f} | Age: {time_since_signal/3600:.1f}h")
            
    except Exception as e:
        print(f"⚠️ Warning: Failed to fetch historical data: {e}")
        pass

    # --- MEAN-REVERSION FACTOR (Simplified - Less Aggressive) ---
    # 🔧 FIX: Only fade at EXTREME levels (>90 or <10), follow momentum elsewhere
    if current_score > 90:
        slope_multiplier = -1.5  # Fade extreme bullishness
    elif current_score < 10:
        slope_multiplier = -1.5  # Fade extreme bearishness
    else:
        slope_multiplier = 1.5   # Follow momentum in 10-90 range
    
    # --- DYNAMIC BOOSTS ---
    vix_mom = results.get("VIX_Momentum", 0)
    dxy_mom = results.get("DXY_Momentum", 0)
    
    boost1 = 0.0
    if vix_mom > 1.0 and dxy_mom < -0.5: 
        boost1 = 3.0
    elif vix_mom < -1.0 and dxy_mom > 0.5: 
        boost1 = -3.0

    bias = (slope * slope_multiplier * 10) + boost1
    
    strength_threshold = 3.0
    
    if bias > strength_threshold:
        action = "LONG"
    elif bias < -strength_threshold:
        action = "SHORT"
    else:
        action = "FLAT"

    # --- 🔧 NEW: SIGNAL INVALIDATION LOGIC ---
    # If we had a recent active signal (LONG/SHORT) and strength has collapsed, invalidate trade
    invalidation_triggered = False
    
    if last_action in ["LONG", "SHORT"] and time_since_signal < 14400:  # Within last 4 hours
        current_strength = abs(bias)
        strength_decay = last_strength - current_strength
        
        # Invalidate if:
        # 1. Strength dropped by more than 4.0 points, OR
        # 2. Current strength is now below 2.0 (weak signal)
        if strength_decay > 4.0 or current_strength < 2.0:
            action = "FLAT"
            bias = 0.0
            invalidation_triggered = True
            print(f"  🚨 SIGNAL INVALIDATION TRIGGERED:")
            print(f"     Last: {last_action} @ {last_strength:.1f} strength")
            print(f"     Now: {current_strength:.1f} strength (decay: -{strength_decay:.1f})")
            print(f"     → Forcing FLAT to close position")

    # --- RSI Confirmation Filter ---
    gold_rsi = results.get("GOLD_RSI")
    rsi_threshold_long = 30
    rsi_threshold_short = 70

    filter_reason = None
    if gold_rsi is not None and not invalidation_triggered:  # Don't apply RSI if already invalidated
        if action == "LONG" and gold_rsi > rsi_threshold_short:
            action = "FLAT"
            filter_reason = f"RSI filter hit: {gold_rsi:.1f} > {rsi_threshold_short}"
        elif action == "SHORT" and gold_rsi < rsi_threshold_long:
            action = "FLAT"
            filter_reason = f"RSI filter hit: {gold_rsi:.1f} < {rsi_threshold_long}"
    
    if action == "FLAT" and not invalidation_triggered:
        bias = 0.0
        
    # --- Dynamic Stop-Loss Calculation ---
    gold_vol_annual = results.get("GOLD_Volatility", 20.0)
    current_gold_price = results.get("GOLD_Current")
    
    if current_gold_price is None or current_gold_price == 0:
        suggested_sl_points = 20.0  # 🔧 INCREASED from 15.0 (safer fallback)
    else:
        time_scale_factor = np.sqrt(252 * 13)
        risk_points = current_gold_price * (gold_vol_annual / 100) / time_scale_factor * 2.5
        suggested_sl_points = round(max(20.0, risk_points), 1)  # 🔧 INCREASED minimum from 5.0 to 20.0
        
    # --- Position Sizing Factor ---
    scaled_strength = min(abs(bias), 6.0)
    position_size_factor = round(1.0 + (scaled_strength / 6.0) * 1.0, 2)
        
    print(f"  Final Action: {action} (Bias: {bias:+.1f} | SL: {suggested_sl_points:.1f} pts | Size Factor: {position_size_factor:.2f})")
    if filter_reason:
        print(f"  Reason FLAT: {filter_reason}")

    return {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "bias": "BULL" if bias > 0 else "BEAR" if bias < 0 else "NEUTRAL",
        "strength": round(abs(bias), 1),
        "confidence": min(95, 60 + abs(bias) * 3),
        "action": action,
        "projected_move_pct": round(bias * 0.22, 1),
        "suggested_sl_points": suggested_sl_points,
        "position_size_factor": position_size_factor
    }

# ==============================
# 4. MAIN EXECUTION
# ==============================
def main():
    print("=" * 60)
    print("🚀 FINANCIAL DATA FETCHER - STARTED (V2.1 - Signal Invalidation)")
    print("=" * 60)
    
    if not GITHUB_TOKEN or GITHUB_TOKEN in ["YOUR_GITHUB_TOKEN_HERE", "YOUR_PERSONAL_ACCESS_TOKEN_HERE"]:
        print("!!! CRITICAL FAILURE !!!")
        print("GITHUB_TOKEN is NOT set or is still a placeholder. Check line 14.")
        print("Please set the GH_TOKEN environment variable or manually update the script.")
        print("EXECUTION HALTED.")
        print("=" * 60)
        return
    
    results = {}
    
    print("\n📊 Fetching market data...")
    for key, ticker in TICKERS.items():
        print(f"  → {key} ({ticker})...", end=" ")
        current, mom, vol, rsi = fetch_financial_data(ticker)
        
        if key == "VIX" and current is None:
            print("fallback to VXX...", end=" ")
            current, mom, vol, rsi = fetch_financial_data("VXX")
        if key == "GOLD" and current is None:
            print("fallback to GLD...", end=" ")
            current, mom, vol, rsi = fetch_financial_data("GLD")
            
        if current is not None:
            safe_current = current.item() if hasattr(current, 'item') else current
            safe_mom = mom.item() if hasattr(mom, 'item') else mom
            safe_vol = vol.item() if hasattr(vol, 'item') else vol
            safe_rsi = rsi.item() if hasattr(rsi, 'item') else rsi if rsi is not None else None
            
            results[f"{key}_Current"] = safe_current
            results[f"{key}_Momentum"] = safe_mom
            results[f"{key}_Volatility"] = safe_vol
            if safe_rsi is not None:
                results[f"{key}_RSI"] = safe_rsi
                            
            print(f"✓ (Price: {safe_current:.2f}, Mom: {safe_mom:+.2f}%)")
        else:
            print("✗ FAILED")
        
        time.sleep(1)

    composite = calculate_composite_score(results)
    print(f"\n📈 Composite Score: {composite:.2f}")

    # ==============================
    # APPEND TO scores.csv
    # ==============================
    print("\n💾 Updating scores.csv...")
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
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
        print(f"  ✅ {SCORES_FILE} uploaded successfully ({len(df_legacy)} readings)")

    # ==============================
    # GENERATE bias_signal.csv
    # ==============================
    print("\n🎯 Generating trading bias...")
    signal = generate_predictive_bias(results, composite)
    signal_df = pd.DataFrame([signal])
    bias_content = signal_df.to_csv(index=False)
    
    success = upload_file_to_github("bias_signal.csv", bias_content, f"Signal Update - {current_time_str}")
    
    if success:
        print(f"  ✅ bias_signal.csv uploaded")
        print(f"    → Action: {signal['action']}")
        print(f"    → SL Points: {signal['suggested_sl_points']:+.1f}")
        print(f"    → Position Size Factor: {signal['position_size_factor']:.2f} (Use for risk scaling)")
    else:
        print("  ❌ Failed to upload bias_signal.csv - SEE API ERROR ABOVE")

    print("\n" + "=" * 60)
    print("✅ EXECUTION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
