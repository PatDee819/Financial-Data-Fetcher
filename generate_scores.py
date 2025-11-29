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
# NEW: TIME-OF-DAY FILTER (Phase 1, #2)
# ==============================
def is_valid_trading_time():
    """
    Avoid low-liquidity Asian session hours: 00:00-03:59 UTC and 21:00-23:59 UTC
    (Assumes execution environment is running in UTC)
    """
    current_hour = datetime.now().hour # UTC hour
    
    # Avoid 00:00-03:59 UTC (dead Asian) and 21:00-23:59 UTC (post-NY close)
    if 0 <= current_hour < 4 or current_hour >= 21:
        return False
    return True


# ==============================
# 1. FETCH & COMPOSITE FUNCTIONS
# ==============================

def normalize_value(value, min_val, max_val):
    """Normalize value to 0-100 scale."""
    if value is None or min_val == max_val:
        return 50
    return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))

def fetch_financial_data(ticker, period="5d", interval="30m", retries=3):
    """
    Fetch data and calculate momentum, volatility, RSI (for GOLD), and Volume Ratio (for GOLD).
    ENHANCED: Better error handling for pandas Series formatting issues.
    """
    for attempt in range(retries):
        try:
            hist_data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if hist_data.empty or len(hist_data) < 20:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue

            # 🔧 FIX: Force conversion to Python float to avoid Series formatting errors
            try:
                current_price = float(hist_data["Close"].iloc[-1])
                initial_price = float(hist_data["Close"].iloc[-15])
            except (TypeError, ValueError, IndexError) as e:
                print(f"price conversion error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue
            
            momentum = ((current_price - initial_price) / initial_price) * 100
            
            log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            daily_volatility = log_returns.std() * np.sqrt(252 * 13)
            volatility = float(daily_volatility * 100)

            current_rsi = None
            volume_ratio = 1.0

            if ticker == TICKERS["GOLD"] or ticker == "GLD" or ticker == "GDX":
                # --- RSI Calculation ---
                period_rsi = 14
                delta = hist_data["Close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=period_rsi - 1, adjust=False).mean()
                avg_loss = loss.ewm(com=period_rsi - 1, adjust=False).mean()
                rs = avg_gain / avg_loss.replace(0, 1e-10) 
                rsi_value = 100 - (100 / (1 + rs))
                current_rsi = float(rsi_value.iloc[-1])
                
                # --- Volume Ratio Calculation for GOLD ---
                if 'Volume' in hist_data.columns and not hist_data['Volume'].empty:
                    try:
                        current_volume = float(hist_data["Volume"].iloc[-1])
                        avg_volume = float(hist_data["Volume"].mean())
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    except (TypeError, ValueError):
                        volume_ratio = 1.0

            return current_price, momentum, volatility, current_rsi, volume_ratio
            
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            
    return None, None, None, None, None

def calculate_composite_score(results):
    """
    Calculate weighted composite score.
    """
    
    inputs = {
        # LEVELS AND VOLATILITY (Total Weight: 0.44)
        "VIX_Level": (results.get("VIX_Current"), 10, 50, 0.12),
        "VIX_Volatility": (results.get("VIX_Volatility"), 0, 40, 0.05),
        "GVZ_Level": (results.get("GVZ_Current"), 10, 40, 0.03),
        "GVZ_Volatility": (results.get("GVZ_Volatility"), 0, 40, 0.02),
        "DXY_Level": (results.get("DXY_Current"), 80, 120, 0.12),
        "DXY_Volatility": (results.get("DXY_Volatility"), 0, 30, 0.05),
        "GOLD_Level": (results.get("GOLD_Current"), 1800, 3000, 0.02),
        "GOLD_Volatility": (results.get("GOLD_Volatility"), 0, 25, 0.03),
        
        # MOMENTUM (Total Weight: 0.56)
        "VIX_Momentum": (results.get("VIX_Momentum"), -5.0, 5.0, 0.10),
        "GVZ_Momentum": (results.get("GVZ_Momentum"), -5.0, 5.0, 0.02),
        "DXY_Momentum": (results.get("DXY_Momentum"), -3.0, 3.0, 0.08),
        "GOLD_Momentum": (results.get("GOLD_Momentum"), -1.0, 1.0, 0.36),
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
# 3. PREDICTIVE BIAS GENERATOR (ENHANCED v2.4 - Time/Volume Filters)
# ==============================

def generate_predictive_bias(results, current_score):
    """
    Generate trading bias with Time-of-Day and Volume Confirmation filters.
    """
    
    # 🛑 NEW: TIME-OF-DAY FILTER (Phase 1, #2)
    if not is_valid_trading_time():
        print(f"  ⏰ LOW liquidity hours detected (UTC {datetime.now().hour:02}:00) - Signal suppressed")
        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "bias": "NEUTRAL",
            "strength": 0.0,
            "confidence": 0,
            "action": "FLAT",
            "projected_move_pct": 0.0,
            "suggested_sl_points": 0.0,
            "position_size_factor": 0.0
        }

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
                
        # Fetch last signal from bias_signal.csv for invalidation check
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

    # --- MEAN-REVERSION FACTOR ---
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

    initial_bias = (slope * slope_multiplier * 10) + boost1
    
    strength_threshold = 3.0
    
    if initial_bias > strength_threshold:
        action = "LONG"
    elif initial_bias < -strength_threshold:
        action = "SHORT"
    else:
        action = "FLAT"
        
    bias = initial_bias

    # --- NEW: VOLUME CONFIRMATION (Phase 2, #7) ---
    volume_ratio = results.get("GOLD_VolumeRatio", 1.0)
    multiplier = 1.0
    
    if volume_ratio < 0.7:  # Low volume = weak signal
        multiplier = 0.7
        print(f"  📊 Volume penalty: {volume_ratio:.2f}x avg (bias reduced by 30%)")
    elif volume_ratio > 1.5:  # High volume = strong signal
        multiplier = 1.15
        print(f"  📊 Volume boost: {volume_ratio:.2f}x avg (bias enhanced by 15%)")
        
    bias *= multiplier
    
    # 🛑 Re-evaluate action based on new, adjusted bias
    if action != "FLAT":
        if abs(bias) < strength_threshold:
             print(f"  📉 Volume filter demoted signal to FLAT (Adjusted bias: {bias:+.1f})")
             action = "FLAT"
             bias = 0.0


    # --- 🔧 V2.3 UPDATE: TIGHTENED SIGNAL INVALIDATION LOGIC ---
    invalidation_triggered = False
    
    if last_action in ["LONG", "SHORT"] and time_since_signal < 14400:  # Within last 4 hours
        current_strength = abs(bias)
        strength_decay = last_strength - current_strength
        
        # NEW TIGHTENED THRESHOLDS (v2.3):
        if strength_decay > 2.0 or current_strength < 3.0:
            action = "FLAT"
            bias = 0.0
            invalidation_triggered = True
            print(f"  🚨 PROTECTIVE EXIT TRIGGERED (v2.3 - Tightened 2.0 Decay):")
            print(f"      Last: {last_action} @ {last_strength:.1f} strength")
            print(f"      Now: {current_strength:.1f} strength (decay: -{strength_decay:.1f})")
            print(f"      Threshold: 2.0 decay OR <3.0 strength (TIGHTENED)")
            print(f"      → Forcing FLAT to close position immediately.")

    # --- RSI Confirmation Filter ---
    gold_rsi = results.get("GOLD_RSI")
    rsi_threshold_long = 30
    rsi_threshold_short = 70

    filter_reason = None
    if gold_rsi is not None and not invalidation_triggered:
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
        suggested_sl_points = 20.0 
    else:
        time_scale_factor = np.sqrt(252 * 13)
        risk_points = current_gold_price * (gold_vol_annual / 100) / time_scale_factor * 2.5
        suggested_sl_points = round(max(20.0, risk_points), 1)
        
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
    print("🚀 FINANCIAL DATA FETCHER - STARTED")
    print("📌 VERSION: v2.5 - Enhanced GOLD Fallback & Error Handling")
    print("🔧 CHANGES: Triple Fallback (GC=F→GLD→GDX) | Series Fix")
    print("=" * 60)
    
    if not GITHUB_TOKEN or GITHUB_TOKEN in ["YOUR_GITHUB_TOKEN_HERE", "YOUR_PERSONAL_ACCESS_TOKEN_HERE"]:
        print("!!! CRITICAL FAILURE !!!")
        print("GITHUB_TOKEN is NOT set. Check environment variables.")
        print("EXECUTION HALTED.")
        print("=" * 60)
        return
    
    results = {}
    
    print("\n📊 Fetching market data...")
    for key, ticker in TICKERS.items():
        print(f"  → {key} ({ticker})...", end=" ")
        current, mom, vol, rsi, volume_ratio = fetch_financial_data(ticker) 
        
        # Enhanced VIX fallback
        if key == "VIX" and current is None:
            print("fallback to VXX...", end=" ")
            current, mom, vol, rsi, volume_ratio = fetch_financial_data("VXX")
        
        # 🆕 ENHANCED: Triple fallback for GOLD
        if key == "GOLD":
            if current is None:
                print("GC=F failed, trying GLD...", end=" ")
                current, mom, vol, rsi, volume_ratio = fetch_financial_data("GLD")
            
            if current is None:
                print("GLD failed, trying GDX...", end=" ")
                current, mom, vol, rsi, volume_ratio = fetch_financial_data("GDX")
            
            if current is None:
                print("❌ ALL GOLD TICKERS FAILED - CRITICAL ERROR")
                # Continue to next ticker instead of breaking
                continue
            
        if current is not None:
            # 🔧 FIX: Force conversion to native Python float
            safe_current = float(current)
            safe_mom = float(mom)
            safe_vol = float(vol)
            safe_rsi = float(rsi) if rsi is not None else None
            safe_volume_ratio = float(volume_ratio)
            
            results[f"{key}_Current"] = safe_current
            results[f"{key}_Momentum"] = safe_mom
            results[f"{key}_Volatility"] = safe_vol
            if safe_rsi is not None:
                results[f"{key}_RSI"] = safe_rsi
            if key == "GOLD":
                 results[f"{key}_VolumeRatio"] = safe_volume_ratio 
                    
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
        'GOLD_VolumeRatio': results.get('GOLD_VolumeRatio'), 
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
    print("\n🎯 Generating trading bias (v2.5 - Enhanced Fallback)...")
    signal = generate_predictive_bias(results, composite)
    signal_df = pd.DataFrame([signal])
    bias_content = signal_df.to_csv(index=False)
    
    success = upload_file_to_github("bias_signal.csv", bias_content, f"Signal Update v2.5 - {current_time_str}")
    
    if success:
        print(f"  ✅ bias_signal.csv uploaded")
        print(f"    → Action: {signal['action']}")
        print(f"    → Strength: {signal['strength']:.1f}")
        print(f"    → SL Points: {signal['suggested_sl_points']:.1f}")
        print(f"    → Position Size Factor: {signal['position_size_factor']:.2f}")
    else:
        print("  ❌ Failed to upload bias_signal.csv - SEE API ERROR ABOVE")

    print("\n" + "=" * 60)
    print("✅ EXECUTION COMPLETED (v2.5)")
    print("🔧 Active: Triple GOLD Fallback + Series Fix")
    print("=" * 60)

if __name__ == "__main__":
    main()
