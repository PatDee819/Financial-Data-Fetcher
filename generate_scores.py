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

# ==============================
# GITHUB CONFIG (EDIT THESE)
# ==============================
GITHUB_TOKEN = os.getenv("GH_TOKEN") 
REPO_OWNER = "PatDee819"
REPO_NAME = "Financial-Data-Fetcher"
BRANCH = "main"

# ==============================
# 1. FETCH & COMPOSITE 
# ==============================
def fetch_financial_data(ticker, period="5d", interval="30m", retries=3):
    """Fetch current price, momentum, and volatility for a given ticker."""
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
            print(f"⚠️  Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)
    return None, None, None

def normalize_value(value, min_val, max_val):
    """Normalize value to 0-100 scale."""
    if value is None or min_val == max_val: 
        return 50
    return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))

def calculate_composite_score(results):
    """Calculate weighted composite score from market indicators."""
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
        "GOLD_Level": (results.get("GOLD_Current"), 2000, 5000, 0.02), 
        "GOLD_Momentum": (results.get("GOLD_Momentum"), -1.5, 1.5, 0.20),
        "GOLD_Volatility": (results.get("GOLD_Volatility"), 0, 25, 0.05),
    }
    score = sum(normalize_value(v, minv, maxv) * w for v, minv, maxv, w in inputs.values() if v is not None)
    return round(score, 2)

# ==============================
# 2. UPLOAD TO GITHUB VIA API
# ==============================
def upload_file_to_github(file_path, file_content, commit_message):
    """Upload or update a file in GitHub repository."""
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN is missing. Cannot upload to GitHub.")
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
        print(f"❌ GitHub upload failed: {response.status_code} - {response.text}")
        return False

# ==============================
# 3. PREDICTIVE BIAS GENERATOR
# ==============================
def generate_predictive_bias(results, current_score):
    """Generate trading bias based on historical trends and current momentum."""
    slope = 0
    scores = []
    
    try:
        # Fetch historical scores from GitHub
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/scores.csv"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            hist = pd.read_csv(StringIO(response.text))
            scores = hist['Composite_Score'].dropna().tail(12).tolist()

            if len(scores) >= 6:
                recent = np.mean(scores[-3:])
                prior = np.mean(scores[-6:-3])
                slope = recent - prior
                print(f"✅ Historical slope calculated: {slope:.2f} (from {len(scores)} readings)")
            else:
                print(f"⚠️  Only {len(scores)} readings available. Need 6 for slope calculation.")
        else:
            print(f"⚠️  scores.csv not found or inaccessible (HTTP {response.status_code})")
            
    except Exception as e:
        print(f"⚠️  Cannot access scores.csv: {e}")

    # Extract current momentum values
    vix_mom = results.get("VIX_Momentum", 0)
    gvz_mom = results.get("GVZ_Momentum", 0)
    dxy_mom = results.get("DXY_Momentum", 0)
    gold_mom = results.get("GOLD_Momentum", 0)

    # Calculate momentum boosts
    boost1 = 3.0 if gvz_mom > 1.5 and vix_mom < -1 else 0
    boost2 = 2.5 if dxy_mom < -0.5 and results.get("GOLD_Volatility", 100) < 15 else 0
    boost3 = 1.5 if len(scores) >= 3 and (scores[-1] - scores[-2]) > (scores[-2] - scores[-3]) else 0

    # Calculate projected score and bias
    projected = current_score + (slope * 2.0) + boost1 + boost2 + boost3
    bias = projected - current_score

    # Determine trading action
    if bias > 3.0:
        action = "LONG"
    elif bias < -3.0:
        action = "SHORT"
    else:
        action = "FLAT"

    print(f"🎯 Bias Calculation: slope={slope:.2f}, boost1={boost1:.1f}, boost2={boost2:.1f}, boost3={boost3:.1f}")
    print(f"🎯 Final Bias: {bias:.1f} | Action: {action}")

    return {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "bias": "BULL" if bias > 0 else "BEAR" if bias < 0 else "NEUTRAL",
        "strength": round(bias, 1),
        "confidence": min(95, 60 + abs(bias) * 3),
        "action": action,
        "projected_move_pct": round(bias * 0.22, 1)
    }

# ==============================
# 4. MAIN EXECUTION
# ==============================
def main():
    print("=" * 60)
    print("🚀 FINANCIAL DATA FETCHER - STARTED")
    print(f"⏰ Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Fetch market data for all tickers
    tickers = {"VIX": "^VIX", "GVZ": "^GVZ", "DXY": "DX-Y.NYB", "GOLD": "GC=F"}
    results = {}
    
    print("\n📊 Fetching market data...")
    for key, ticker in tickers.items():
        print(f"  → {key} ({ticker})...", end=" ")
        current, mom, vol = fetch_financial_data(ticker)
        
        # Fallback tickers if primary fails
        if key == "VIX" and current is None:
            print("fallback to VXX...", end=" ")
            current, mom, vol = fetch_financial_data("VXX")
        if key == "GOLD" and current is None:
            print("fallback to GLD...", end=" ")
            current, mom, vol = fetch_financial_data("GLD")
        
        results[f"{key}_Current"] = current
        results[f"{key}_Momentum"] = mom
        results[f"{key}_Volatility"] = vol
        
        if current is not None:
            print(f"✓ (Price: {current:.2f}, Mom: {mom:+.2f}%)")
        else:
            print("✗ FAILED")
        
        time.sleep(3)

    # Calculate composite score
    composite = calculate_composite_score(results)
    results["Composite_Score"] = composite
    print(f"\n📈 Composite Score: {composite:.2f}")

    # ==============================
    # APPEND TO scores.csv (FIXED)
    # ==============================
    print("\n💾 Updating scores.csv...")
    
    try:
        url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/scores.csv"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            existing_df = pd.read_csv(StringIO(response.text))
            
            # Check if Timestamp is first column
            if 'Timestamp' in existing_df.columns and existing_df.columns[0] != 'Timestamp':
                print("  ⚠️  Wrong column order detected - reordering...")
                cols = ['Timestamp'] + [col for col in existing_df.columns if col != 'Timestamp']
                existing_df = existing_df[cols]
            
            print(f"  📖 Loaded {len(existing_df)} existing readings")
        else:
            existing_df = pd.DataFrame()
            print("  📝 Creating new scores.csv (first reading)")
            
    except Exception as e:
        existing_df = pd.DataFrame()
        print(f"  📝 Creating new scores.csv: {e}")

    # Create current reading with proper column order
    current_data = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        'Composite_Score': results.get('Composite_Score')
    }

    current_reading = pd.DataFrame([current_data])

    # Append to historical data
    if not existing_df.empty:
        df_legacy = pd.concat([existing_df, current_reading], ignore_index=True)
    else:
        df_legacy = current_reading

    # Keep only last 24 readings (12 hours)
    df_legacy = df_legacy.tail(48)
    
    print(f"  💾 Total readings after append: {len(df_legacy)}")

    # Upload scores.csv
    scores_content = df_legacy.to_csv(index=False)
    if upload_file_to_github("scores.csv", scores_content, f"Reading #{len(df_legacy)} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
        print(f"  ✅ scores.csv uploaded successfully ({len(df_legacy)} readings)")
    else:
        print("  ❌ Failed to upload scores.csv")

    # ==============================
    # GENERATE bias_signal.csv
    # ==============================
    print("\n🎯 Generating trading bias...")
    signal = generate_predictive_bias(results, composite)
    signal_df = pd.DataFrame([signal])
    bias_content = signal_df.to_csv(index=False)
    
    if upload_file_to_github("bias_signal.csv", bias_content, f"Update bias - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
        print(f"  ✅ bias_signal.csv uploaded")
        print(f"     → Action: {signal['action']}")
        print(f"     → Bias: {signal['bias']}")
        print(f"     → Strength: {signal['strength']:+.1f}")
        print(f"     → Confidence: {signal['confidence']:.0f}%")
    else:
        print("  ❌ Failed to upload bias_signal.csv")

    print("\n" + "=" * 60)
    print("✅ EXECUTION COMPLETED")
    print("=" * 60)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
