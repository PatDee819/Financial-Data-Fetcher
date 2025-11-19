import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
import os

# --- CONFIGURATION ---
REPO_OWNER = "YOUR_GITHUB_USERNAME"  # <--- REPLACE THIS
REPO_NAME = "YOUR_REPO_NAME"        # <--- REPLACE THIS
BRANCH = "main"                     # Or 'master', depending on your repo

TICKERS = {
    "GOLD": "GC=F",     # Gold Futures (Primary Trade Asset)
    "VIX": "^VIX",      # VIX (Equity Fear/Risk-Off)
    "GVZ": "^GVZ",      # Gold Volatility Index
    "DXY": "DX-Y.NYB"   # US Dollar Index
}
OUTPUT_FILE = "bias_signal.csv"
SCORES_FILE = "scores.csv"

# --- HELPER FUNCTIONS ---

def normalize_value(value, min_val, max_val):
    """Normalize value to 0-100 scale."""
    if value is None or min_val == max_val: 
        return 50
    # Ensure value stays within the defined min/max range for normalization
    return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))

def fetch_financial_data(ticker, period="5d", interval="30m", retries=3):
    """Fetch financial data, calculate momentum and volatility, including RSI for GOLD."""
    for attempt in range(retries):
        try:
            hist_data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if hist_data.empty or len(hist_data) < 5:
                print(f"    [FAIL] Data too short for {ticker}. Attempt {attempt + 1}")
                continue

            current_price = hist_data["Close"].iloc[-1]
            initial_price = hist_data["Close"].iloc[-5] # Price 5 periods ago (2.5 hours)
            
            # Momentum (% change over 5 periods)
            momentum = ((current_price - initial_price) / initial_price) * 100
            
            # Volatility (Annualized Standard Deviation of Log Returns)
            log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            daily_volatility = log_returns.std() * np.sqrt(252 * 13) # 252 days * 13 (30m periods per day)
            volatility = daily_volatility * 100 # Convert to percentage

            # --- RSI CALCULATION (Only for GOLD) ---
            current_rsi = None
            if ticker == TICKERS["GOLD"]:
                period_rsi = 14
                delta = hist_data["Close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                # Use EMA for smoothing
                avg_gain = gain.ewm(com=period_rsi - 1, adjust=False).mean()
                avg_loss = loss.ewm(com=period_rsi - 1, adjust=False).mean()
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))
                current_rsi = rsi_value.iloc[-1]

            return current_price, momentum, volatility, current_rsi
            
        except Exception as e:
            print(f"    [ERROR] Failed to fetch data for {ticker}: {e}. Attempt {attempt + 1}")

    return None, 0.0, 0.0, None # Return default values on final failure


def calculate_composite_score(results):
    """
    Calculate weighted composite score (0-100) from market indicators.
    Tighter momentum ranges enhance sensitivity to extreme moves.
    """
    
    # Define (Value, Min Range, Max Range, Weight)
    inputs = {
        # LEVELS (Higher Score = Higher Price/Level = Risk-Off)
        "VIX_Level": (results.get("VIX_Current"), 10, 50, 0.10),  # 10%
        "GVZ_Level": (results.get("GVZ_Current"), 10, 40, 0.15),  # 15%
        "DXY_Level": (results.get("DXY_Current"), 80, 120, 0.10), # 10%
        "GOLD_Level": (results.get("GOLD_Current"), 1800, 3000, 0.02), # 2% (Lowest weight on price level)
        
        # VOLATILITY (Higher Score = Higher Volatility = Risk-Off)
        "VIX_Volatility": (results.get("VIX_Volatility"), 0, 40, 0.05), # 5%
        "GVZ_Volatility": (results.get("GVZ_Volatility"), 0, 40, 0.05), # 5%
        "DXY_Volatility": (results.get("DXY_Volatility"), 0, 30, 0.05), # 5%
        "GOLD_Volatility": (results.get("GOLD_Volatility"), 0, 25, 0.05), # 5%
        
        # MOMENTUM (TIGHTER SCALING APPLIED HERE)
        # Tighter ranges amplify the score, triggering mean-reversion logic faster.
        # Negative momentum is mapped to low scores (Risk-On)
        "VIX_Momentum": (results.get("VIX_Momentum"), -5.0, 5.0, 0.05),    # 5%
        "GVZ_Momentum": (results.get("GVZ_Momentum"), -5.0, 5.0, 0.10),    # 10%
        "DXY_Momentum": (results.get("DXY_Momentum"), -3.0, 3.0, 0.05),    # 5%
        "GOLD_Momentum": (results.get("GOLD_Momentum"), -1.0, 1.0, 0.20),   # 20% (Highest weight)
    }
    
    score = sum(normalize_value(v, minv, maxv) * w for v, minv, maxv, w in inputs.values() if v is not None)
    return round(score, 2)


def generate_predictive_bias(results, current_score):
    """
    Generate trading bias using dynamic momentum and mean-reversion.
    Includes the NEW RSI filter and Dynamic SL points suggestion.
    """
    scores = []
    slope = 0
    
    # --- 1. Fetch Historical Scores for Slope Calculation ---
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
                print(f"✅ Historical slope calculated: {slope:.2f} (from {len(scores)} readings)")
            else:
                print(f"⚠️ Only {len(scores)} readings available. Need 6 for slope calculation.")
        else:
            print(f"⚠️ scores.csv not found or inaccessible (HTTP {response.status_code})")
            
    except Exception as e:
        print(f"⚠️ Cannot access scores.csv: {e}")

    # --- 2. MEAN-REVERSION FACTOR (New Logic) ---
    # Dynamically reverse the slope multiplier when the score is at market extremes.
    
    if current_score > 80: # Highly overbought, start fading the trend
        slope_multiplier = -2.0  # Reverse the slope impact
        print(f"🔄 Mean Reversion Active: Score > 80. Slope Multiplier: {slope_multiplier:.1f}")
    elif current_score < 20: # Highly oversold, start fading the trend
        slope_multiplier = -2.0  # Reverse the slope impact
        print(f"🔄 Mean Reversion Active: Score < 20. Slope Multiplier: {slope_multiplier:.1f}")
    else:
        slope_multiplier = 2.0   # Neutral zone, follow the trend (momentum)
        
    
    # --- 3. DYNAMIC BOOSTS (Adjusted Logic) ---
    vix_mom = results.get("VIX_Momentum", 0)
    gvz_mom = results.get("GVZ_Momentum", 0)
    dxy_mom = results.get("DXY_Momentum", 0)
    gold_vol = results.get("GOLD_Volatility", 100)
    
    boost1 = 0
    if gvz_mom > 1.0 and vix_mom < -1.0:
        boost1 = 3.0  # Bullish: Gold Volatility rising, Equity Fear (VIX) falling
    elif gvz_mom < -1.0 and vix_mom > 1.0:
        boost1 = -3.0 # Bearish: Gold Volatility falling, Equity Fear (VIX) rising
        
    boost2 = 0
    if dxy_mom < -0.5 and gold_vol < 15:
        boost2 = 2.5 # Bullish: DXY weak, GOLD low vol (stable long)
    elif dxy_mom > 0.5 and gold_vol < 15:
        boost2 = -2.5 # Bearish: DXY strong, GOLD low vol (stable short)

    boost3 = 0
    if len(scores) >= 3 and (scores[-1] - scores[-2]) > 0: # Check for acceleration
        boost3 = 1.5
    elif len(scores) >= 3 and (scores[-1] - scores[-2]) < 0:
        boost3 = -1.5
        
    # --- 4. Calculate Final Bias ---
    projected = current_score + (slope * slope_multiplier) + boost1 + boost2 + boost3
    bias = projected - current_score

    # Determine initial trading action
    if bias > 3.0:
        action = "LONG"
    elif bias < -3.0:
        action = "SHORT"
    else:
        action = "FLAT"

    # --- 5. RSI Confirmation Filter (Step 1 implementation from prior response) ---
    gold_rsi = results.get("GOLD_RSI")
    rsi_threshold_long = 40  # Oversold (Buy if signal is LONG and RSI < 40)
    rsi_threshold_short = 60 # Overbought (Sell if signal is SHORT and RSI > 60)

    if gold_rsi is not None:
        if action == "LONG" and gold_rsi > rsi_threshold_short:
            print(f"⚠️ LONG Blocked by RSI ({gold_rsi:.1f}): Gold is Overbought.")
            action = "FLAT" 
            
        elif action == "SHORT" and gold_rsi < rsi_threshold_long:
            print(f"⚠️ SHORT Blocked by RSI ({gold_rsi:.1f}): Gold is Oversold.")
            action = "FLAT" 
    
    # Update bias if action was filtered to FLAT
    if action == "FLAT":
        bias = 0.0
    else:
        # Recalculate bias for logging purposes, incorporating only non-filtered moves
        bias = projected - current_score 

    # --- 6. Dynamic Stop-Loss (SL) Calculation (Step 2 implementation) ---
    gold_vol_annual = results.get("GOLD_Volatility", 20.0) 
    current_gold_price = results.get("GOLD_Current")
    
    if current_gold_price is None or current_gold_price == 0:
        suggested_sl_points = 15.0 # Fallback risk
    else:
        # Factor to convert annualized volatility to a 30m period estimate: sqrt(252 trading days * 13 periods per day)
        time_scale_factor = np.sqrt(252 * 13) 
        # Calculate Risk_Points = Price * Volatility * Safety_Factor(4.0) / Time_Scale_Factor
        risk_points = current_gold_price * (gold_vol_annual / 100) / time_scale_factor * 4.0 
        
        # Min SL of 5.0 points (0.50 price movement on a $1 point scale)
        suggested_sl_points = round(max(5.0, risk_points), 1) 
        
    print(f"🎯 Bias: {bias:.1f} | Action: {action} | RSI: {gold_rsi:.1f} | SL: {suggested_sl_points:.1f} pts")

    return {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "bias": "BULL" if bias > 0 else "BEAR" if bias < 0 else "NEUTRAL",
        "strength": round(abs(bias), 1), # Use absolute strength for the EA
        "confidence": min(95, 60 + abs(bias) * 3),
        "action": action,
        "projected_move_pct": round(bias * 0.22, 1),
        "suggested_sl_points": suggested_sl_points # NEW PARAMETER for EA
    }

# --- MAIN EXECUTION ---

def main():
    print("--- Running Gold Predictive Bias Generator ---")
    results = {}

    for key, ticker in TICKERS.items():
        print(f"Fetching data for {key} ({ticker})...")
        current, mom, vol, rsi = fetch_financial_data(ticker)
        
        if current is None:
            print(f"Skipping {key} due to data failure.")
            continue
            
        results[f"{key}_Current"] = current
        results[f"{key}_Momentum"] = mom
        results[f"{key}_Volatility"] = vol
        if rsi is not None:
            results[f"{key}_RSI"] = rsi

    if not results:
        print("❌ All data fetches failed. Cannot proceed.")
        return

    # --- 1. Calculate Composite Score ---
    current_score = calculate_composite_score(results)
    print(f"\n📈 Composite Score: {current_score:.2f} (0=Extreme Risk-On, 100=Extreme Risk-Off)")
    
    # --- 2. Generate Predictive Bias & Action ---
    signal_data = generate_predictive_bias(results, current_score)

    # --- 3. Write Output Files ---
    
    # a) Write Signal to CSV for the EA
    signal_df = pd.DataFrame([signal_data])
    signal_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Signal saved to {OUTPUT_FILE}: {signal_data['action']} @ {signal_data['suggested_sl_points']} pts SL")

    # b) Append Score to Historical Scores File
    score_entry = {
        "timestamp": signal_data["timestamp"],
        "Composite_Score": current_score,
    }
    score_df = pd.DataFrame([score_entry])

    if os.path.exists(SCORES_FILE):
        score_df_hist = pd.read_csv(SCORES_FILE)
        score_df = pd.concat([score_df_hist.tail(99), score_df], ignore_index=True) # Keep file small
    
    score_df.to_csv(SCORES_FILE, index=False)
    print(f"✅ Score appended to {SCORES_FILE}")

if __name__ == "__main__":
    main()
