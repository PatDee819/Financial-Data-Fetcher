import yfinance as yf
from datetime import datetime

print("=" * 60)
print("GOLD TICKER TEST")
print(f"Current Time: {datetime.now()}")
print("=" * 60)

tickers_to_test = ["GC=F", "GLD", "GOLD", "GDX"]

for ticker in tickers_to_test:
    print(f"\n🔍 Testing: {ticker}")
    try:
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        
        if data.empty:
            print(f"   ❌ EMPTY - No data returned")
        else:
            print(f"   ✅ SUCCESS - {len(data)} bars received")
            print(f"   Last Close: {data['Close'].iloc[-1]:.2f}")
            print(f"   Date Range: {data.index[0]} to {data.index[-1]}")
            
            # Check volume
            if 'Volume' in data.columns:
                print(f"   Volume Available: YES (avg: {data['Volume'].mean():.0f})")
            else:
                print(f"   Volume Available: NO")
                
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
