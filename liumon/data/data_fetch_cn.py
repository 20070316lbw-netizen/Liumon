import os
import sys
import baostock as bs
import pandas as pd
from datetime import datetime, date
from tqdm import tqdm

# Ensure config is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PRICE_DIR, CN_DIR

def _std_to_bs(symbol: str) -> str:
    if symbol.upper().endswith(".SS"): return f"sh.{symbol[:-3]}"
    if symbol.upper().endswith(".SZ"): return f"sz.{symbol[:-3]}"
    return symbol

def fetch_stock_data(ticker_std, start_date="2014-01-01", end_date=None):
    """
    Catch-all fetching function: Handles both initial download and incremental updates.
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
        
    file_path = os.path.join(PRICE_DIR, f"{ticker_std}.parquet")
    
    # Logic for incremental vs full
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        last_date = existing_df.index.max()
        actual_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if actual_start > end_date:
            return "UP_TO_DATE", existing_df
        is_incremental = True
    else:
        actual_start = start_date
        is_incremental = False
        existing_df = None

    try:
        rs = bs.query_history_k_data_plus(
            _std_to_bs(ticker_std),
            "date,open,high,low,close,volume,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
            start_date=actual_start, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        
        rows = []
        while rs.next(): rows.append(rs.get_row_data())
        
        if not rows:
            return "NO_NEW_DATA", existing_df
            
        new_df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume","turn","pe","pb","ps","pcf","is_st"])
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df["ticker"] = ticker_std
        for c in ["open","high","low","close","volume","turn","pe","pb","ps","pcf"]:
            new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
        
        new_df.set_index("date", inplace=True)
        
        if is_incremental:
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
        else:
            combined_df = new_df.sort_index()
            
        combined_df.to_parquet(file_path, compression="snappy")
        return "SUCCESS", combined_df
        
    except Exception as e:
        print(f"Error fetching {ticker_std}: {e}")
        return "ERROR", None

def get_index_tickers():
    """HS300 + CSI500 Universe"""
    bs.login()
    tickers = []
    for query_func in [bs.query_hs300_stocks, bs.query_zz500_stocks]:
        rs = query_func()
        while rs.next():
            r = rs.get_row_data()
            tickers.append(r[1].split(".")[1] + (".SS" if r[1].startswith("sh") else ".SZ"))
    bs.logout()
    return list(set(tickers))

def main():
    print("="*60)
    print("  Liumon Data Hub: A-Share Historical & Increment Fetcher")
    print("="*60)
    
    bs.login()
    tickers = get_index_tickers()
    print(f">> Project Universe: {len(tickers)} stocks (HS300 + CSI500)")
    
    success_stocks = []
    
    # Process a subset or all
    # For demonstration/first run, we might want to just show one stock's audit as requested
    for i, ticker in enumerate(tqdm(tickers, desc="Syncing Data")):
        status, df = fetch_stock_data(ticker)
        if status == "SUCCESS":
            success_stocks.append(ticker)
        
        # User visibility: Show info for the first successful stock found
        if i == 0 and df is not None:
            print(f"\n[Audit] Data Sample for {ticker}:")
            print("-" * 30)
            print(df.tail(3))
            print("\n[Audit] Data Info:")
            df.info()
            print("\n[Audit] Statistics (Recent):")
            print(df.tail(100).describe())
            print("-" * 30)

    bs.logout()
    print(f"\n[DONE] Successfully synced {len(success_stocks)} stocks. Data located in: {PRICE_DIR}")

if __name__ == "__main__":
    main()
