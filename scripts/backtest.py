import sys
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    tqdm.write = print

# 加入系统路径以引入主模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_signal import generate_signal
from backtest.config import BACKTEST_CONFIG
from backtest.signal_recorder import SignalRecorder
from backtest.performance_analyzer import analyze_performance

def compute_future_actual_data(ticker: str, date: str, horizon: int):
    """
    计算基于指定日期的未来 N 天真实收益率、真实波动率 (std) 及真实振幅 (High-Low/Close)
    """
    try:
        trade_date = datetime.strptime(date, "%Y-%m-%d")
        end_date = trade_date + timedelta(days=horizon + 10)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=date, end=end_date.strftime("%Y-%m-%d"))
        
        if df.empty or len(df) < horizon + 1:
            return None, None, None
            
        # 裁剪出确切的 horizon 窗口数据 (含 T 日)
        window_df = df.iloc[:horizon+1]
        
        p_entry = window_df.iloc[0]["Close"]
        p_exit = window_df.iloc[-1]["Close"]
        
        # 指标1：收益率
        actual_return = float((p_exit / p_entry) - 1.0)
        
        # 指标2：真实波动率 (收益率序列的 std)
        daily_returns = window_df["Close"].pct_change().dropna()
        realized_vol = float(daily_returns.std()) if len(daily_returns) > 1 else 0.0
        
        # 指标3：真实振幅 (期间最高 - 期间最低) / 期初
        actual_max = window_df["High"].max()
        actual_min = window_df["Low"].min()
        actual_range_pct = float((actual_max - actual_min) / p_entry)
        
        return actual_return, realized_vol, actual_range_pct
        
    except Exception as e:
        print(f"  [X] Failed fetching actual return for {ticker} at {date}: {e}")
        return None, None, None

def process_single_ticker(args):
    ticker, sample_dates, record_file_path = args
    print(f"\n[Worker] ========== Starting Backtest for {ticker} ==========")
    stock_results = []
    
    # 为了避免多个进程打印重叠，可以去掉 tqdm 或者简单降级，这里保留以看到进度
    for date in tqdm(sample_dates, desc=f"Processing {ticker}", leave=False):
        try:
            signal = generate_signal(ticker=ticker, as_of_date=date)
            
            fut_ret_1d, fut_vol_1d, fut_range_1d = compute_future_actual_data(ticker, date, horizon=1)
            fut_ret_5d, fut_vol_5d, fut_range_5d = compute_future_actual_data(ticker, date, horizon=5)
            
            if fut_ret_1d is None or fut_ret_5d is None:
                continue
                
            meta = signal.get("metadata", {})
            record = {
                "date": date,
                "ticker": ticker,
                "regime": signal.get("regime", "UNKNOWN"),
                "z_score": signal.get("z_score", 0.0),
                "regime_strength": signal.get("regime_strength", 0.0),
                "direction": signal["direction"],
                "mean_return": signal["mean_return"],
                "uncertainty": signal["uncertainty"],
                "predicted_range_pct": signal.get("predicted_range_pct", 0.0),
                "adjusted_position_strength": signal["adjusted_position_strength"],
                "sentiment_score": meta.get("sentiment_score", 0.0),
                "risk_factor": meta.get("risk_factor", 0.0),
                "future_return_1d": fut_ret_1d,
                "realized_vol_1d": fut_vol_1d,
                "actual_range_1d": fut_range_1d,
                "future_return_5d": fut_ret_5d,
                "realized_vol_5d": fut_vol_5d,
                "actual_range_5d": fut_range_5d
            }
            stock_results.append(record)
            
        except Exception as e:
            tqdm.write(f"  [X] Failed processing {ticker} at {date}: {e}")
            
    if stock_results:
        # 为防止多线程锁冲突写坏 JSONL，各进程写自己的小独立卷文件
        import json
        ticker_file = record_file_path.replace(".jsonl", f"_{ticker}.jsonl")
        with open(ticker_file, "w", encoding="utf-8") as f:
            for rec in stock_results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(stock_results), ticker_file
    return 0, None


def run_backtest():
    import concurrent.futures
    import multiprocessing
    
    universe = BACKTEST_CONFIG["universe"]
    start_date = BACKTEST_CONFIG["start_date"]
    end_date = BACKTEST_CONFIG["end_date"]
    
    recorder = SignalRecorder(BACKTEST_CONFIG["output_dir"])
    record_file_path = recorder.record_file
    
    print("Pre-fetching SPY calendar to determine trading days...")
    spy = yf.Ticker("SPY")
    spy_df = spy.history(start=start_date, end=end_date)
    if spy_df.empty:
        print("Failed to fetch trading days. Aborting.")
        return
        
    trading_dates = spy_df.index.tz_localize(None).strftime("%Y-%m-%d").tolist()
    print(f"Found {len(trading_dates)} trading days from {start_date} to {end_date}.")
    sample_dates = trading_dates
    print(f"Selected {len(sample_dates)} sampling dates per stock for backtesting (Expected total: {len(sample_dates) * len(universe)}).")

    # 并发安全阀门：总核心的一半，且硬压不超过 8
    total_cores = multiprocessing.cpu_count()
    safe_workers = min(8, max(1, total_cores // 2))
    print(f"\n⚡ 核聚变引擎启动：检测到 {total_cores} 个逻辑核心，安全起见将挂载 {safe_workers} 个车道并发回测！\n")

    tasks = [(ticker, sample_dates, record_file_path) for ticker in universe]
    
    # 启用多进程池发包
    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
        results = list(executor.map(process_single_ticker, tasks))
    
    # 文件大一统合流
    total_records = 0
    with open(record_file_path, "w", encoding="utf-8") as outfile:
        for count, t_file in results:
            if t_file and os.path.exists(t_file):
                total_records += count
                with open(t_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                os.remove(t_file) # 切割完毕后销毁碎片
                
    print(f"\n========== 🚀 All Tasks Completed! Total Records: {total_records} ==========")
    if total_records > 0:
        analyze_performance(record_file_path)
    else:
        print("No valid results collected.")

if __name__ == "__main__":
    # 多进程 Windows 挂载保护
    import multiprocessing
    multiprocessing.freeze_support()
    run_backtest()
