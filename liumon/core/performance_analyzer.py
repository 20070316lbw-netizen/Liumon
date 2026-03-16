import os
import json
import numpy as np
import pandas as pd

def analyze_performance(log_file: str):
    """
    Kronos Alpha V2 纯统计解析：
    不仅评测收盘方向，更硬核检验波动率相关性、区间预判能力以及状态 (Regime) 捕捉。
    """
    if not os.path.exists(log_file):
        print("Performance Analyzer failed: Log file not found.")
        return
        
    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if not records:
         print("No data available to analyze.")
         return
         
    df = pd.DataFrame(records)
    total_samples = len(df)
    
    print("\n" + "="*80)
    print("📈 KRONOS ALPHA V2 - REGIME & VOLATILITY ANALYSIS REPORT 📈")
    print("="*80)
    print(f"Total Samples Processed: {total_samples}")
    
    if total_samples == 0:
        return

    for horizon in ["1d", "5d"]:
        target_ret_col = f"future_return_{horizon}"
        target_vol_col = f"realized_vol_{horizon}"
        target_range_col = f"actual_range_{horizon}"
        if target_ret_col not in df.columns:
            print(f"Skipping horizon {horizon} because column {target_ret_col} is missing from log.")
            continue
            
        # 必须剔除 None 才能算 mean 等，不然全算成 nan 或者报错
        valid_df = df.dropna(subset=[target_ret_col]).copy()
        if valid_df.empty:
            print(f"No valid data available for {horizon} horizon (maybe all target returns are None).")
            continue
        print(f"\n" + "-"*35 + f" ⏳ Horizon: {horizon.upper()} " + "-"*35)
        
        # -------------------------------------------------------------
        # Part 1: V1 遗留基础维度 (总体胜率与方向相关度)
        # -------------------------------------------------------------
        global_avg_ret = df[target_ret_col].mean()
        print(f"\n[1] DIRECTION & MOMENTUM (Classic)")
        print(f"Global Benchmark (Hold average return):  {global_avg_ret:.2%}")
        
        # 活跃信号胜率 (仅限强趋势行情下的调用)
        active_signals = df[df["regime"].str.contains("STRONG")]
        if not active_signals.empty:
            buy_mask = (active_signals["direction"] == "BUY") & (active_signals[target_ret_col] > 0)
            sell_mask = (active_signals["direction"] == "SELL") & (active_signals[target_ret_col] < 0)
            hits = (buy_mask | sell_mask).sum()
            win_rate = hits / len(active_signals)
            print(f"Strong Regime Win Rate (Directional):    {win_rate:.2%} ({hits}/{len(active_signals)})")
        else:
            print("No STRONG regime signals generated.")
            
        corr_dir = valid_df["z_score"].corr(valid_df[target_ret_col])
        print(f"Pearson Corr (Z-Score/Strength, Return): {corr_dir:.4f}")

        # -------------------------------------------------------------
        # Part 2: V2 新增维度 - 波动评估能力 (Volatility & Range)
        # -------------------------------------------------------------
        print(f"\n[2] VOLATILITY & DISTRIBUTION AWARENESS (Phase 6 Core)")
        if target_range_col in df.columns and "predicted_range_pct" in valid_df.columns:
            # 去除可能存在的 NaN 极值
            clean_df = valid_df.dropna(subset=['uncertainty', target_vol_col, 'predicted_range_pct', target_range_col])
            if not clean_df.empty:
                # 假设 1: 预测标准差 (uncertainty) 应该与真实未来波动率 (realized_vol) 正相关
                corr_vol = clean_df["uncertainty"].corr(clean_df[target_vol_col])
                print(f"Corr(Predicted Std, Realized Vol):       {corr_vol:.4f}  <-- Volatility Awareness")
                
                # 假设 2: 预测宽幅 (predicted_range) 应该与真实极值振幅 (actual_range) 正相关
                corr_range = clean_df["predicted_range_pct"].corr(clean_df[target_range_col])
                print(f"Corr(Predicted Range, Actual Range):     {corr_range:.4f}  <-- Range Awareness")
            else:
                print("Missing clean volatility data for correlation.")

        # -------------------------------------------------------------
        # Part 3: V2 新增维度 - 行情状态切片评估 (Regime Performance)
        # -------------------------------------------------------------
        print(f"\n[3] REGIME SEGMENTATION PERFORMANCE")
        # 对数据进行三种状态切片：STRONG_UP, STRONG_DOWN, RANGING
        up_df = df[df["regime"] == "STRONG_TREND_UP"]
        down_df = df[df["regime"] == "STRONG_TREND_DOWN"]
        range_df = df[df["regime"] == "RANGING_MIXED"]
        
        def safe_mean(subset, col):
            return f"{subset[col].mean():.2%}" if not subset.empty else "N/A   "
            
        print("Regime Type         | Sample Count | Avg Future Return | Strategy Theoretical RTN")
        print("--------------------+--------------+-------------------+--------------------------")
        
        # 上涨趋势理论收益 = 持有多仓的真实收益
        rtn_up = up_df[target_ret_col].mean() if not up_df.empty else 0
        print(f"STRONG_TREND_UP     | {len(up_df):<12} | {safe_mean(up_df, target_ret_col):<17} | {rtn_up:.2%} (All Long)")
        
        # 下跌趋势理论收益 = 做空持仓的真实收益 (反转真实收益的符号)
        rtn_down = -down_df[target_ret_col].mean() if not down_df.empty else 0
        print(f"STRONG_TREND_DOWN   | {len(down_df):<12} | {safe_mean(down_df, target_ret_col):<17} | {rtn_down:.2%} (All Short)")
        
        # 震荡趋势极简测算：假设不作为方向操作，仅提供基准
        print(f"RANGING_MIXED       | {len(range_df):<12} | {safe_mean(range_df, target_ret_col):<17} | N/A (Reduced Size)")

    print("\n" + "="*80 + "\n")
