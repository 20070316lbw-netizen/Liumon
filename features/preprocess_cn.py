"""
build_enhanced_features_cn.py
=============================
A 股 Alpha Genome 特征工程强化版。
1. 动量 (Momentum)
2. 纯净波动率 (Ortho Volatility): 剥离动量后的 60d 波动残差。
3. 价值因子 (Value): 采用 1/PS (S/P) 作为核心。
4. 行业中性化排名。
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress
import statsmodels.api as sm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PRICE_DIR, FUND_DIR, CN_DIR

SAVE_PATH = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')

def calculate_stock_features(ticker_file):
    df = pd.read_parquet(ticker_file)
    if df.empty or len(df) < 250:
        return None

    ticker = os.path.basename(ticker_file).replace(".parquet", "")
    df = df.sort_index()

    # ── 1. 动量 (Momentum) ──
    df["mom_20d"] = df["close"].pct_change(20)
    df["mom_60d"] = df["close"].pct_change(60)
    df["mom_120d"] = df["close"].pct_change(120)
    # 12-1 Momentum: 过去 1 年收益剔除最近 1 个月
    df["mom_12m_minus_1m"] = df["close"].shift(22) / df["close"].shift(252) - 1

    # ── 2. 波动率 (Volatility) ──
    df["vol_60d"] = df["close"].pct_change().rolling(60).std()

    # ── 3. 价值因子 (Value) ──
    # S/P = 1 / PS (Baostock 提供的 ps 是 P/S)
    df["sp_ratio"] = 1.0 / (df["ps"] + 1e-6)
    
    # 初始化基本面列，确保即使 join 失败也不会报错
    df["roe"] = np.nan
    df["np_growth"] = np.nan

    # ── 4. 基本面因子 (暂不启用) ──
    # ...
    
    # 填充最终默认值 (已废弃)

    # ── 5. 换手率 (Liquidity) ──
    df["turn_20d"] = df["turn"].rolling(20).mean()

    df["ticker"] = ticker
    # 保留原始 close 用于标签计算
    df["raw_close"] = df["close"]
    
    return df.reset_index()

def winsorize_mad(series, n=3):
    """MAD 去极值"""
    median = series.median()
    mad = (series - median).abs().median()
    lower = median - n * 1.4826 * mad
    upper = median + n * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)

def neutralize_feature(df, feature_col, target_cols=['size_proxy']):
    """市值/行业中性化 (OLS 取残差)"""
    df = df.copy()
    mask = df[[feature_col] + target_cols].notna().all(axis=1)
    if mask.sum() < 20:
        return df[feature_col]
        
    y = df.loc[mask, feature_col]
    X = df.loc[mask, target_cols]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    df.loc[mask, f"{feature_col}_neu"] = model.resid
    # 如果回归失败或样本太少，保留原值
    df[f"{feature_col}_neu"] = df[f"{feature_col}_neu"].fillna(df[feature_col])
    return df[f"{feature_col}_neu"]

def orthogonalize_vol(group):
    """
    截面正交化：剥离动量因子对波动率的影响，提取纯净波动残差 (Vol Res)。
    """
    group = group.copy()
    # 检查必要列是否存在
    if "vol_60d" not in group.columns or "mom_60d" not in group.columns:
        group["vol_60d_res"] = group.get("vol_60d", np.nan)
        return group
        
    mask = group[["vol_60d", "mom_60d"]].notna().all(axis=1)
    if mask.sum() > 20:
        # 使用线性回归提取残差
        slope, intercept, _, _, _ = linregress(group.loc[mask, "mom_60d"], group.loc[mask, "vol_60d"])
        group.loc[mask, "vol_60d_res"] = group.loc[mask, "vol_60d"] - (intercept + slope * group.loc[mask, "mom_60d"])
    else:
        # 样本不足时保留原值
        group["vol_60d_res"] = group["vol_60d"]
    return group

def main():
    print("AlphaRanker — A 股增强特征工程 (Genome v1)")
    
    price_files = glob.glob(os.path.join(PRICE_DIR, "*.parquet"))
    print(f"Found {len(price_files)} stocks.")

    # 1. 提取所有股票的原始指标
    all_stocks = []
    for f in tqdm(price_files, desc="Processing Tickers"):
        res = calculate_stock_features(f)
        if res is not None:
            all_stocks.append(res)
    
    full_panel = pd.concat(all_stocks, ignore_index=True)
    full_panel["date"] = pd.to_datetime(full_panel["date"])
    
    # --- 新增: 多周期未来标签 (Multi-horizon Labels) ---
    print("Computing multi-horizon labels (5d, 20d, 60d, 120d)...")
    full_panel = full_panel.sort_values(["ticker", "date"])
    for d in [5, 20, 60, 120]:
        full_panel[f"label_{d}d"] = (
            full_panel.groupby("ticker")["raw_close"].shift(-d) /
            full_panel["raw_close"] - 1
        )
    
    # 2. 月末采样 (Aligning with US logic)
    print("Sampling month-end dates...")
    full_panel["ym"] = full_panel["date"].dt.to_period("M")
    
    # 显式找到每只股票每月的最大日期
    panel_me = full_panel.sort_values("date").groupby(["ticker", "ym"]).tail(1).copy()
    panel_me = panel_me.reset_index(drop=True)
    
    # 恢复实盘残余标签：月底采样面板上的 shift(-1) 能自动对接未走完的下月残余收益
    # 彻底解决：先显式排序，再根据 ticker 分组进行 shift，防止索引错位
    panel_me = panel_me.sort_values(["ticker", "date"])
    panel_me["label_next_month"] = (
        panel_me.groupby("ticker")["raw_close"].shift(-1) /
        panel_me["raw_close"] - 1
    )
    
    # 4. 截面积正交化 (Ortho Vol)
    print("Performing cross-sectional orthogonalization (Vol ~ Mom)...")
    
    # 显式使用 'date' 列进行分组
    print(f"Columns before grouping: {panel_me.columns.tolist()}")
    if "date" not in panel_me.columns:
        panel_me = panel_me.reset_index()
    
    # 彻底解决：先按日期分组，然后在内部函数中进行正交化
    temp_list = []
    for d, grp in tqdm(panel_me.groupby("date"), desc="Orthogonalizing"):
        temp_list.append(orthogonalize_vol(grp))
    panel_me = pd.concat(temp_list, ignore_index=True)
    
    # 5. 行业合并与预处理管线
    from config import IND_MAP_PATH
    if os.path.exists(IND_MAP_PATH):
        ind_df = pd.read_parquet(IND_MAP_PATH)
        panel_me = pd.merge(panel_me, ind_df, on="ticker", how="left")
        panel_me["industry_name"] = panel_me["industry_name"].fillna("Unknown")
    
    # ── 新增：市值代理变量 ──
    panel_me["size_proxy"] = np.log(panel_me["raw_close"] * panel_me["volume"] + 1e-6)
    
    rank_cols = ["mom_20d", "mom_60d", "mom_12m_minus_1m", "vol_60d_res", "sp_ratio", "turn_20d"]
    
    print("Applying Preprocessing Pipeline (MAD -> Size Neutral -> Industry De-mean)...")
    temp_list = []
    for d, grp in tqdm(panel_me.groupby("date"), desc="Neutralizing"):
        grp = grp.copy()
        for col in rank_cols:
            if col in grp.columns:
                # 1. MAD 去极值
                grp[col] = winsorize_mad(grp[col])
                # 2. 市值中性化
                grp[col] = neutralize_feature(grp, col, target_cols=['size_proxy'])
                # 3. 行业中性化 (去均值并排名)
                grp[f"{col}_rank"] = grp.groupby("industry_name")[col].rank(pct=True)
        temp_list.append(grp)
    panel_me = pd.concat(temp_list, ignore_index=True)
    
    # ── 6. 标签 Rank 化 (Rank Label for LambdaRank) ──
    print("Converting next month returns to rank labels (0-4)...")
    def to_rank_label(group, n_bins=5):
        if len(group) < n_bins:
            group["label_next_month_rank"] = 0
            return group
        # 增加 labels=False 返回整型 [0, 1, 2, 3, 4]
        group["label_next_month_rank"] = pd.qcut(group["label_next_month"], n_bins, labels=False, duplicates='drop')
        return group
    
    # 在每个日期截面内进行分箱
    temp_list = []
    for d, grp in tqdm(panel_me.groupby("date"), desc="Binning Labels"):
        temp_list.append(to_rank_label(grp))
    panel_me = pd.concat(temp_list, ignore_index=True)
    
    # 7. 保存
    panel_me = panel_me.dropna(subset=["label_next_month", "mom_60d"])
    panel_me.to_parquet(SAVE_PATH, compression="snappy")
    print(f"Enhanced features saved to {SAVE_PATH}")
    print(f"Final shape: {panel_me.shape}")

if __name__ == "__main__":
    main()
