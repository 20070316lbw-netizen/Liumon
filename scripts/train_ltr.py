"""
train_cn_regime_model.py
========================
A 股 Alpha Genome + 市场状态自适应模型。
引入特征：
- 动量 (Mom)
- 纯净波动残差 (Vol Res)
- 价值 (S/P)
- 状态标签 (Regime Tags: 基于 HS300 趋势)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import BASE_DIR, MODEL_DIR, CN_DIR

# 特征路径
FEATURES_PATH = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
MODEL_PATH    = os.path.join(MODEL_DIR, "cn_regime_genome.pkl")

# 核心基因特征
FEATURE_COLS = [
    "mom_60d_rank", "mom_20d_rank", 
    "vol_60d_res_rank", "sp_ratio_rank", 
    "turn_20d_rank"
]

def add_regime_tags(df):
    """
    根据日期添加市场状态标签（A 股版）
    加载由 fetch_macro_cn.py 生成的无未来函数宏观标签。
    """
    macro_path = os.path.join(CN_DIR, 'macro_regime.parquet')
    if not os.path.exists(macro_path):
        print(f"⚠️ 警告: 未找到宏观标签文件 {macro_path}，使用默认值 0")
        df["regime"] = 0
        return df

    macro_df = pd.read_parquet(macro_path)
    # 确保日期格式一致
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    df["date"] = pd.to_datetime(df["date"])
    
    # 合并宏观标签
    df = pd.merge(df, macro_df[["date", "regime"]], on="date", how="left")
    # 填充缺失值（如果是由于日期不对齐产生的）
    df["regime"] = df["regime"].ffill().fillna(0).astype(int)
    
    print(f">> 已注入宏观状态标签。样本包含 Bull/Bear 比例: {df['regime'].mean():.2%}")
    return df

def prepare_data(df):
    df = df[df["label_next_month"].notna()].copy().sort_values("date")
    
    # 填充
    for col in FEATURE_COLS:
        df[col] = df.groupby('date')[col].transform(lambda x: x.fillna(x.median())).fillna(0.5)

    # 离散化标签
    pct = df.groupby("date")["label_next_month"].transform(lambda x: x.rank(pct=True))
    df["relevance"] = pd.cut(pct, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                             labels=[0, 1, 2, 3, 4], include_lowest=True).astype(int)

    X = df[FEATURE_COLS + ["regime"]].values.astype(np.float32)
    y = df["relevance"].values.astype(np.int32)
    q_sizes = df.groupby("date", sort=True).size().values
    
    return df, X, y, q_sizes

def calculate_metrics(model, df, X, label_col='label_next_month'):
    preds = model.predict(X)
    df = df.copy()
    df["pred"] = preds
    
    ics = []
    ndcgs = []
    for d, grp in df.groupby("date"):
        if len(grp) > 20:
            ic, _ = spearmanr(grp["pred"], grp[label_col])
            ics.append(ic)
            
            y_true = [grp["relevance"].values]
            y_score = [grp["pred"].values]
            ndcg = ndcg_score(y_true, y_score, k=10)
            ndcgs.append(ndcg)
    
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic != 0 else 0
    mean_ndcg = np.mean(ndcgs)
    
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "t_stat": t_stat,
        "mean_ndcg": mean_ndcg
    }

def main():
    print("="*60)
    print("  Alpha Genome: A 股状态感应模型训练")
    print("="*60)
    
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ 缺少特征文件: {FEATURES_PATH}")
        return

    df = pd.read_parquet(FEATURES_PATH)
    df = add_regime_tags(df)
    
    # 按照用户要求：2024 为 OOS 核心评估期
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] < '2024-01-01']
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')]
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("⚠️ 数据切分失败，检查日期范围。回滚使用 80/20 切分。")
        all_dates = sorted(df["date"].unique())
        n = len(all_dates)
        train_dates = all_dates[:int(n*0.8)]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[~df["date"].isin(train_dates)]
    else:
        print(f">> 切分完成: Train ({train_df['date'].min().date()} ~ {train_df['date'].max().date()}) | Test (2024 All)")
    
    tr_df, X_train, y_train, q_train = prepare_data(train_df)
    va_df, X_test, y_test, q_test = prepare_data(test_df)
    
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "importance_type": "gain",
        "verbose": -1,
        "seed": 42
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train, group=q_train)
    
    print(f"正在训练... (样本: {len(X_train)} | 截面: {len(q_train)})")
    model = lgb.train(params, lgb_train, num_boost_round=200)
    
    # 评估
    train_metrics = calculate_metrics(model, tr_df, X_train)
    test_metrics = calculate_metrics(model, va_df, X_test)
    
    print("\n" + "="*40)
    print("  Alpha Genome: IS vs OOS Metrics")
    print("="*40)
    print(f"  [Train - IS] IC: {train_metrics['mean_ic']:.4f} | NDCG@10: {train_metrics['mean_ndcg']:.4f}")
    print(f"  [Test  - OOS] IC: {test_metrics['mean_ic']:.4f} | NDCG@10: {test_metrics['mean_ndcg']:.4f}")
    print(f"  Gap (Overfit): {train_metrics['mean_ic'] - test_metrics['mean_ic']:.4f}")
    print("="*40)
    
    # 特征重要性
    imp = pd.DataFrame({
        "feature": FEATURE_COLS + ["regime"],
        "gain": model.feature_importance(importance_type="gain")
    }).sort_values("gain", ascending=False)
    print(f"\n[Feature Importance]\n{imp.to_string(index=False)}")

    # 保存
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_COLS + ["regime"]}, f)
    print(f"[DONE] 模型已保存: {MODEL_PATH}")

if __name__ == "__main__":
    main()
