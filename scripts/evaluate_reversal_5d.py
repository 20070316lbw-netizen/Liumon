import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.metrics import ndcg_score

# Add parent directory to path to allow importing from config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, MODEL_DIR, CN_DIR

# 特征路径
FEATURES_PATH = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def compute_ic(pred, future_returns):
    mask = ~np.isnan(pred) & ~np.isnan(future_returns)
    if mask.sum() < 5:
        return np.nan
    ic, _ = spearmanr(pred[mask], future_returns[mask])
    return ic

def prepare_data(df, label_col="label_20d", feature_cols=None):
    df = df[df[label_col].notna()].copy().sort_values("date")

    # 填充
    for col in feature_cols:
        if col != "regime":
            df[col] = df.groupby('date')[col].transform(lambda x: x.fillna(x.median())).fillna(0.5)

    # 离散化标签
    pct = df.groupby("date")[label_col].transform(lambda x: x.rank(pct=True))
    df["relevance"] = pd.cut(pct, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                             labels=[0, 1, 2, 3, 4], include_lowest=True).astype(int)

    X = df[feature_cols + ["regime"]].values.astype(np.float32)
    y = df["relevance"].values.astype(np.int32)
    q_sizes = df.groupby("date", sort=True).size().values

    return df, X, y, q_sizes

def add_regime_tags(df):
    macro_path = os.path.join(CN_DIR, 'macro_regime.parquet')
    if not os.path.exists(macro_path):
        df["regime"] = 0
        return df
    macro_df = pd.read_parquet(macro_path)
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    df["date"] = pd.to_datetime(df["date"])
    df = pd.merge(df, macro_df[["date", "regime"]], on="date", how="left")
    df["regime"] = df["regime"].ffill().fillna(0).astype(int)
    return df

def main():
    print("="*60)
    print("  计算 5日反转策略 IC 稳定性验证 ")
    print("="*60)

    if not os.path.exists(FEATURES_PATH):
        print(f"❌ 缺少特征文件: {FEATURES_PATH}")
        return

    df = pd.read_parquet(FEATURES_PATH)

    df = add_regime_tags(df)

    # Required columns for evaluation
    required_cols = ['label_5d', 'label_20d']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必要标签列: {missing_cols}")
        return

    FEATURE_COLS = [
        "mom_60d_rank", "mom_20d_rank",
        "vol_60d_res_rank", "sp_ratio_rank",
        "turn_20d_rank"
    ]

    # 按照用户要求：训练 20d 模型
    print(">> Training 20d model...")
    df['date'] = pd.to_datetime(df['date'])

    all_dates = sorted(df["date"].unique())
    n = len(all_dates)
    train_dates = all_dates[:int(n*0.8)]
    train_df = df[df["date"].isin(train_dates)]
    test_df = df[~df["date"].isin(train_dates)]

    target_label = "label_20d"

    tr_df, X_train, y_train, q_train = prepare_data(train_df, label_col=target_label, feature_cols=FEATURE_COLS)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 5,
        "feature_fraction": 0.8,
        "importance_type": "gain",
        "verbose": -1,
        "seed": 42
    }

    lgb_train = lgb.Dataset(X_train, label=y_train, group=q_train)
    model = lgb.train(params, lgb_train, num_boost_round=50)

    print(">> 20d model trained.")

    # Predict on Test Data
    df_eval = test_df.copy()
    for col in FEATURE_COLS:
        if col != "regime":
            df_eval[col] = df_eval.groupby('date')[col].transform(lambda x: x.fillna(x.median())).fillna(0.5)

    X_eval = df_eval[FEATURE_COLS + ["regime"]].values.astype(np.float32)
    df_eval["pred"] = model.predict(X_eval)
    df_eval["pred_reversed"] = -df_eval["pred"]

    print(">> Evaluating predictions...")

    results = []

    for d, grp in df_eval.groupby("date"):
        if len(grp) > 5:
            ic_original_5d = compute_ic(grp["pred"].values, grp["label_5d"].values)
            ic_reversed_5d = compute_ic(grp["pred_reversed"].values, grp["label_5d"].values)
            ic_reversed_20d = compute_ic(grp["pred_reversed"].values, grp["label_20d"].values)

            results.append({
                "date": d,
                "IC_original_5d": ic_original_5d,
                "IC_reversed_5d": ic_reversed_5d,
                "IC_reversed_20d": ic_reversed_20d
            })

    results_df = pd.DataFrame(results)

    # Drop rows with NaN in the columns we care about
    results_df = results_df.dropna(subset=["IC_original_5d", "IC_reversed_5d", "IC_reversed_20d"])

    csv_path = os.path.join(REPORTS_DIR, "reversal_5d_ic_series.csv")
    results_df.to_csv(csv_path, index=False)
    print(f">> IC series saved to {csv_path}")

    # Calculate statistics for IC_reversed_5d
    ic_reversed_5d_series = results_df["IC_reversed_5d"]
    mean_ic = ic_reversed_5d_series.mean()
    std_ic = ic_reversed_5d_series.std()
    ir = mean_ic / std_ic if std_ic != 0 else np.nan
    pos_ic_ratio = (ic_reversed_5d_series > 0).sum() / len(ic_reversed_5d_series) * 100
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_reversed_5d_series))) if std_ic != 0 else np.nan

    # Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df["date"], results_df["IC_reversed_5d"], label="IC_reversed_5d", color="lightblue", alpha=0.7)
        results_df.set_index("date")["IC_reversed_5d"].rolling(window=10).mean().plot(label="10d MA", color="darkblue", linewidth=2)
        plt.title('Daily IC_reversed_5d Series (with 10-day MA)')
        plt.xlabel('Date')
        plt.ylabel('Rank IC')
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid(True)
        img_path = os.path.join(REPORTS_DIR, "reversal_5d_curve.png")
        plt.savefig(img_path)
        print(f"图表已保存至 {img_path}")
    except Exception as e:
        print(f"Failed to plot: {e}")

    # Markdown Report
    report_content = f"""# 5日反转策略 IC 稳定性验证报告

## 实验逻辑
将 20d 模型的预测分数取反 (`-pred`)，计算其对 `label_5d` 的 IC，以验证反转信号是否稳定为正。
同时作为 sanity check，也计算了反转信号对 `label_20d` 的 IC。

## 统计指标分析 (IC_reversed_5d)
```text
IC_reversed_5d:
  Mean IC   : {mean_ic:.4f}
  Std IC    : {std_ic:.4f}
  IR (IC/Std): {ir:.4f}
  正IC占比   : {pos_ic_ratio:.2f}%
  t-stat    : {t_stat:.4f}
```

## 稳定性判断
根据预设的判断标准：
- **Mean IC_reversed_5d**: {"✅ > 0.02" if mean_ic > 0.02 else "❌ <= 0.02"} ({mean_ic:.4f})
- **正IC占比**: {"✅ > 55%" if pos_ic_ratio > 55 else "❌ <= 55%"} ({pos_ic_ratio:.2f}%)
- **t-stat**: {"✅ > 2.0" if t_stat > 2.0 else "❌ <= 2.0"} ({t_stat:.4f})

**结论**:
{ "反转策略值得进一步开发" if mean_ic > 0.02 and pos_ic_ratio > 55 and t_stat > 2.0 else "负IC可能只是噪声，不可利用" }

"""
    report_path = os.path.join(REPORTS_DIR, "reversal_5d_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"报告已保存至 {report_path}")

if __name__ == "__main__":
    main()
