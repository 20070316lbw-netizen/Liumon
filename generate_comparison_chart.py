import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from config import BASE_DIR
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

def main():
    FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'cn', 'cn_features_enhanced.parquet')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cn_regime_genome.pkl')

    if not os.path.exists(FEATURES_PATH) or not os.path.exists(MODEL_PATH):
        print("Features or model not found.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PATH)
    from scripts.train import add_regime_tags
    df = add_regime_tags(df)
    with open(MODEL_PATH, "rb") as f:
        model_obj = pickle.load(f)

    model = model_obj["model"]
    features = model_obj["features"]

    df = df.dropna(subset=features).copy()
    if df.empty:
        print("Data is empty after dropping missing features.")
        sys.exit(1)

    X = df[features].values.astype(np.float32)
    df["pred"] = model.predict(X)

    latest_date = df["date"].max()
    print(f"Latest prediction date: {latest_date}")

    valid_hist = df.dropna(subset=["label_next_month"]).copy()

    if valid_hist.empty:
        print("No historical data with valid future labels found.")
        sys.exit(1)

    all_valid_dates = sorted(valid_hist["date"].unique())
    if len(all_valid_dates) < 2:
        print("Not enough historical data points to select the second-to-last date.")
        sys.exit(1)

    hist_latest_date = all_valid_dates[-2]
    print(f"Latest date with available real returns: {hist_latest_date}")

    cross_section = valid_hist[valid_hist["date"] == hist_latest_date].copy()

    top_stocks = cross_section.sort_values("pred", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_stocks, x="ticker", y="label_next_month", color="steelblue")
    plt.title(f"Actual Next Month Return of Top 10 Predicted Stocks ({hist_latest_date.date()})")
    plt.ylabel("Actual Return")
    plt.xlabel("Ticker")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"top10_actual_returns_{hist_latest_date.strftime('%Y%m%d')}.png"))
    plt.close()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=cross_section, x="pred", y="label_next_month", alpha=0.6)

    ic, pval = spearmanr(cross_section["pred"], cross_section["label_next_month"])
    plt.title(f"Prediction vs Actual Returns ({hist_latest_date.date()})\nRank IC: {ic:.4f} (p={pval:.4f})")
    plt.xlabel("Predicted Alpha Score")
    plt.ylabel("Actual Next Month Return")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"pred_vs_actual_{hist_latest_date.strftime('%Y%m%d')}.png"))
    plt.close()

    print(f"Charts generated successfully. Rank IC for {hist_latest_date.date()}: {ic:.4f}")

if __name__ == "__main__":
    main()
