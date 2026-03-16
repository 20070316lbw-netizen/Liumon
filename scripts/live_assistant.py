import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pickle
from config import CN_DIR, MODEL_DIR
from alpharanker.configs.cap_aware_weights import get_weights

FEATURES_PATH = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
MACRO_PATH = os.path.join(CN_DIR, 'macro_regime.parquet')
BUDGET = 500.0  # 500 RMB

def generate_signals(df, latest_macro, use_model=True):
    print(f"\n[500元小助手] 正在基于 {df['date'].max().date()} 数据生成信号...")
    print(f"当前市场状态: {latest_macro}")
    
    # 0. 准备 regime 编码 (用于模型输入)
    df['regime'] = 1 if latest_macro == "Bull" else 0
    
    # 1. 价格过滤 (由于 100 股起购，本金 500 元意味着单价必须 < 5.0)
    price_limit = 4.80 
    candidates = df[df['raw_close'] <= price_limit].copy()
    
    if candidates.empty:
        print("❌ 当前市场无单价 < 4.8 元的标的。")
        return
    
    # 2. 获取权重 (静态版本备用)
    weights = get_weights("ZZ500", horizon_days=20)
    
    # 3. 计算评分
    # 静态评分 (始终计算用于对比)
    candidates['static_score'] = 0
    for f, w in weights.items():
        if f in candidates.columns:
            candidates['static_score'] += candidates[f] * w
            
    # 模型评分
    if use_model:
        model_path = os.path.join(MODEL_DIR, "cn_regime_genome.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)
            model = model_obj["model"]
            feature_cols = model_obj["features"]
            
            X = candidates[feature_cols].fillna(0.5).values.astype(np.float32)
            candidates['alpha_score'] = model.predict(X)
        else:
            print("⚠️ 未找到 LTR 模型，回滚至静态评分。")
            candidates['alpha_score'] = candidates['static_score']
    else:
        candidates['alpha_score'] = candidates['static_score']
            
    # 4. 排名并选择 Top 2 (基于 alpha_score)
    top_picks = candidates.sort_values('alpha_score', ascending=False).head(2)
    
    print("\n" + "-"*40)
    print("推荐持仓信号 (500元本金上限)")
    print("-"*40)
    
    if top_picks.empty:
        print("未找到符合条件的标的。")
        return

    for _, row in top_picks.iterrows():
        cost_100 = row['raw_close'] * 100
        print(f"股票: {row['ticker']} | 代码: {row['ticker']}")
        print(f"当前价格: {row['raw_close']:.2f} 元")
        print(f"买入单位: 100 股")
        print(f"预计成本: {cost_100:.1f} 元 (+ 佣金)")
        print(f"LambdaRank评分: {row['alpha_score']:.4f}")
        print(f"静态权重评分:  {row['static_score']:.4f}")
        print("-" * 20)

def main():
    if not os.path.exists(FEATURES_PATH):
        print("❌ 特征库不存在。")
        return

    df = pd.read_parquet(FEATURES_PATH)
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date].copy()
    
    regime = "Unknown"
    if os.path.exists(MACRO_PATH):
        macro_df = pd.read_parquet(MACRO_PATH)
        m_match = macro_df[macro_df['date'] <= latest_date].tail(1)
        if not m_match.empty:
            regime = "Bull" if m_match['regime'].iloc[0] == 1 else "Bear"

    generate_signals(latest_df, regime)

if __name__ == "__main__":
    main()
