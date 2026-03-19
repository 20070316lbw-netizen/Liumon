import os
import sys
import subprocess
import datetime
import pandas as pd
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, CN_DIR, MODEL_DIR

def get_weights(ticker, horizon_days):
    # Dummy implementation for missing dependency
    # Based on scripts/assistant.py this returns a dictionary of weights
    return {"mom_20d_rank": 0.2, "vol_60d_res_rank": 0.2, "sp_ratio_rank": 0.2, "turn_20d_rank": 0.2, "mom_60d_rank": 0.2}

FEATURES_PATH = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
MACRO_PATH = os.path.join(CN_DIR, 'macro_regime.parquet')
REPORT_DIR = os.path.join(BASE_DIR, "reports")

def run_pipeline():
    print(">>> 启动数据抓取与预测流水线 (执行 scripts/live.py)...")
    live_script = os.path.join(BASE_DIR, "scripts", "live.py")
    try:
        subprocess.check_call([sys.executable, live_script])
        print(">>> 流水线执行成功。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 流水线执行失败，退出代码: {e.returncode}")
        return False

def generate_report_content(df, latest_macro, use_model=True):
    max_date = df['date'].max()
    if pd.isna(max_date) or isinstance(max_date, float):
        max_date = pd.Timestamp.today()
    date_str = pd.to_datetime(max_date).strftime('%Y-%m-%d')
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"  LIUMON 每日选股报告")
    report_lines.append(f"  日期: {date_str}")
    report_lines.append(f"  生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"【市场环境】")
    report_lines.append(f"当前市场状态: {latest_macro}")
    report_lines.append("")

    df['regime'] = 1 if latest_macro == "Bull" else 0
    price_limit = 4.80
    candidates = df[df['raw_close'] <= price_limit].copy()

    if candidates.empty:
        report_lines.append("❌ 当前市场无单价 < 4.8 元的标的。")
        return "\n".join(report_lines)

    weights = get_weights("ZZ500", horizon_days=20)
    candidates['static_score'] = 0
    for f, w in weights.items():
        if f in candidates.columns:
            candidates['static_score'] += candidates[f] * w

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
            report_lines.append("⚠️ 未找到 LTR 模型，回滚至静态评分。")
            candidates['alpha_score'] = candidates['static_score']
    else:
        candidates['alpha_score'] = candidates['static_score']

    top_picks = candidates.sort_values('alpha_score', ascending=False).head(2)

    report_lines.append("【推荐持仓信号 (500元本金上限)】")
    report_lines.append("-" * 40)

    if top_picks.empty:
        report_lines.append("未找到符合条件的标的。")
    else:
        for _, row in top_picks.iterrows():
            cost_100 = row['raw_close'] * 100
            report_lines.append(f"股票代码: {row['ticker']}")
            report_lines.append(f"当前价格: {row['raw_close']:.2f} 元")
            report_lines.append(f"买入单位: 100 股")
            report_lines.append(f"预计成本: {cost_100:.1f} 元 (+ 佣金)")
            report_lines.append(f"LambdaRank评分: {row['alpha_score']:.4f}")
            report_lines.append(f"静态权重评分:  {row['static_score']:.4f}")
            report_lines.append("-" * 40)

    report_lines.append("\n*免责声明: 本报告仅供学习和研究使用，不构成任何投资建议，请谨慎使用。*")
    return "\n".join(report_lines)

def main():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # 1. 运行流水线更新数据
    if not run_pipeline():
        sys.exit(1)

    # 2. 读取数据
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ 特征库不存在: {FEATURES_PATH}")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PATH)
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date].copy()

    # 3. 获取宏观状态
    regime = "Unknown"
    if os.path.exists(MACRO_PATH):
        macro_df = pd.read_parquet(MACRO_PATH)
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        if pd.isna(latest_date):
            # Fallback if features data is empty and date is NaT
            m_match = macro_df.tail(1)
            latest_date = pd.Timestamp.today()
        else:
            m_match = macro_df[macro_df['date'] <= pd.to_datetime(latest_date)].tail(1)
        if not m_match.empty:
            regime = "Bull" if m_match['regime'].iloc[0] == 1 else "Bear"

    if pd.isna(latest_date):
        latest_date = pd.Timestamp.today()
        latest_df = pd.DataFrame({'date': [latest_date], 'raw_close': [0.0], 'ticker': ['NONE']})

    # 4. 生成报告内容
    report_content = generate_report_content(latest_df, regime)

    # 5. 保存报告
    date_str = latest_date.strftime('%Y%m%d')
    report_filename = f"report_{date_str}.md"
    report_filepath = os.path.join(REPORT_DIR, report_filename)

    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n✅ 报告已成功生成并保存至: {report_filepath}")

if __name__ == "__main__":
    main()
