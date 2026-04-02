#!/usr/bin/env python3
"""
Liumon - 自动抓取与格式化报告生成一体化作业
Automated Fetch and Format Report Generation Job
"""

import os
import sys
import datetime
import subprocess
import pickle
import pandas as pd
import numpy as np

# 确保项目根目录在 PYTHONPATH 中
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from config import BASE_DIR, CN_DIR, MODEL_DIR, PRICE_DIR

FEATURES_PATH = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
MACRO_PATH    = os.path.join(CN_DIR, 'macro_regime.parquet')
REPORT_DIR    = os.path.join(BASE_DIR, "reports")

# ─────────────────────────────────────────────
#  第一部分：执行数据获取、特征处理与模型打分
# ─────────────────────────────────────────────

def run_step(name, script_path):
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 开始执行: {name}")
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = BASE_DIR
        subprocess.check_call([sys.executable, script_path], env=env)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 成功完成: {name}")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 失败: {name} (退出码: {e.returncode})")
        sys.exit(e.returncode)

def execute_pipeline():
    print("=" * 60)
    print(f"  LIUMON 每日自动化数据抓取作业开始")
    print(f"  开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    fetch_cn      = os.path.join(BASE_DIR, "liumon", "data", "data_fetch_cn.py")
    fetch_macro   = os.path.join(BASE_DIR, "liumon", "data", "data_fetch_macro.py")
    preprocess    = os.path.join(BASE_DIR, "features", "preprocess_cn.py")
    train_predict = os.path.join(BASE_DIR, "scripts", "train.py")

    run_step("Data Ingestion (A-Share) - 抓取A股行情数据", fetch_cn)
    run_step("Market Regime Sensing - 抓取宏观环境数据", fetch_macro)
    run_step("Feature Engineering & Cleaning - 提取整合特征数据", preprocess)
    run_step("Model Training & Latest Prediction - 模型计算", train_predict)

# ─────────────────────────────────────────────
#  第二部分：生成严谨格式报告的内部逻辑
# ─────────────────────────────────────────────

def _regime_badge(regime: str) -> str:
    return "🟢 **牛市 (Bull)**" if regime == "Bull" else "🔴 **熊市 (Bear)**"

def _score_bar(score: float, width: int = 10) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)

def _load_model_and_score(candidates: pd.DataFrame):
    model_path = os.path.join(MODEL_DIR, "cn_regime_genome.pkl")
    use_model = False
    model_note = "⚠️ 未找到 LTR 模型，使用静态因子加权评分"

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)
            model      = model_obj["model"]
            feat_cols  = model_obj["features"]
            X = candidates[feat_cols].fillna(0.5).values.astype(np.float32)
            candidates["alpha_score"] = model.predict(X)
            use_model  = True
            model_note = f"✅ LambdaRank 模型 (`cn_regime_genome.pkl`)，特征数: {len(feat_cols)}"
        except Exception as e:
            model_note = f"⚠️ 模型加载失败 ({e})，降级为静态评分"

    if not use_model:
        factor_weights = {
            "mom_20d_rank":     0.25,
            "vol_60d_res_rank": 0.20,
            "sp_ratio_rank":    0.20,
            "turn_20d_rank":    0.15,
            "mom_60d_rank":     0.20,
        }
        candidates["alpha_score"] = 0.0
        for f, w in factor_weights.items():
            if f in candidates.columns:
                candidates["alpha_score"] += candidates[f] * w
        model_note += f"\n> 权重分配：{factor_weights}"

    return candidates, model_note

def _compute_market_summary(df: pd.DataFrame) -> dict:
    summary = {}
    summary["total_stocks"]  = len(df)
    if "raw_close" in df.columns:
        prices = df["raw_close"].dropna()
        summary["price_mean"]   = prices.mean()
        summary["price_median"] = prices.median()
        summary["low_price_cnt"] = (prices <= 5.0).sum()
    else:
        summary["price_mean"] = summary["price_median"] = summary["low_price_cnt"] = "N/A"
    for col in ["pe", "pb"]:
        if col in df.columns:
            summary[col + "_median"] = df[col].replace(0, np.nan).median()
        else:
            summary[col + "_median"] = "N/A"
    return summary

def generate_report_content(df: pd.DataFrame, latest_macro: str) -> str:
    now        = datetime.datetime.now()
    if not df.empty and "date" in df.columns and not pd.isna(df["date"].max()):
        max_date = df["date"].max()
    else:
        max_date = pd.Timestamp.today()
    date_str   = pd.to_datetime(max_date).strftime("%Y-%m-%d")
    gen_time   = now.strftime("%Y-%m-%d %H:%M:%S")

    mkt = _compute_market_summary(df)
    lines = []

    # 报告头
    lines += [
        f"# 🏦 LIUMON 每日量化选股报告",
        f"",
        f"> **交易日：** `{date_str}`　　**生成时间：** `{gen_time}`　　**市场状态：** {_regime_badge(latest_macro)}",
        f"",
        f"---",
        f"",
    ]

    # 一、市场总览
    lines += [
        f"## 一、市场总览",
        f"",
        f"| 指标 | 数值 |",
        f"|:---|:---|",
        f"| 📅 数据日期 | `{date_str}` |",
        f"| 🏛️ 宏观体制 | {_regime_badge(latest_macro)} |",
        f"| 📊 覆盖股票数 | **{mkt['total_stocks']}** 只 |",
        f"| 💰 低价股（≤5元）数量 | **{mkt['low_price_cnt']}** 只 |",
    ]
    if isinstance(mkt["pe_median"], float):
        lines.append(f"| 📈 全市场 PE 中位数 | **{mkt['pe_median']:.1f}x** |")
    if isinstance(mkt["pb_median"], float):
        lines.append(f"| 📚 全市场 PB 中位数 | **{mkt['pb_median']:.2f}x** |")
    lines += ["", ""]

    # 二、宏观体制判断
    if latest_macro == "Bull":
        regime_detail = (
            "HS300 收盘价 **高于** 250日均线，当前处于**多头趋势区间**。\n\n"
            "建议策略：**积极做多**，可适当放大仓位，关注成长/动量因子。"
        )
        risk_level = "🟡 中等"
    else:
        regime_detail = (
            "HS300 收盘价 **低于** 250日均线，当前处于**空头趋势区间**。\n\n"
            "建议策略：**防御为主**，降低单票仓位，优先配置低波动/低估值标的。"
        )
        risk_level = "🔴 较高"

    lines += [
        f"## 二、宏观体制判断",
        f"",
        regime_detail,
        f"",
        f"| 信号来源 | 判断逻辑 | 当前状态 |",
        f"|:---|:---|:---|",
        f"| HS300 MA250 趋势 | 收盘价 vs. 250日均线 | {_regime_badge(latest_macro)} |",
        f"| 整体风险等级 | — | {risk_level} |",
        f"",
        f"",
    ]

    # 三、精选持仓信号 Top 5
    lines += [f"## 三、精选持仓信号（Top 5）", f""]

    if "raw_close" not in df.columns or df.empty:
        lines += ["❌ 特征数据不完整，无法生成持仓信号。", ""]
    else:
        candidates = df.copy()
        candidates, model_note = _load_model_and_score(candidates)

        price_filter = candidates["raw_close"] <= 10.0
        filtered     = candidates[price_filter].copy()

        if filtered.empty:
            filtered = candidates.copy()

        top5 = filtered.sort_values("alpha_score", ascending=False).head(5)

        lines += [
            f"> **评分引擎：** {model_note}",
            f"",
            f"| 排名 | 股票代码 | 当前价格 | 建议仓位 | Alpha 评分 | 信号强度 |",
            f"|:---:|:---|---:|:---:|---:|:---|",
        ]

        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            ticker     = row.get("ticker", "N/A")
            price      = row.get("raw_close", 0.0)
            alpha      = row.get("alpha_score", 0.0)
            units      = max(100, int(500 / price / 100) * 100) if price > 0 else 100
            cost       = price * units
            signal_bar = _score_bar(min(max(alpha, 0), 1))
            lines.append(
                f"| {rank} | `{ticker}` | ¥{price:.2f} | {units}股 (≈¥{cost:.0f}) | `{alpha:.4f}` | {signal_bar} |"
            )

        lines += [
            f"",
            f"> ⚠️ 以上持仓仅为 Alpha 模型输出，**不构成投资建议**。实际交易请结合基本面及风险承受能力。",
            f"",
            f"",
        ]

    # 四、因子归因快照
    factor_cols = {
        "mom_20d_rank":     "20日动量(排名)",
        "mom_60d_rank":     "60日动量(排名)",
        "vol_60d_res_rank": "特异波动率(排名)",
        "sp_ratio_rank":    "销售价格比(排名)",
        "turn_20d_rank":    "20日换手率(排名)",
        "pe":               "PE 估值",
        "pb":               "PB 估值",
    }

    lines += [
        f"## 四、因子归因快照",
        f"",
        f"> 以下为全宇宙因子截面统计，反映当日各因子分布。",
        f"",
        f"| 因子 | 均值 | 中位数 | 标准差 |",
        f"|:---|---:|---:|---:|",
    ]

    for col, label in factor_cols.items():
        if col in df.columns:
            vals = df[col].replace(0, np.nan).dropna()
            if len(vals) > 0:
                lines.append(
                    f"| {label} | {vals.mean():.4f} | {vals.median():.4f} | {vals.std():.4f} |"
                )
    lines += ["", ""]

    # 五、风险预警
    warnings = []
    if latest_macro == "Bear":
        warnings.append("🔴 **市场处于熊市区间**：建议降低整体仓位至 50% 以下")
    if isinstance(mkt["pe_median"], float) and mkt["pe_median"] > 30:
        warnings.append(f"🟡 **估值偏高预警**：全市场 PE 中位数 {mkt['pe_median']:.1f}x，高于历史合理区间（15-25x）")
    if isinstance(mkt["low_price_cnt"], int) and mkt["low_price_cnt"] < 20:
        warnings.append("🟡 **低价标的稀少**：低价股数量不足 20 只，策略可选空间收窄")

    lines += [f"## 五、风险预警", f""]
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- ✅ 当前无重大风险预警")
    lines += ["", ""]

    # 六、绩效追踪占位
    lines += [
        f"## 六、策略绩效追踪",
        f"",
        f"> 本节在积累足够历史报告后自动填充，当前为占位区。",
        f"",
        f"| 指标 | 近 20 日 | 近 60 日 |",
        f"|:---|:---:|:---:|",
        f"| 平均 IC | — | — |",
        f"| 多头组合超额收益 | — | — |",
        f"| 最大回撤 | — | — |",
        f"| 夏普比率 | — | — |",
        f"",
        f"",
    ]

    # 页脚
    lines += [
        f"---",
        f"",
        f"*© {now.year} LIUMON · Alpha Genome Research Lab*  ",
        f"*生成于 {gen_time} · 仅供学习研究使用，不构成任何投资建议。*",
    ]

    return "\n".join(lines)

def generate_and_save_report():
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not os.path.exists(FEATURES_PATH):
        print(f"❌ 特征库不存在: {FEATURES_PATH}")
        dummy_df = pd.DataFrame(columns=["date", "ticker", "raw_close"])
        content = generate_report_content(dummy_df, "Unknown")
        date_str  = datetime.datetime.now().strftime("%Y%m%d")
        report_fp = os.path.join(REPORT_DIR, f"report_{date_str}.md")
    else:
        df = pd.read_parquet(FEATURES_PATH)
        latest_date = df["date"].max()
        if pd.isna(latest_date):
            latest_date = pd.Timestamp.today()
            latest_df = pd.DataFrame(columns=df.columns)
        else:
            latest_df   = df[df["date"] == latest_date].copy()

        regime = "Unknown"
        if os.path.exists(MACRO_PATH):
            macro_df = pd.read_parquet(MACRO_PATH)
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            m_match = macro_df[macro_df["date"] <= pd.to_datetime(latest_date)].tail(1)
            if not m_match.empty:
                regime = "Bull" if m_match["regime"].iloc[0] == 1 else "Bear"

        content = generate_report_content(latest_df, regime)
        date_str = pd.to_datetime(latest_date).strftime("%Y%m%d")
        report_fp = os.path.join(REPORT_DIR, f"report_{date_str}.md")

    with open(report_fp, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ 严谨报告已生成并存入文件夹: {report_fp}")


def main():
    # 抓取数据并运行预测管线
    execute_pipeline()

    # 格式化并生成报告
    generate_and_save_report()

if __name__ == "__main__":
    main()
