"""
Liumon - 专业每日量化选股报告生成器
Professional Daily Quant Report Generator
"""

import os
import sys
import datetime
import pickle
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, CN_DIR, MODEL_DIR, PRICE_DIR

FEATURES_PATH = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
MACRO_PATH    = os.path.join(CN_DIR, 'macro_regime.parquet')
REPORT_DIR    = os.path.join(BASE_DIR, "reports")

# ─────────────────────────────────────────────
#  辅助函数
# ─────────────────────────────────────────────

def _regime_badge(regime: str) -> str:
    return "🟢 **牛市 (Bull)**" if regime == "Bull" else "🔴 **熊市 (Bear)**"

def _score_bar(score: float, width: int = 10) -> str:
    """将 0-1 的评分渲染为文本进度条"""
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)

def _pct(val: float) -> str:
    return f"{val * 100:.2f}%"

def _load_model_and_score(candidates: pd.DataFrame):
    """尝试用 LambdaRank 模型打分，失败时降级到静态评分"""
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
    """从最新截面数据计算市场摘要统计"""
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


# ─────────────────────────────────────────────
#  核心：报告内容生成
# ─────────────────────────────────────────────

def generate_report_content(df: pd.DataFrame, latest_macro: str) -> str:
    """
    生成专业六节 Markdown 日报。

    Parameters
    ----------
    df           : 最新截面特征数据（含 ticker / raw_close / alpha_score 等列）
    latest_macro : "Bull" | "Bear"
    """
    now        = datetime.datetime.now()
    max_date   = df["date"].max() if "date" in df.columns else pd.Timestamp.today()
    date_str   = pd.to_datetime(max_date).strftime("%Y-%m-%d")
    gen_time   = now.strftime("%Y-%m-%d %H:%M:%S")
    report_date = now.strftime("%Y年%m月%d日")

    mkt = _compute_market_summary(df)

    lines = []

    # ── 报告头 ────────────────────────────────
    lines += [
        f"# 🏦 LIUMON 每日量化选股报告",
        f"",
        f"> **交易日：** `{date_str}`　　**生成时间：** `{gen_time}`　　**市场状态：** {_regime_badge(latest_macro)}",
        f"",
        f"---",
        f"",
    ]

    # ── 一、市场总览 ───────────────────────────
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

    # ── 二、宏观体制判断 ──────────────────────
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

    # ── 三、精选持仓信号 Top 5 ────────────────
    lines += [
        f"## 三、精选持仓信号（Top 5）",
        f"",
    ]

    if "raw_close" not in df.columns or df.empty:
        lines += ["❌ 特征数据不完整，无法生成持仓信号。", ""]
    else:
        candidates = df.copy()

        # 打分
        candidates, model_note = _load_model_and_score(candidates)

        # 筛选：价格 ≤ 10 元（适当放宽以保证有候选）
        price_filter = candidates["raw_close"] <= 10.0
        filtered     = candidates[price_filter].copy()

        if filtered.empty:
            filtered = candidates.copy()  # 全宇宙回退

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

    # ── 四、因子归因快照 ──────────────────────
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

    # ── 五、风险预警 ──────────────────────────
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

    # ── 六、绩效追踪占位 ──────────────────────
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

    # ── 页脚 ──────────────────────────────────
    lines += [
        f"---",
        f"",
        f"*© {now.year} LIUMON · Alpha Genome Research Lab*  ",
        f"*生成于 {gen_time} · 仅供学习研究使用，不构成任何投资建议。*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. 读取数据
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ 特征库不存在: {FEATURES_PATH}")
        print("   请先运行「抓取数据」以生成特征文件。")
        # 生成一份空壳报告
        dummy_df = pd.DataFrame({
            "date":      [pd.Timestamp.today()],
            "ticker":    ["N/A"],
            "raw_close": [0.0],
        })
        content = generate_report_content(dummy_df, "Unknown")
    else:
        df = pd.read_parquet(FEATURES_PATH)
        latest_date = df["date"].max()
        latest_df   = df[df["date"] == latest_date].copy()

        # 2. 获取宏观状态
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
        print(f"\n✅ 报告已生成: {report_fp}")
        return

    # 空壳报告兜底
    date_str  = datetime.datetime.now().strftime("%Y%m%d")
    report_fp = os.path.join(REPORT_DIR, f"report_{date_str}.md")
    with open(report_fp, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n⚠️ 使用空壳数据生成占位报告: {report_fp}")


if __name__ == "__main__":
    main()
