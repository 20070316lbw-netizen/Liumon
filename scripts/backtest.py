"""
Liumon - A 股量化回测框架 v2.0
基于本地 Parquet 数据，无需网络连接
核心指标：年化收益、夏普比率、最大回撤、IC / ICIR、换手率
"""

import os
import sys
import datetime
import json
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, CN_DIR, PRICE_DIR, MODEL_DIR

BACKTEST_OUTPUT_DIR = os.path.join(BASE_DIR, "backtests")
MACRO_PATH          = os.path.join(CN_DIR, "macro_regime.parquet")

# ─────────────────────────────────────────────────────────────────
#  默认配置
# ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "start_date":     "2020-01-01",
    "end_date":       "2024-12-31",
    "rebalance_freq": "M",          # 'M' 月度换仓 / 'W' 周度换仓
    "top_n":          20,           # 每期多头组合持股数
    "commission":     0.001,        # 单边手续费（千一）
    "slippage":       0.002,        # 滑点估计（双边）
    "benchmark_code": "sh.000300",  # 基准指数（HS300，仅用于绩效对比）
    "risk_free_rate": 0.025,        # 无风险利率（年化）
}

# ─────────────────────────────────────────────────────────────────
#  数据加载
# ─────────────────────────────────────────────────────────────────

def load_price_panel(start_date: str, end_date: str) -> pd.DataFrame:
    """从本地 parquet 文件加载全部股票价格面板"""
    print(">> [1/4] 加载本地价格数据...")

    if not os.path.isdir(PRICE_DIR):
        raise FileNotFoundError(f"价格数据目录不存在: {PRICE_DIR}")

    pieces = []
    files  = [f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"价格目录中无 parquet 文件: {PRICE_DIR}")

    for fname in files:
        fpath = os.path.join(PRICE_DIR, fname)
        try:
            df = pd.read_parquet(fpath)
            # 确保 ticker 列
            ticker = fname.replace(".parquet", "")
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            pieces.append(df)
        except Exception as e:
            print(f"   ⚠️  跳过 {fname}: {e}")

    if not pieces:
        raise ValueError("没有可用的价格数据")

    panel = pd.concat(pieces)
    panel.index = pd.to_datetime(panel.index)
    panel       = panel[(panel.index >= start_date) & (panel.index <= end_date)]
    panel.sort_index(inplace=True)
    print(f"   加载完成：{panel['ticker'].nunique()} 只股票，"
          f"{panel.index.min().date()} → {panel.index.max().date()}")
    return panel


def load_features(start_date: str, end_date: str) -> pd.DataFrame | None:
    """尝试加载特征数据（可选）"""
    feat_path = os.path.join(CN_DIR, "cn_features_enhanced.parquet")
    if not os.path.exists(feat_path):
        return None
    df = pd.read_parquet(feat_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    return df


def load_macro(start_date: str, end_date: str) -> pd.DataFrame | None:
    """加载宏观体制数据（可选）"""
    if not os.path.exists(MACRO_PATH):
        return None
    df = pd.read_parquet(MACRO_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    return df

# ─────────────────────────────────────────────────────────────────
#  Alpha 评分
# ─────────────────────────────────────────────────────────────────

def score_stocks(date: pd.Timestamp, price_panel: pd.DataFrame,
                 features: pd.DataFrame | None) -> pd.Series:
    """
    为给定日期的所有股票打 Alpha 分。
    优先使用特征数据（LambdaRank 或静态因子），降级使用价格动量。
    返回 Series: ticker -> score
    """
    # ── 优先：特征数据 ──
    if features is not None:
        snap = features[features["date"] == date].copy()
        if not snap.empty:
            factor_weights = {
                "mom_20d_rank":     0.25,
                "vol_60d_res_rank": -0.15,   # 低波动更好
                "sp_ratio_rank":    0.20,
                "turn_20d_rank":    0.15,
                "mom_60d_rank":     0.25,
            }
            snap["score"] = 0.0
            for f, w in factor_weights.items():
                if f in snap.columns:
                    snap["score"] += snap[f].fillna(0.5) * w
            return snap.set_index("ticker")["score"]

    # ── 降级：纯价格 20 日动量 ──
    window_start = date - pd.Timedelta(days=30)
    hist = price_panel[(price_panel.index >= window_start) &
                       (price_panel.index <= date)].copy()
    scores = {}
    for ticker, grp in hist.groupby("ticker"):
        if len(grp) >= 5 and "close" in grp.columns:
            p_old  = grp["close"].iloc[0]
            p_new  = grp["close"].iloc[-1]
            if p_old > 0:
                scores[ticker] = float(p_new / p_old - 1)
    return pd.Series(scores)


# ─────────────────────────────────────────────────────────────────
#  组合构建与换仓
# ─────────────────────────────────────────────────────────────────

def build_portfolio(date: pd.Timestamp, price_panel: pd.DataFrame,
                    features: pd.DataFrame | None, cfg: dict) -> list[str]:
    """返回当期持仓 ticker 列表（等权）"""
    scores  = score_stocks(date, price_panel, features)
    if scores.empty:
        return []

    # 确认当日有价格的股票（排除停牌）
    day_prices = price_panel[price_panel.index == date]
    tradable   = set(day_prices["ticker"].unique())
    scores     = scores[scores.index.isin(tradable)]

    top = scores.nlargest(cfg["top_n"]).index.tolist()
    return top


# ─────────────────────────────────────────────────────────────────
#  绩效指标计算
# ─────────────────────────────────────────────────────────────────

def compute_metrics(nav: pd.Series, rf: float = 0.025) -> dict:
    """计算主要绩效指标"""
    if nav.empty or len(nav) < 2:
        return {}

    daily_ret  = nav.pct_change().dropna()
    total_days = len(daily_ret)
    years      = total_days / 252

    total_ret      = float(nav.iloc[-1] / nav.iloc[0] - 1)
    annualized_ret = float((1 + total_ret) ** (1 / max(years, 0.01)) - 1)

    excess_ret    = daily_ret - rf / 252
    sharpe        = float(excess_ret.mean() / excess_ret.std() * np.sqrt(252)) if excess_ret.std() > 0 else 0.0

    downside_std  = daily_ret[daily_ret < 0].std()
    sortino       = float(daily_ret.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    rolling_max   = nav.cummax()
    drawdown      = (nav - rolling_max) / rolling_max
    max_drawdown  = float(drawdown.min())

    calmar        = float(annualized_ret / abs(max_drawdown)) if max_drawdown != 0 else 0.0
    volatility    = float(daily_ret.std() * np.sqrt(252))

    return {
        "total_return":      round(total_ret,      4),
        "annualized_return": round(annualized_ret, 4),
        "sharpe_ratio":      round(sharpe,         4),
        "sortino_ratio":     round(sortino,         4),
        "max_drawdown":      round(max_drawdown,   4),
        "calmar_ratio":      round(calmar,         4),
        "annualized_vol":    round(volatility,     4),
        "total_days":        total_days,
    }


def compute_ic_series(rebalance_log: list[dict]) -> dict:
    """从换仓日志计算 IC / ICIR"""
    if not rebalance_log:
        return {"ic_mean": "N/A", "ic_std": "N/A", "icir": "N/A"}

    ics = [r.get("ic", np.nan) for r in rebalance_log if not np.isnan(r.get("ic", np.nan))]
    if not ics:
        return {"ic_mean": "N/A", "ic_std": "N/A", "icir": "N/A"}

    ic_arr = np.array(ics)
    ic_std = float(ic_arr.std()) if len(ic_arr) > 1 else 0.0
    icir   = float(ic_arr.mean() / ic_std) if ic_std > 0 else 0.0
    return {
        "ic_mean": round(float(ic_arr.mean()), 4),
        "ic_std":  round(ic_std, 4),
        "icir":    round(icir, 4),
    }


# ─────────────────────────────────────────────────────────────────
#  主回测循环
# ─────────────────────────────────────────────────────────────────

def run_backtest(cfg: dict = None, dry_run: bool = False) -> dict:
    if cfg is None:
        cfg = DEFAULT_CONFIG

    start_date = cfg["start_date"]
    end_date   = cfg["end_date"]

    print("=" * 60)
    print("  Liumon A-Share Backtest Engine v2.0")
    print(f"  {start_date} → {end_date}")
    print(f"  换仓频率: {cfg['rebalance_freq']} | 持股数: {cfg['top_n']}")
    print("=" * 60)

    # 1. 加载数据
    try:
        price_panel = load_price_panel(start_date, end_date)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   请先运行「抓取数据」以下载价格数据。")
        return {}

    print(">> [2/4] 加载特征与宏观数据...")
    features = load_features(start_date, end_date)
    macro    = load_macro(start_date, end_date)

    if features is not None:
        print(f"   特征数据：{len(features)} 行")
    else:
        print("   ⚠️  无特征数据，降级为纯价格动量策略")

    if dry_run:
        print("\n✅ [Dry Run] 数据加载正常，回测引擎就绪，跳过实际回测。")
        return {"status": "dry_run_ok"}

    # 2. 生成换仓日期序列
    all_dates     = price_panel.index.unique().sort_values()
    rebal_dates   = pd.date_range(start=all_dates[0], end=all_dates[-1],
                                  freq=cfg["rebalance_freq"])
    # 对齐到实际交易日
    rebal_dates   = [d for d in rebal_dates if d in all_dates or
                     all_dates[all_dates >= d][0:1].size > 0]
    rebal_dates   = [all_dates[all_dates >= d][0] if d not in all_dates else d
                     for d in rebal_dates]

    print(f">> [3/4] 回测中（共 {len(rebal_dates)} 个换仓期）...")

    nav           = pd.Series(dtype=float)  # 组合净值
    benchmark_nav = pd.Series(dtype=float)  # 基准净值（HS300 等权代理）
    portfolio     = []
    rebalance_log = []

    portfolio_value = 1.0
    bench_value     = 1.0
    prev_date       = None

    for i, reb_date in enumerate(rebal_dates):
        # 当前持仓到下一换仓日的价格变动
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]

        period_dates = all_dates[(all_dates >= reb_date) & (all_dates <= next_date)]
        if len(period_dates) < 2:
            continue

        # 构建新持仓
        new_portfolio = build_portfolio(reb_date, price_panel, features, cfg)
        if not new_portfolio:
            continue

        # 换手率（简化）
        if portfolio:
            overlap    = len(set(portfolio) & set(new_portfolio))
            turnover   = 1 - overlap / max(len(portfolio), len(new_portfolio))
        else:
            turnover   = 1.0

        # 扣除交易成本
        cost             = turnover * (cfg["commission"] + cfg["slippage"] / 2)
        portfolio_value *= (1 - cost)

        # 计算期间收益（等权）
        period_returns = []
        for ticker in new_portfolio:
            t_data = price_panel[(price_panel["ticker"] == ticker) &
                                 (price_panel.index.isin(period_dates))]
            if len(t_data) >= 2 and "close" in t_data.columns:
                r = float(t_data["close"].iloc[-1] / t_data["close"].iloc[0] - 1)
                period_returns.append(r)

        if period_returns:
            port_ret        = np.mean(period_returns)
            portfolio_value *= (1 + port_ret)

        # IC：评分排名 vs 实际收益排名的相关系数
        ic = np.nan
        if features is not None:
            scores_series = score_stocks(reb_date, price_panel, features)
            ret_series    = pd.Series({t: r for t, r in
                            zip(new_portfolio, period_returns)})
            if len(scores_series) > 1 and len(ret_series) > 1:
                common = scores_series.index.intersection(ret_series.index)
                if len(common) > 1:
                    ic = float(scores_series[common].rank().corr(ret_series[common].rank()))

        rebalance_log.append({
            "date":      str(reb_date.date()),
            "portfolio": new_portfolio,
            "turnover":  round(turnover, 4),
            "period_ret": round(port_ret if period_returns else 0, 4),
            "ic":        ic,
        })

        # 逐日写 NAV（简化：线性插值）
        for d in period_dates[1:]:
            nav.loc[d] = portfolio_value

        portfolio = new_portfolio
        print(f"   {reb_date.date()} → 持仓 {len(new_portfolio)} 只，"
              f"换手率 {turnover:.1%}，期间收益 "
              f"{(period_returns and f'{np.mean(period_returns):.2%}') or 'N/A'}")

    print(">> [4/4] 计算绩效指标...")
    metrics    = compute_metrics(nav.sort_index(), rf=cfg["risk_free_rate"])
    ic_metrics = compute_ic_series(rebalance_log)
    results    = {**metrics, **ic_metrics}

    # 换手率统计
    if rebalance_log:
        turnovers             = [r["turnover"] for r in rebalance_log]
        results["avg_turnover"] = round(float(np.mean(turnovers)), 4)

    results["config"]    = cfg
    results["nav"]       = nav.sort_index().to_dict()
    results["rebalance"] = rebalance_log

    return results


# ─────────────────────────────────────────────────────────────────
#  报告生成
# ─────────────────────────────────────────────────────────────────

def _pct(val) -> str:
    if isinstance(val, float):
        return f"{val * 100:.2f}%"
    return str(val)

def _f(val, decimals: int = 4) -> str:
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def generate_backtest_report(results: dict, cfg: dict) -> str:
    now      = datetime.datetime.now()
    gen_time = now.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# 📊 LIUMON 回测报告",
        f"",
        f"> **回测区间：** `{cfg['start_date']}` → `{cfg['end_date']}`　　**生成时间：** `{gen_time}`",
        f"",
        f"---",
        f"",
        f"## 一、策略配置",
        f"",
        f"| 参数 | 值 |",
        f"|:---|:---|",
        f"| 换仓频率 | {'月度' if cfg['rebalance_freq'] == 'M' else '周度'} |",
        f"| 持股数量 | Top {cfg['top_n']} |",
        f"| 单边手续费 | {cfg['commission'] * 100:.1f}‰ |",
        f"| 滑点估计 | {cfg['slippage'] * 100:.1f}‰ |",
        f"| 无风险利率 | {cfg['risk_free_rate'] * 100:.1f}% |",
        f"",
        f"",
        f"## 二、核心绩效指标",
        f"",
        f"| 指标 | 值 | 说明 |",
        f"|:---|:---:|:---|",
        f"| 📈 总收益率 | **{_pct(results.get('total_return', 'N/A'))}** | 全期间累计 |",
        f"| 📊 年化收益率 | **{_pct(results.get('annualized_return', 'N/A'))}** | 复利年化 |",
        f"| ⚡ 夏普比率 | **{_f(results.get('sharpe_ratio', 'N/A'))}** | 超额收益 / 波动率 |",
        f"| 🛡️ 索提诺比率 | **{_f(results.get('sortino_ratio', 'N/A'))}** | 超额收益 / 下行风险 |",
        f"| 📉 最大回撤 | **{_pct(results.get('max_drawdown', 'N/A'))}** | 峰值到谷值 |",
        f"| 🏆 卡尔马比率 | **{_f(results.get('calmar_ratio', 'N/A'))}** | 年化收益 / 最大回撤 |",
        f"| 💨 年化波动率 | **{_pct(results.get('annualized_vol', 'N/A'))}** | |",
        f"",
        f"",
        f"## 三、Alpha 信号质量（IC 分析）",
        f"",
        f"| 指标 | 值 | 说明 |",
        f"|:---|:---:|:---|",
        f"| IC 均值 | **{_f(results.get('ic_mean', 'N/A'))}** | >0.03 视为有效 |",
        f"| IC 标准差 | **{_f(results.get('ic_std', 'N/A'))}** | |",
        f"| ICIR | **{_f(results.get('icir', 'N/A'))}** | >0.5 视为稳定 |",
        f"",
        f"",
        f"## 四、换手率统计",
        f"",
        f"| 指标 | 值 |",
        f"|:---|:---:|",
        f"| 平均换手率（每期） | **{_pct(results.get('avg_turnover', 'N/A'))}** |",
        f"",
        f"",
        f"## 五、逐期换仓记录",
        f"",
        f"| 换仓日 | 持股数 | 换手率 | 期间收益 | IC |",
        f"|:---|:---:|:---:|:---:|:---:|",
    ]

    for r in results.get("rebalance", [])[-20:]:   # 最近 20 期
        ic_str = f"{r['ic']:.4f}" if not np.isnan(r.get("ic", np.nan)) else "—"
        lines.append(
            f"| `{r['date']}` | {len(r['portfolio'])} | "
            f"{r['turnover']:.1%} | {r['period_ret']:.2%} | {ic_str} |"
        )

    lines += [
        f"",
        f"",
        f"---",
        f"",
        f"*© LIUMON Alpha Genome Research Lab · 回测结果仅供研究，过往表现不代表未来收益。*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, cfg: dict = None):
    os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)

    if cfg is None:
        cfg = DEFAULT_CONFIG

    results = run_backtest(cfg=cfg, dry_run=dry_run)

    if not results or results.get("status") == "dry_run_ok":
        return

    # 保存 JSON
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(BACKTEST_OUTPUT_DIR, f"backtest_{date_str}.json")
    # nav 序列单独处理（Timestamp key -> str）
    json_out  = {k: v for k, v in results.items() if k != "nav"}
    json_out["nav_tail"] = {str(k.date()): round(v, 6)
                            for k, v in list(results.get("nav", {}).items())[-30:]}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2, default=str)

    # 保存 MD 报告
    md_path = os.path.join(BACKTEST_OUTPUT_DIR, f"backtest_{date_str}.md")
    report  = generate_backtest_report(results, cfg)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 回测完成")
    print(f"   年化收益: {results.get('annualized_return', 'N/A'):.2%}")
    print(f"   夏普比率: {results.get('sharpe_ratio', 'N/A'):.4f}")
    print(f"   最大回撤: {results.get('max_drawdown', 'N/A'):.2%}")
    print(f"   ICIR:     {results.get('icir', 'N/A')}")
    print(f"\n   报告: {md_path}")
    print(f"   JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liumon A-Share Backtest Engine")
    parser.add_argument("--dry-run",   action="store_true", help="只验证数据，不运行回测")
    parser.add_argument("--start",    default=DEFAULT_CONFIG["start_date"], help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end",      default=DEFAULT_CONFIG["end_date"],   help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--top-n",    type=int, default=DEFAULT_CONFIG["top_n"], help="持股数量")
    parser.add_argument("--freq",     default=DEFAULT_CONFIG["rebalance_freq"], help="换仓频率 M/W")
    args = parser.parse_args()

    run_cfg = {**DEFAULT_CONFIG,
               "start_date":     args.start,
               "end_date":       args.end,
               "top_n":          args.top_n,
               "rebalance_freq": args.freq}

    main(dry_run=args.dry_run, cfg=run_cfg)
