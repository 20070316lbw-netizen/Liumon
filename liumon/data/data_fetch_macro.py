import os
import sys
import pandas as pd
import baostock as bs
from datetime import date

# 确保配置可访问
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CN_DIR

OUTPUT_PATH   = os.path.join(CN_DIR, "macro_regime.parquet")
INITIAL_START = "2014-01-01"    # 首次全量抓取起点
MA_WINDOW     = 250             # HS300 趋势均线周期


def _flush(msg: str):
    """打印并立即刷新到终端（防止 Rich/缓冲吞输出）"""
    print(msg, flush=True)


def _fetch_hs300_close(start_date: str, end_date: str) -> pd.DataFrame:
    """从 baostock 拉取 HS300 日收盘，返回 date-indexed DataFrame"""
    rs = bs.query_history_k_data_plus(
        "sh.000300", "date,close",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3"
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["close"] = pd.to_numeric(df["close"])
    df["date"]  = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def _build_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    在完整 close 序列上计算宏观体制标签：
      regime = 1 (多头)：前一日收盘价 > 前一日 MA250
      regime = 0 (空头)：前一日收盘价 ≤ 前一日 MA250
    使用昨日数据而非当日，严格防止前瞻偏差。
    """
    df = df.copy().sort_index()
    df["hs300_ma250"]       = df["close"].rolling(window=MA_WINDOW).mean()
    df["hs300_close_prev"]  = df["close"].shift(1)
    df["hs300_ma250_prev"]  = df["hs300_ma250"].shift(1)
    df["regime"] = (df["hs300_close_prev"] > df["hs300_ma250_prev"]).astype(int)
    df.dropna(subset=["hs300_ma250_prev"], inplace=True)
    df.reset_index(inplace=True)
    return df[["date", "hs300_close_prev", "hs300_ma250_prev", "regime"]].copy()


def fetch_and_build_macro_regimes():
    _flush("=" * 60)
    _flush("  Liumon Macro Engine: HS300 Regime Modeling")
    _flush("=" * 60)

    today_str = date.today().strftime("%Y-%m-%d")

    # ── 判断全量 vs 增量 ──────────────────────────────────────
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_parquet(OUTPUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        last_date = existing["date"].max()
        next_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if next_date > today_str:
            _flush(f"[宏观数据] 已是最新（最新日期: {last_date.date()}），无需更新。")
            # 打印最新状态
            latest = existing.tail(1).iloc[0]
            regime_label = "🟢 牛市 (Bull)" if latest["regime"] == 1 else "🔴 熊市 (Bear)"
            _flush(f"[当前体制] {latest['date'].date()} → {regime_label}")
            return

        # 需要往前推 MA_WINDOW 天确保均线计算正确
        # 取已有数据最后 300 条作为上下文 + 拉新增数据
        context_df = existing.set_index("date").rename(columns={
            "hs300_close_prev": "_", "hs300_ma250_prev": "__"
        }).pipe(lambda x: x)   # existing 里没有原始 close，需要重拉 close

        # 所以增量时，我们拉从 (last_date - 300天) 到 today 的 close，
        # 与原始 close 合并重算，保证 MA250 连续正确
        context_start = (last_date - pd.Timedelta(days=360)).strftime("%Y-%m-%d")
        mode = "增量"
        fetch_start = context_start
        _flush(f"[宏观数据] 增量模式：上次 {last_date.date()} → 今日 {today_str}")
    else:
        fetch_start = INITIAL_START
        mode = "全量"
        _flush(f"[宏观数据] 首次全量抓取：{INITIAL_START} → {today_str}")

    # ── 登录 baostock ─────────────────────────────────────────
    _flush(f"[baostock] 正在登录...")
    lg = bs.login()
    if lg.error_code != "0":
        _flush(f"❌ baostock 登录失败: {lg.error_msg}")
        return
    _flush(f"[baostock] 登录成功，开始拉取 HS300 收盘数据...")

    # ── 拉取数据 ──────────────────────────────────────────────
    raw_df = _fetch_hs300_close(fetch_start, today_str)
    bs.logout()
    _flush(f"[baostock] 已登出。获取 {len(raw_df)} 条记录。")

    if raw_df.empty:
        _flush("⚠️  未获取到任何数据，请检查网络或日期范围。")
        return

    # ── 全量 or 增量合并 ──────────────────────────────────────
    if mode == "全量":
        full_close = raw_df
    else:
        # 合并：已有 close 数据（从 2014 起）
        if os.path.exists(OUTPUT_PATH):
            old_res = pd.read_parquet(OUTPUT_PATH)
            old_res["date"] = pd.to_datetime(old_res["date"])
            # 我们没有存原始 close，只能用 hs300_close_prev 推算
            # 重拉从 INITIAL_START 到 context_start 的旧数据 —— 但这样太慢
            # 更好的方法：基于旧的 hs300_close_prev （已知上一日收盘）
            # 直接把新抓到的 close 截取并和旧数据拼接重算
            # 简化处理：新增数据的 context_start 已确保 MA 窗口够长，直接用新拉的 raw_df
            # （覆盖旧的 context_start 之后部分，保留 context_start 之前部分 regime 标签）
            before_mask = old_res["date"] < pd.to_datetime(fetch_start)
            old_tail    = old_res[~before_mask]
            full_close  = raw_df  # raw_df 已从 context_start 起，MA 计算会正确
        else:
            full_close = raw_df

    # ── 重新计算体制标签 ──────────────────────────────────────
    _flush("[计算] 构建宏观体制标签（MA250 趋势判断）...")
    result_df = _build_regimes(full_close)

    # ── 增量时：拼接历史 + 新计算结果 ────────────────────────
    if mode == "增量" and os.path.exists(OUTPUT_PATH):
        old_res  = pd.read_parquet(OUTPUT_PATH)
        old_res["date"] = pd.to_datetime(old_res["date"])
        before   = old_res[old_res["date"] < pd.to_datetime(fetch_start)]
        result_df = pd.concat([before, result_df], ignore_index=True)
        result_df = result_df.drop_duplicates(subset="date", keep="last").sort_values("date")

    # ── 保存 ──────────────────────────────────────────────────
    result_df.to_parquet(OUTPUT_PATH, index=False)
    _flush(f"[保存] 共 {len(result_df)} 条体制数据 → {OUTPUT_PATH}")

    # ── 审计输出 ──────────────────────────────────────────────
    _flush("\n[Audit] 最近 5 天宏观状态:")
    _flush("-" * 40)
    _flush(str(result_df.tail(5).to_string(index=False)))
    _flush("\n[Audit] 体制分布:")
    _flush(str(result_df["regime"].value_counts(normalize=True).rename({1: "牛市(Bull)", 0: "熊市(Bear)"})))
    _flush("-" * 40)

    latest = result_df.iloc[-1]
    regime_label = "🟢 牛市 (Bull)" if latest["regime"] == 1 else "🔴 熊市 (Bear)"
    _flush(f"\n[当前体制] {pd.to_datetime(latest['date']).date()} → {regime_label}")
    _flush("[DONE] 宏观状态数据同步完成。")


def main():
    fetch_and_build_macro_regimes()


if __name__ == "__main__":
    main()
