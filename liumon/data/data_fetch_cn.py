import os
import sys
import json
import signal
import baostock as bs
import pandas as pd
from datetime import datetime, date
from tqdm import tqdm

# 确保配置可访问
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PRICE_DIR, CN_DIR

# ── 检查点文件路径 ──────────────────────────────────────────────
CHECKPOINT_FILE = os.path.join(CN_DIR, ".fetch_checkpoint.json")


# ── 辅助：baostock 格式转换 ─────────────────────────────────────
def _std_to_bs(symbol: str) -> str:
    if symbol.upper().endswith(".SS"): return f"sh.{symbol[:-3]}"
    if symbol.upper().endswith(".SZ"): return f"sz.{symbol[:-3]}"
    return symbol


# ── 检查点读写 ──────────────────────────────────────────────────
def load_checkpoint() -> dict:
    """读取上次中断时保存的进度"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                cp = json.load(f)
            print(f"[断点续传] 发现上次进度：已完成 {len(cp.get('done', []))} 只股票，"
                  f"日期：{cp.get('fetch_date', '未知')}")
            return cp
        except Exception:
            pass
    return {"done": [], "failed": [], "fetch_date": None}


def save_checkpoint(done: list, failed: list, fetch_date: str):
    """持久化当前抓取进度"""
    cp = {"done": done, "failed": failed, "fetch_date": fetch_date}
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


def clear_checkpoint():
    """任务完成后清除检查点"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("[检查点] 已清除进度文件（任务完成）")


# ── 重复数据清理 ────────────────────────────────────────────────
def deduplicate_parquet(file_path: str) -> int:
    """
    读取 parquet 文件，删除重复日期行（保留最新一条），返回清理的重复行数。
    """
    df = pd.read_parquet(file_path)
    original_len = len(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    dup_count = original_len - len(df)
    if dup_count > 0:
        df.to_parquet(file_path, compression="snappy")
    return dup_count


def run_dedup_scan(price_dir: str = None) -> dict:
    """
    扫描所有 parquet 文件，删除重复行，输出统计报告。
    Returns: {"files_checked": int, "files_cleaned": int, "rows_removed": int}
    """
    if price_dir is None:
        price_dir = PRICE_DIR

    stats = {"files_checked": 0, "files_cleaned": 0, "rows_removed": 0}
    files = [f for f in os.listdir(price_dir) if f.endswith(".parquet")]

    print(f"\n[去重扫描] 共发现 {len(files)} 个 parquet 文件...")
    cleaned = []
    for fname in tqdm(files, desc="Dedup Scan"):
        fpath = os.path.join(price_dir, fname)
        try:
            removed = deduplicate_parquet(fpath)
            stats["files_checked"] += 1
            if removed > 0:
                stats["files_cleaned"] += 1
                stats["rows_removed"] += removed
                cleaned.append(f"{fname}: 清除 {removed} 条重复")
        except Exception as e:
            print(f"  ⚠️  跳过 {fname}: {e}")

    if cleaned:
        print(f"\n[去重报告] 清理了以下文件:")
        for item in cleaned:
            print(f"  - {item}")
    print(f"\n[去重完成] 检查: {stats['files_checked']} | "
          f"清理: {stats['files_cleaned']} | 去除重复行: {stats['rows_removed']}")
    return stats


# ── 股票数据抓取（支持增量 + 去重） ────────────────────────────
def fetch_stock_data(ticker_std, start_date="2014-01-01", end_date=None):
    """
    全量/增量抓取：
    - 若本地文件已存在，从最新日期+1天开始增量拉取
    - 合并时自动去重（保留最新一条），防止重复数据积累
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    file_path = os.path.join(PRICE_DIR, f"{ticker_std}.parquet")

    # ── 增量逻辑 ──────────────────────────────
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)

        # 顺便去除已保存数据中的重复行
        existing_len = len(existing_df)
        existing_df  = existing_df[~existing_df.index.duplicated(keep="last")].sort_index()
        if len(existing_df) < existing_len:
            existing_df.to_parquet(file_path, compression="snappy")   # 回写去重后的数据

        last_date    = existing_df.index.max()
        actual_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if actual_start > end_date:
            return "UP_TO_DATE", existing_df
        is_incremental = True
    else:
        actual_start   = start_date
        is_incremental = False
        existing_df    = None

    # ── 从 baostock 拉取 ───────────────────────
    try:
        rs = bs.query_history_k_data_plus(
            _std_to_bs(ticker_std),
            "date,open,high,low,close,volume,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
            start_date=actual_start, end_date=end_date,
            frequency="d", adjustflag="2"
        )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return "NO_NEW_DATA", existing_df

        new_df = pd.DataFrame(rows, columns=["date","open","high","low","close",
                                              "volume","turn","pe","pb","ps","pcf","is_st"])
        new_df["date"]   = pd.to_datetime(new_df["date"])
        new_df["ticker"] = ticker_std
        for c in ["open","high","low","close","volume","turn","pe","pb","ps","pcf"]:
            new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
        new_df.set_index("date", inplace=True)

        # ── 合并 + 去重（新数据优先） ──────────
        if is_incremental:
            combined_df = pd.concat([existing_df, new_df])
        else:
            combined_df = new_df

        # 关键去重：按日期去重，保留最新抓取的一条
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")].sort_index()

        combined_df.to_parquet(file_path, compression="snappy")
        return "SUCCESS", combined_df

    except Exception as e:
        print(f"Error fetching {ticker_std}: {e}")
        return "ERROR", None


# ── 获取成分股列表 ──────────────────────────────────────────────
def get_index_tickers() -> list:
    """
    获取 HS300 + CSI500 全量成分股列表。
    注意：调用前必须已完成 bs.login()，内部不再管理 session。
    """
    tickers = []
    for query_func in [bs.query_hs300_stocks, bs.query_zz500_stocks]:
        rs = query_func()
        while rs.next():
            r = rs.get_row_data()
            tickers.append(r[1].split(".")[1] + (".SS" if r[1].startswith("sh") else ".SZ"))
    return list(set(tickers))


# ── 主函数（含断点续传 + 中断保护） ────────────────────────────
def main():
    print("=" * 60)
    print("  Liumon Data Hub: A-Share Historical & Increment Fetcher")
    print("  [支持 断点续传 | 去重清理]")
    print("=" * 60)

    today_str = date.today().strftime("%Y-%m-%d")

    # 1. 读取检查点
    cp        = load_checkpoint()
    done_set  = set(cp.get("done", []))
    failed_list = list(cp.get("failed", []))

    # 如果是全新的一天，清空旧检查点（避免日期错乱）
    if cp.get("fetch_date") and cp["fetch_date"] != today_str:
        print(f"[检查点] 检测到旧日期检查点 ({cp['fetch_date']})，自动清除并重新抓取。")
        done_set    = set()
        failed_list = []

    # 2. 登录 baostock
    bs.login()
    tickers = get_index_tickers()
    print(f">> Project Universe: {len(tickers)} stocks (HS300 + CSI500)")

    # 过滤掉已完成的股票
    remaining = [t for t in tickers if t not in done_set]
    skipped   = len(tickers) - len(remaining)
    if skipped > 0:
        print(f"[断点续传] 跳过已完成 {skipped} 只，本次剩余 {len(remaining)} 只")

    # 3. 中断信号处理（Ctrl+C 优雅保存）
    interrupted = False
    success_this_run = []

    def _handle_interrupt(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n⚠️  检测到中断！正在保存进度...")

    signal.signal(signal.SIGINT, _handle_interrupt)

    # 4. 先做一次去重扫描（快速，在抓取前清理旧数据）
    if os.path.isdir(PRICE_DIR) and os.listdir(PRICE_DIR):
        print("\n[预处理] 检查已有数据中的重复行...")
        dedup_stats = run_dedup_scan(PRICE_DIR)

    # 5. 逐股抓取
    audit_done = False
    pbar = tqdm(remaining, desc="Syncing Data")
    for ticker in pbar:
        if interrupted:
            break

        pbar.set_postfix_str(ticker)
        status, df = fetch_stock_data(ticker)

        if status == "SUCCESS":
            done_set.add(ticker)
            success_this_run.append(ticker)
        elif status == "UP_TO_DATE":
            done_set.add(ticker)   # 也算完成
        elif status == "ERROR":
            failed_list.append(ticker)

        # 每 10 只保存一次检查点
        if len(success_this_run) % 10 == 0 and success_this_run:
            save_checkpoint(list(done_set), failed_list, today_str)

        # 审计：显示第一只成功的股票样本
        if not audit_done and status == "SUCCESS" and df is not None:
            audit_done = True
            print(f"\n[Audit] Data Sample for {ticker}:")
            print("-" * 30)
            print(df.tail(3))
            print("-" * 30)

    # 6. 中断后保存检查点，否则清除
    bs.logout()

    if interrupted:
        save_checkpoint(list(done_set), failed_list, today_str)
        print(f"\n⏸️  进度已保存（完成 {len(done_set)}/{len(tickers)}）。")
        print("   下次运行会从断点自动继续。")
    else:
        # 全部完成，清除检查点
        clear_checkpoint()
        total_done = len(done_set)
        print(f"\n✅ [DONE] 本次成功同步 {len(success_this_run)} 只，累计完成 {total_done} 只。")
        if failed_list:
            print(f"⚠️  失败股票 ({len(failed_list)} 只): {failed_list[:10]}{'...' if len(failed_list)>10 else ''}")
        print(f"   数据保存至: {PRICE_DIR}")


if __name__ == "__main__":
    main()
