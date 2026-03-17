import os
import tempfile
import numpy as np
import pandas as pd
import pytest

# 导入项目中可能用到的配置或依赖
# 我们会在测试中模拟特征生成和切分逻辑，或者直接测试相关模块的功能

# --- 辅助函数：生成 Dummy 数据 ---
def generate_dummy_data(days=250, tickers=5):
    """
    生成模拟的 OHLCV 和一些基本字段的数据。
    """
    dates = pd.date_range(start="2022-01-01", periods=days, freq="B")
    data = []
    for ticker in range(tickers):
        ticker_str = f"TICKER_{ticker:02d}"

        # 基础数据：以 100 开始的随机游走
        np.random.seed(42 + ticker) # 保证每次生成一致
        returns = np.random.normal(0, 0.02, days)
        close = 100 * np.exp(np.cumsum(returns))
        volume = np.random.randint(1000, 10000, days)
        ps = np.random.uniform(0.5, 5, days)

        df = pd.DataFrame({
            "date": dates,
            "ticker": ticker_str,
            "close": close,
            "raw_close": close,
            "volume": volume,
            "ps": ps,
            "turn": np.random.uniform(0.01, 0.1, days)
        })
        data.append(df)

    return pd.concat(data, ignore_index=True)


# --- 第一部分：数据泄露审计 (Leakage Audits) ---

def test_feature_shift_audit():
    """
    Feature shift audit: 测试因子的滚动计算（如 mom_20d）严格没有使用未来数据。
    """
    df = generate_dummy_data(days=50, tickers=1)
    df = df.set_index("date")

    # 我们测试 pct_change 的行为，等同于特征工程中的计算
    # df["mom_20d"] = df["close"].pct_change(20)

    T_index = 30
    T_date = df.index[T_index]

    # 原始计算
    original_mom_20d = df["close"].pct_change(20).loc[T_date]

    # 修改 T 之后的数据 (未来数据)
    df_modified = df.copy()
    df_modified.loc[df_modified.index > T_date, "close"] *= 2.0

    # 再次计算
    new_mom_20d = df_modified["close"].pct_change(20).loc[T_date]

    # 断言：修改未来数据不应影响 T 时刻的因子值
    assert np.isclose(original_mom_20d, new_mom_20d), "报错：特征滚动计算使用了未来数据！"


def test_label_generation_audit():
    """
    Label generation audit: 测试标签生成逻辑，确保索引 T 处预测的标签严格来自日期 >T 的数据。
    """
    # 模拟 features/preprocess_cn.py 中的月末采样和标签生成逻辑
    df = generate_dummy_data(days=60, tickers=2)
    # 确保我们每个月有多条数据
    df["ym"] = df["date"].dt.to_period("M")
    panel_me = df.sort_values("date").groupby(["ticker", "ym"]).tail(1).copy()
    panel_me = panel_me.reset_index(drop=True)
    panel_me = panel_me.sort_values(["ticker", "date"])

    # T 时刻的标签是基于下个月底的数据
    panel_me["label_next_month"] = (
        panel_me.groupby("ticker")["raw_close"].shift(-1) / panel_me["raw_close"] - 1
    )

    ticker_to_test = "TICKER_00"
    ticker_data = panel_me[panel_me["ticker"] == ticker_to_test].copy()

    if len(ticker_data) >= 2:
        T_idx = 0
        T_plus_1_idx = 1

        T_label = ticker_data.iloc[T_idx]["label_next_month"]

        # 验证 1：标签确实是由 T+1 的数据决定的
        manual_label = ticker_data.iloc[T_plus_1_idx]["raw_close"] / ticker_data.iloc[T_idx]["raw_close"] - 1
        assert np.isclose(T_label, manual_label), "报错：标签生成逻辑错误，不符合预期的 T+1 收益！"

        # 修改未来数据 (T+1 的价格)，检查标签是否发生变化，以此验证确实关联
        ticker_data.iloc[T_plus_1_idx, ticker_data.columns.get_loc("raw_close")] *= 1.1
        ticker_data["new_label_next_month"] = (
            ticker_data.groupby("ticker")["raw_close"].shift(-1) / ticker_data["raw_close"] - 1
        )
        new_T_label = ticker_data.iloc[T_idx]["new_label_next_month"]

        assert not np.isclose(T_label, new_T_label), "报错：修改未来收益后标签未改变，说明标签未使用未来的真实数据！"


def test_train_test_split_audit():
    """
    Train/test split audit: 复现并断言 scripts/train.py 中的训练集/测试集切分逻辑。
    确保训练集的最大日期严格小于测试集的最小日期。
    """
    # 构造跨越 2023 和 2024 年的数据
    dates_2023 = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
    dates_2024 = pd.date_range(start="2024-01-01", end="2024-12-31", freq="B")

    df = pd.DataFrame({"date": dates_2023.union(dates_2024), "value": 1})

    # 模拟 scripts/train.py 的切分
    train_df = df[df['date'] < '2024-01-01']
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')]

    max_train_date = train_df['date'].max()
    min_test_date = test_df['date'].min()

    assert max_train_date < min_test_date, f"报错：时间切分发生泄漏！训练集最大日期 {max_train_date} >= 测试集最小日期 {min_test_date}"

    # 模拟 80/20 fallback
    df_fallback = pd.DataFrame({"date": pd.date_range(start="2023-01-01", periods=100, freq="B")})
    all_dates = sorted(df_fallback["date"].unique())
    n = len(all_dates)
    train_dates = all_dates[:int(n*0.8)]

    train_df_fb = df_fallback[df_fallback["date"].isin(train_dates)]
    test_df_fb = df_fallback[~df_fallback["date"].isin(train_dates)]

    assert train_df_fb['date'].max() < test_df_fb['date'].min(), "报错：Fallback 切分发生泄漏！"


# --- 第二部分：数据健全性与偏见审计 (Sanity Audits) ---

def test_survivor_bias_audit():
    """
    Survivor bias audit: 模拟多只股票中途上市与退市，测试数据集的每日唯一股票数量是否动态变化，以防范幸存者偏差。
    """
    # 生成 3 只股票，它们有不同的生命周期
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")

    df1 = pd.DataFrame({"date": dates[:6], "ticker": "A"})  # 在第 6 天退市
    df2 = pd.DataFrame({"date": dates[3:], "ticker": "B"})  # 在第 4 天上市
    df3 = pd.DataFrame({"date": dates, "ticker": "C"})      # 贯穿始终

    df = pd.concat([df1, df2, df3], ignore_index=True)

    # 模拟写入和读取 parquet 文件
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "survivor_bias_test.parquet")

    try:
        df.to_parquet(test_file)
        loaded_df = pd.read_parquet(test_file)

        daily_counts = loaded_df.groupby("date")["ticker"].nunique()

        # 每日股票数量必须动态变化
        assert len(daily_counts.unique()) > 1, "报错：每日截面的股票数量是固定的，可能存在幸存者偏差或前向填充错误！"

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
        os.rmdir(temp_dir)


def test_feature_nan_audit():
    """
    Feature NaN audit: 模拟生成 250 天数据，在扣除滚动窗口最大预热期（如120天）后，
    断言所有因子的最大空值率（NaN ratio）必须 < 30%。
    """
    df = generate_dummy_data(days=250, tickers=2)
    df["mom_120d"] = df.groupby("ticker")["close"].pct_change(120)
    df["mom_60d"] = df.groupby("ticker")["close"].pct_change(60)

    # 模拟存取
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "nan_audit_test.parquet")

    try:
        df.to_parquet(test_file)
        loaded_df = pd.read_parquet(test_file)

        # 扣除前 120 天作为预热期
        warmup_period = 120
        # 获取第 120 天对应的日期（假设每个 ticker 有相同的天数，取第一个 ticker 的第 120 个日期）
        start_date = loaded_df["date"].sort_values().unique()[warmup_period]

        valid_df = loaded_df[loaded_df["date"] >= start_date]

        # 检查空值率
        features_to_check = ["mom_120d", "mom_60d"]
        for feature in features_to_check:
            nan_ratio = valid_df[feature].isna().mean()
            assert nan_ratio < 0.30, f"报错：因子 {feature} 的空值率达到了 {nan_ratio:.2%}，超过了 30% 的阈值！"

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
        os.rmdir(temp_dir)


def test_future_return_sanity_check():
    """
    Future return sanity check: 如果生成的标签是未来收益，对一个合理的虚拟分布进行断言，
    保证总体平均收益 mean() < 0.05。如果平均收益过高，报错提示存在未来函数。
    """
    df = generate_dummy_data(days=100, tickers=5)

    # 模拟正常的标签
    df["label_next_month"] = df.groupby("ticker")["raw_close"].shift(-1) / df["raw_close"] - 1

    # 模拟存取
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "future_return_test.parquet")

    try:
        df.to_parquet(test_file)
        loaded_df = pd.read_parquet(test_file)

        # 过滤掉 NaN 的标签
        labels = loaded_df["label_next_month"].dropna()
        mean_return = labels.mean()

        # 断言：正常的未来收益平均值不应大得离谱
        assert mean_return < 0.05, f"报错：总体平均收益 ({mean_return:.2%}) 过高，可能存在未来函数或价格未复权！"

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
        os.rmdir(temp_dir)
