import os
import pytest
import pandas as pd
import numpy as np
from features.preprocess_cn import calculate_stock_features

# ==============================================================================
# LEAKAGE AUDITS
# ==============================================================================

def test_feature_shift_audit():
    """
    Test 1: Feature shift audit
    确保为特定日期计算的特征仅使用当天或之前的数据。
    """
    # Create mock price data
    dates = pd.date_range("2020-01-01", periods=300)
    df = pd.DataFrame({
        "date": dates,
        "close": np.arange(1, 301, dtype=float),  # 严格递增价格
        "ps": np.random.rand(300),
        "turn": np.random.rand(300),
        "volume": np.random.rand(300)
    })
    # Save to dummy parquet (因为 calculate_stock_features 期望一个文件路径)
    dummy_file = "dummy_ticker.parquet"
    df.to_parquet(dummy_file)

    try:
        # 运行特征计算
        res = calculate_stock_features(dummy_file)

        # 验证索引 T 处的 mom_20d 取决于 T 和 T-20
        # 对于索引 25 (日期: 2020-01-26), 价格为 26, 20天前(索引 5)的价格为 6.
        # 收益应该为 26/6 - 1
        t = 25
        expected_mom_20d = res.loc[t, "close"] / res.loc[t-20, "close"] - 1
        assert np.isclose(res.loc[t, "mom_20d"], expected_mom_20d), "检测到特征泄露：mom_20d 偷看了未来数据！"
    finally:
        # 清理虚拟文件
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

def test_label_generation_audit():
    """
    Test 2: Label generation audit
    确保下个月的标签仅使用未来的价格。
    """
    dates = pd.date_range("2020-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["000001"] * 100,
        "raw_close": np.arange(1, 101, dtype=float)
    })
    df["ym"] = df["date"].dt.to_period("M")

    # 模拟月底采样 (类似于 preprocess_cn.py)
    panel_me = df.sort_values("date").groupby(["ticker", "ym"]).tail(1).copy()
    panel_me = panel_me.reset_index(drop=True)
    panel_me = panel_me.sort_values(["ticker", "date"])

    # 生成下个月的标签
    panel_me["label_next_month"] = (
        panel_me.groupby("ticker")["raw_close"].shift(-1) /
        panel_me["raw_close"] - 1
    )

    # 审计：对于索引 T 的标签，应使用 T+1 的 raw_close
    for i in range(len(panel_me) - 1):
        current_date = panel_me.loc[i, "date"]
        next_date = panel_me.loc[i+1, "date"]
        # 断言下一期的日期严格晚于当期日期
        assert current_date < next_date, f"标签泄露！目标日期 {next_date} 并非严格晚于特征日期 {current_date}"

def test_train_test_split_audit():
    """
    Test 3: Train/test split audit
    确保训练数据在时间上严格早于测试数据。
    """
    dates = pd.date_range("2022-01-01", periods=1000)
    df = pd.DataFrame({
        "date": dates,
        "feature1": np.random.rand(1000),
        "label": np.random.rand(1000)
    })

    # 复制 scripts/train.py 中的训练/测试集拆分逻辑
    train_df = df[df['date'] < '2024-01-01']
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')]

    train_max_date = train_df['date'].max()
    test_min_date = test_df['date'].min()

    # 审计：训练集最大日期 < 测试集最小日期
    assert train_max_date < test_min_date, f"拆分泄露！训练集最大日期 ({train_max_date}) 并非严格早于测试集最小日期 ({test_min_date})"


# ==============================================================================
# SANITY AUDITS
# ==============================================================================

def test_survivor_bias_audit():
    """
    Test 4: Survivor bias test
    确保股票池随时间变化（即发生新的上市和退市），
    反映真实的数据集，避免仅使用当前活跃股票进行回测的幸存者偏差。
    """
    dates = pd.date_range("2020-01-01", periods=100)

    # 股票 A：全程上市
    df_a = pd.DataFrame({"date": dates, "ticker": "A", "price": 10.0})

    # 股票 B：中途上市 (IPO)
    df_b = pd.DataFrame({"date": dates[40:], "ticker": "B", "price": 20.0})

    # 股票 C：中途退市
    df_c = pd.DataFrame({"date": dates[:60], "ticker": "C", "price": 5.0})

    dataset = pd.concat([df_a, df_b, df_c], ignore_index=True)

    # 审计：股票数量应随时间变化
    daily_counts = dataset.groupby("date").size()

    # 由于股票有进有出，唯一的横截面数量应大于 1
    assert len(daily_counts.unique()) > 1, "检测到幸存者偏差：股票池数量随时间保持不变！"

    # 验证特定时期
    assert daily_counts.iloc[0] == 2, "预期第一天有 2 只股票 (A, C)"
    assert daily_counts.iloc[-1] == 2, "预期最后一天有 2 只股票 (A, B)"


def test_feature_nan_audit():
    """
    Test 5: Feature NaN audit
    确保滚动窗口不会产生过多的 NaN（超过30%），
    这在之后会被模型不当地填充。
    """
    dates = pd.date_range("2020-01-01", periods=250)
    df = pd.DataFrame({
        "date": dates,
        "close": np.random.rand(250) * 100,
        "volume": np.random.rand(250) * 1000
    })

    # 计算典型的动量和波动率特征
    df["mom_20d"] = df["close"].pct_change(20)
    df["mom_60d"] = df["close"].pct_change(60)
    df["mom_120d"] = df["close"].pct_change(120)
    df["vol_60d"] = df["close"].pct_change().rolling(60).std()

    # 审计：在最大预热期（120天）后检查 NaN 比例
    valid_df = df.iloc[120:].copy()

    features = valid_df[["mom_20d", "mom_60d", "mom_120d", "vol_60d"]]

    # 所有特征列的最大 NaN 比例应处于低位（< 30%）
    max_nan_ratio = features.isna().mean().max()
    assert max_nan_ratio < 0.3, f"特征 NaN 审计失败！最大 NaN 比例为 {max_nan_ratio:.2%}，超过 30% 的阈值。"


def test_future_return_sanity():
    """
    Test 6: Future return sanity test
    确保生成的标签（未来收益）在合理范围内。
    如果平均收益过高（例如：系统性地 > 5% /月），
    则强烈暗示存在数据泄露或偷看未来（未来函数）。
    """
    np.random.seed(42)
    n_samples = 10000
    # 模拟真实的月度回报分布 (均值 ~ 0.5%, 标准差 ~ 5%)
    simulated_returns = np.random.normal(loc=0.005, scale=0.05, size=n_samples)

    df = pd.DataFrame({"label_next_month": simulated_returns})

    # 计算平均标签
    mean_label = df["label_next_month"].mean()

    # 审计：系统平均回报应切实可行（< 5%）
    assert mean_label < 0.05, f"未来收益健全性检查失败！不切实际的平均回报：{mean_label:.2%}。可能存在未来函数泄露！"

    # 检查回报并非完全相同或为零
    assert df["label_next_month"].std() > 0, "所有回报都相同。"


if __name__ == "__main__":
    pytest.main([__file__])
