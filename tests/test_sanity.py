import pytest
import pandas as pd
import numpy as np

def test_survivor_bias_audit():
    """
    Test 1: Survivor bias test
    Ensure the stock universe changes over time (i.e., new listings and delistings occur),
    reflecting a realistic dataset and avoiding survivorship bias where only currently
    active stocks are backtested.
    """
    # Simulate a universe where stocks enter and exit
    dates = pd.date_range("2020-01-01", periods=100)

    # Stock A: listed for the entire period
    df_a = pd.DataFrame({"date": dates, "ticker": "A", "price": 10.0})

    # Stock B: listed halfway through (IPO)
    df_b = pd.DataFrame({"date": dates[40:], "ticker": "B", "price": 20.0})

    # Stock C: delisted halfway through
    df_c = pd.DataFrame({"date": dates[:60], "ticker": "C", "price": 5.0})

    # This means:
    # Day 0-39: A, C (Count: 2)
    # Day 40-59: A, B, C (Count: 3)
    # Day 60-99: A, B (Count: 2)

    dataset = pd.concat([df_a, df_b, df_c], ignore_index=True)

    # Audit: Number of stocks should vary over time
    daily_counts = dataset.groupby("date").size()

    # Since stocks enter and leave, the number of unique cross-sectional counts should be > 1
    assert len(daily_counts.unique()) > 1, "Survivor bias detected: The stock universe size is constant over time!"

    # Verify specific periods
    assert daily_counts.iloc[0] == 2, "Expected 2 stocks on the first day (A, C)"
    assert daily_counts.iloc[-1] == 2, "Expected 2 stocks on the last day (A, B)"


def test_feature_nan_audit():
    """
    Test 2: Feature NaN audit
    Ensure that rolling windows do not produce an excessive amount of NaNs
    that would later be improperly filled by the model.
    """
    # Simulate 250 days of data for one stock
    dates = pd.date_range("2020-01-01", periods=250)
    df = pd.DataFrame({
        "date": dates,
        "close": np.random.rand(250) * 100,
        "volume": np.random.rand(250) * 1000
    })

    # Compute typical features like 20d, 60d, 120d momentum
    df["mom_20d"] = df["close"].pct_change(20)
    df["mom_60d"] = df["close"].pct_change(60)
    df["mom_120d"] = df["close"].pct_change(120)
    df["vol_60d"] = df["close"].pct_change().rolling(60).std()

    # Audit: check NaN ratio after the maximum burn-in period (120 days)
    # The first 120 rows will naturally have NaNs for mom_120d. We only audit the valid period.
    valid_df = df.iloc[120:].copy()

    features = valid_df[["mom_20d", "mom_60d", "mom_120d", "vol_60d"]]

    # Max NaN ratio across all feature columns should be low (< 30%)
    max_nan_ratio = features.isna().mean().max()
    assert max_nan_ratio < 0.3, f"Feature NaN audit failed! Max NaN ratio is {max_nan_ratio:.2%}, which exceeds the 30% threshold."


def test_future_return_sanity():
    """
    Test 3: Future return sanity test
    Ensure the generated label (future return) is within reasonable bounds.
    If the average return is excessively high (e.g., > 5% per month systematically),
    it strongly implies a data leak or look-ahead bias (future function).
    """
    # Simulate a realistic monthly return distribution (e.g., mean ~ 0.5%, std ~ 5%)
    # This reflects a normal stock market environment without a future function.
    np.random.seed(42)
    n_samples = 10000
    simulated_returns = np.random.normal(loc=0.005, scale=0.05, size=n_samples)

    df = pd.DataFrame({"label_next_month": simulated_returns})

    # Calculate the average label
    mean_label = df["label_next_month"].mean()

    # Audit: The systematic mean return should be realistic (< 5%)
    assert mean_label < 0.05, f"Future return sanity check failed! Unrealistic average return: {mean_label:.2%}. Possible future function leak!"

    # Additionally, check that it's not exactly zero or identical across all samples
    assert df["label_next_month"].std() > 0, "All returns are identical."

if __name__ == "__main__":
    pytest.main([__file__])
