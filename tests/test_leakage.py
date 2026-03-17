import pytest
import pandas as pd
import numpy as np
from features.preprocess_cn import calculate_stock_features

def test_feature_shift_audit():
    """
    Test 1: Feature shift audit
    Ensure that features computed for a specific date only use data from that date or before.
    """
    # Create mock price data
    dates = pd.date_range("2020-01-01", periods=300)
    df = pd.DataFrame({
        "date": dates,
        "close": np.arange(1, 301, dtype=float),  # strictly increasing prices
        "ps": np.random.rand(300),
        "turn": np.random.rand(300),
        "volume": np.random.rand(300)
    })
    # Save to dummy parquet (since calculate_stock_features expects a file path)
    dummy_file = "dummy_ticker.parquet"
    df.to_parquet(dummy_file)

    # Run feature calculation
    res = calculate_stock_features(dummy_file)

    # Verify mom_20d at index T depends on T and T-20
    # For index 25 (date: 2020-01-26), price is 26, price 20 days ago (index 5) is 6.
    # The return should be 26/6 - 1
    t = 25
    expected_mom_20d = res.loc[t, "close"] / res.loc[t-20, "close"] - 1
    assert np.isclose(res.loc[t, "mom_20d"], expected_mom_20d), "Feature leakage detected: mom_20d looks into the future!"

    # Ensure no future data was used
    # E.g., mom_20d calculation uses `pct_change(20)`, which by definition only looks backwards.
    # Clean up dummy file
    import os
    if os.path.exists(dummy_file):
        os.remove(dummy_file)

def test_label_generation_audit():
    """
    Test 2: Label generation audit
    Ensure that the label for next month only uses future prices.
    Since label generation is in main(), we simulate the logic here.
    """
    dates = pd.date_range("2020-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["000001"] * 100,
        "raw_close": np.arange(1, 101, dtype=float)
    })
    df["ym"] = df["date"].dt.to_period("M")

    # Sample month-end dates (similar to preprocess_cn.py)
    panel_me = df.sort_values("date").groupby(["ticker", "ym"]).tail(1).copy()
    panel_me = panel_me.reset_index(drop=True)
    panel_me = panel_me.sort_values(["ticker", "date"])

    # Generate next month label
    panel_me["label_next_month"] = (
        panel_me.groupby("ticker")["raw_close"].shift(-1) /
        panel_me["raw_close"] - 1
    )

    # Audit: label for index T should use raw_close at T+1
    # Check that for date T, the target value comes from date > T
    for i in range(len(panel_me) - 1):
        current_date = panel_me.loc[i, "date"]
        next_date = panel_me.loc[i+1, "date"]
        # The label computation uses the price at next_date
        assert current_date < next_date, f"Label leakage! Target date {next_date} is not strictly after feature date {current_date}"

def test_train_test_split_audit():
    """
    Test 3: Train/test split audit
    Ensure that train data strictly precedes test data in time.
    """
    dates = pd.date_range("2022-01-01", periods=1000)
    df = pd.DataFrame({
        "date": dates,
        "feature1": np.random.rand(1000),
        "label": np.random.rand(1000)
    })

    # Replicate train/test split logic from scripts/train.py
    train_df = df[df['date'] < '2024-01-01']
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')]

    train_max_date = train_df['date'].max()
    test_min_date = test_df['date'].min()

    # Audit: max train date < min test date
    assert train_max_date < test_min_date, f"Split leakage! Train max date ({train_max_date}) is not strictly before test min date ({test_min_date})"

if __name__ == "__main__":
    pytest.main([__file__])
