import os
import sys
import pandas as pd
import baostock as bs

# 确保配置可访问
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CN_DIR

OUTPUT_PATH = os.path.join(CN_DIR, 'macro_regime.parquet')

def fetch_and_build_macro_regimes(start_date="2014-01-01"):
    print("="*60)
    print("  Liumon Macro Engine: HS300 Regime Modeling")
    print("="*60)
    
    bs.login()
    rs = bs.query_history_k_data_plus("sh.000300", "date,close",
                                      start_date=start_date, frequency="d", adjustflag="3")
    data_list = []
    while rs.next(): data_list.append(rs.get_row_data())
    bs.logout()
    
    df = pd.DataFrame(data_list, columns=["date", "close"])
    df["close"] = pd.to_numeric(df["close"])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    # 技术逻辑 (防止前瞻偏差)
    df['hs300_ma250'] = df['close'].rolling(window=250).mean()
    df['hs300_close_prev'] = df['close'].shift(1)
    df['hs300_ma250_prev'] = df['hs300_ma250'].shift(1)
    
    df['regime'] = 0
    df.loc[df['hs300_close_prev'] > df['hs300_ma250_prev'], 'regime'] = 1
    
    df.dropna(subset=['hs300_ma250_prev'], inplace=True)
    df.reset_index(inplace=True)
    
    # 保存
    res_df = df[['date', 'hs300_close_prev', 'hs300_ma250_prev', 'regime']].copy()
    res_df.to_parquet(OUTPUT_PATH)
    
    # 按照用户请求审核输出
    print(f"\n[Audit] Macro Regime Data (Tail):")
    print("-" * 30)
    print(res_df.tail(5))
    print("\n[Audit] Distribution:")
    print(res_df['regime'].value_counts(normalize=True))
    print("-" * 30)
    
    print(f"[DONE] Macro regimes synced to: {OUTPUT_PATH}")

if __name__ == "__main__":
    fetch_and_build_macro_regimes()
