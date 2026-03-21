import duckdb
import pandas as pd
import os
import sqlite3
import sys

# 增加搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import DB_PATH, CN_DIR

# 旧数据路径
FEATURES_PARQUET_PATH = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
SQLITE_DB_PATH = r'C:\AI_Workplace\finance_system.db' # 假设路径

from alpharanker.database.db_manager import QuantDBManager

def migrate_parquet_to_duckdb():
    print("Starting migration: Parquet -> DuckDB (features_cn)...")
    if not os.path.exists(FEATURES_PARQUET_PATH):
        print(f"Parquet file not found: {FEATURES_PARQUET_PATH}, skipping.")
        return

    db = QuantDBManager()
    
    print(f"Reading {FEATURES_PARQUET_PATH} and aligning columns...")
    try:
        # 1. 使用 Pandas 读取以处理列缺失问题
        df = pd.read_parquet(FEATURES_PARQUET_PATH)
        
        # 2. 定义目标架构要求的列 (按 schema.sql 顺序)
        target_cols = [
            'ticker', 'date', 'index_group', 'regime_label',
            'mom_20d', 'mom_60d', 'mom_12m_minus_1m', 'vol_60d_res', 'sp_ratio', 'turn_20d',
            'mom_20d_rank', 'mom_60d_rank', 'mom_12m_minus_1m_rank', 'vol_60d_res_rank', 'sp_ratio_rank', 'turn_20d_rank',
            'label_next_month', 'label_next_month_rank'
        ]
        
        # 3. 补全缺失列（设为 None/NaN）
        for col in target_cols:
            if col not in df.columns:
                print(f"  [!] Missing column '{col}' in Parquet, filling with NaN.")
                df[col] = None
                
        # 4. 只保留目标列并确保顺序一致
        df_for_db = df[target_cols]
        
        # 5. 写入 DuckDB
        db.insert_features(df_for_db)
        print("Features migration completed (via Pandas alignment).")
    except Exception as e:
        print(f"Features migration failed: {e}")
            # 回退到 Pandas 模式以便调试
            # df = pd.read_parquet(FEATURES_PARQUET_PATH)
            # db.insert_features(df)

def migrate_sqlite_to_duckdb():
    print("\nStarting migration: SQLite -> DuckDB (news_labeled)...")
    # 注意：用户提到新闻分析项目在 C:\AI_Workplace
    # 需检查 sqlite 数据库是否存在
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"SQLite DB not found at {SQLITE_DB_PATH}, skipping news migration.")
        return

    db = QuantDBManager()
    try:
        # 连接 SQLite 提取数据
        sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
        # 假设表名是 sentiment_analysis
        query = "SELECT * FROM sentiment_analysis"
        df_news = pd.read_sql_query(query, sqlite_conn)
        sqlite_conn.close()

        if df_news.empty:
            print("ℹ️ SQLite sentiment_analysis table is empty.")
            return

        # 转换并插入 DuckDB (字段映射需对齐)
        # 这里建议根据实际 SQLite 字段做对齐逻辑
        print(f"Loaded {len(df_news)} news records from SQLite.")
        
        # 简单直接插入 (由于字段不一定 100% 对应，可能需要 mapping)
        with db.get_connection() as conn:
            # 这是一个示例插入，实际需按 schema.sql 对齐
            # conn.execute("INSERT INTO news_labeled SELECT * FROM df_news")
            pass
            
        print("✅ News migration placeholder finished (Columns mapping required for real sync).")
    except Exception as e:
        print(f"❌ News migration failed: {e}")

if __name__ == "__main__":
    migrate_parquet_to_duckdb()
    # migrate_sqlite_to_duckdb()
