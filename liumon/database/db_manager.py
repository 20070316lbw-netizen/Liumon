import duckdb
import pandas as pd
import os
import sys

# 适配不同项目的路径引用
try:
    from config import DB_PATH
except ImportError:
    # 兼容性处理
    DB_PATH = r'C:\QQ_Quant_DB\quant_lab.duckdb'

SCHEMA_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(SCHEMA_DIR, 'schema.sql')

class QuantDBManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_connection(self):
        return duckdb.connect(self.db_path)

    def init_db(self):
        """读取 schema.sql 初始化数据库表结构"""
        if not os.path.exists(SCHEMA_PATH):
            print(f"❌ Schema file not found: {SCHEMA_PATH}")
            return
            
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            sql = f.read()
            
        with self.get_connection() as conn:
            conn.execute(sql)
            print(f"Database initialized at {self.db_path}")

    def insert_prices(self, df):
        """批量写入行情数据 (A股)"""
        if df.empty: return
        with self.get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO prices_cn SELECT * FROM df")
            print(f"Inserted {len(df)} rows into prices_cn")

    def insert_features(self, df):
        """批量写入因子截面"""
        if df.empty: return
        with self.get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO features_cn SELECT * FROM df")
            print(f"Inserted {len(df)} rows into features_cn")

    def insert_news_labeled(self, record):
        """写入单条标注新闻 (Record 为字典或单行 DataFrame)"""
        if isinstance(record, dict):
            df = pd.DataFrame([record])
        else:
            df = record
            
        with self.get_connection() as conn:
            conn.execute("INSERT INTO news_labeled SELECT * FROM df")

    def query_features(self, start_date, end_date, index_group=None):
        """查询因子截面"""
        query = f"SELECT * FROM features_cn WHERE date BETWEEN '{start_date}' AND '{end_date}'"
        if index_group:
            query += f" AND index_group = '{index_group}'"
        
        with self.get_connection() as conn:
            return conn.execute(query).df()

    def query_sentiment(self, ticker, start_date, end_date):
        """查询特定股票在某时段的情感汇总"""
        query = f"""
            SELECT * FROM sentiment_daily 
            WHERE ticker = '{ticker}' 
            AND date BETWEEN '{start_date}' AND '{end_date}'
        """
        with self.get_connection() as conn:
            return conn.execute(query).df()

if __name__ == "__main__":
    db = QuantDBManager()
    db.init_db()
