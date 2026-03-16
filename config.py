import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 中央量化数据库 (DuckDB)
# ============================================================
# 如果系统路径存在则使用，否则回退到本地项目目录
DB_DIR = r'C:\QQ_Quant_DB' if os.path.exists(r'C:\QQ_Quant_DB') else os.path.join(BASE_DIR, 'data', 'db')
DB_PATH = os.path.join(DB_DIR, 'quant_lab.duckdb')
os.makedirs(DB_DIR, exist_ok=True)

# ============================================================
# 中央数据仓库 (在所有项目间共享)
# ============================================================
# 如果系统路径存在则使用，否则回退到本地项目数据根目录
DATA_ROOT = r'C:\Data\Market' if os.path.exists(r'C:\Data\Market') else os.path.join(BASE_DIR, 'data')

# A股 (CN) 数据路径
CN_DIR = os.path.join(DATA_ROOT, 'cn')
PRICE_DIR = os.path.join(CN_DIR, 'prices')
FUND_DIR = os.path.join(CN_DIR, 'fundamentals')
IND_MAP_PATH = os.path.join(CN_DIR, 'stock_industry_map.parquet')

# 研究日志仓库 (同步到GitHub)
# 便携：如果找不到自定义仓库，则指向本地 backtests 目录
LOG_REPO_DIR = os.path.join(BASE_DIR, 'backtests')
LOG_REPO_REPORTS = os.path.join(LOG_REPO_DIR, 'reports')
LOG_REPO_EXPS = os.path.join(LOG_REPO_DIR, 'experiments')
LOG_SYNC_JSON = os.path.join(LOG_REPO_EXPS, 'experiments_log.json')

# 美股数据路径
US_DIR = os.path.join(DATA_ROOT, 'us')
US_PRICE_DIR = os.path.join(US_DIR, 'prices')
US_FUND_DIR = os.path.join(US_DIR, 'fundamentals')

# ============================================================
# 项目本地路径 (特征、模型保留在 AlphaRanker 中)
# ============================================================
FEAT_DIR = os.path.join(BASE_DIR, 'features')
FEAT_PATH = os.path.join(FEAT_DIR, 'panel_features.parquet')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'alpha_ranker.pkl')

# 遗留兼容性：保持 DATA_DIR 指向项目数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 确保目录存在
for d in [PRICE_DIR, FUND_DIR, US_PRICE_DIR, US_FUND_DIR, FEAT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
