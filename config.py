import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Central Quantitative Database (DuckDB)
# ============================================================
DB_DIR = r'C:\QQ_Quant_DB'
DB_PATH = os.path.join(DB_DIR, 'quant_lab.duckdb')
os.makedirs(DB_DIR, exist_ok=True)

# ============================================================
# Central Data Repository (shared across all projects)
# ============================================================
DATA_ROOT = r'C:\Data\Market'

# A-Share (CN) Data Paths
CN_DIR = os.path.join(DATA_ROOT, 'cn')
PRICE_DIR = os.path.join(CN_DIR, 'prices')
FUND_DIR = os.path.join(CN_DIR, 'fundamentals')
IND_MAP_PATH = os.path.join(CN_DIR, 'stock_industry_map.parquet')

# Research Logs Repository (Synced to GitHub)
LOG_REPO_DIR = os.path.join(DATA_ROOT, '..', 'Dev_Workspace', 'research_logs_repo', 'QQ-AI-Quant-Lab')
LOG_REPO_REPORTS = os.path.join(LOG_REPO_DIR, 'reports')
LOG_REPO_EXPS = os.path.join(LOG_REPO_DIR, 'experiments')
LOG_SYNC_JSON = os.path.join(LOG_REPO_EXPS, 'experiments_log.json')

# US Stock Data Paths
US_DIR = os.path.join(DATA_ROOT, 'us')
US_PRICE_DIR = os.path.join(US_DIR, 'prices')
US_FUND_DIR = os.path.join(US_DIR, 'fundamentals')

# ============================================================
# Project-local Paths (features, models stay in AlphaRanker)
# ============================================================
FEAT_DIR = os.path.join(BASE_DIR, 'features')
FEAT_PATH = os.path.join(FEAT_DIR, 'panel_features.parquet')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'alpha_ranker.pkl')

# Legacy compat: keep DATA_DIR pointing to project data dir
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Ensures directories exist
for d in [PRICE_DIR, FUND_DIR, US_PRICE_DIR, US_FUND_DIR, FEAT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
