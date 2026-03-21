# QQ Quantitative Lab: DuckDB 统一数据中台

## 1. 架构定位
本模块实现了基于 **DuckDB** 的轻量级、高性能数据中台，旨在替代离散的 Parquet 文件与传统的 SQLite 数据库。服务于：
- 📈 行情数据存储（A股/美股）
- 🧬 因子特征截面（月末采样）
- 🤖 新闻标注与情感得分
- ⚖️ 实验追踪与 Alpha 得分

## 2. 核心文件
- `schema.sql`: 数据库建表 DDL。所有日期字段统一使用 `DATE` 类型。
- `db_manager.py`: 统一操作接口，支持批量 `INSERT OR IGNORE` 及加速查询。
- `migrate.py`: 数据迁移工具（Parquet/SQLite -> DuckDB）。

## 3. 数据库连接
- **路径定义**: 统一受控于 `src/config.py` 中的 `DB_PATH`。
- **默认位置**: `C:\QQ_Quant_DB\quant_lab.duckdb`

## 4. 表结构说明

### `prices_cn` / `prices_us`
存储日频行情。由于 DuckDB 对 Parquet 的原生支持，历史价格可实现瞬间导入。

### `features_cn`
存储经过中性化与正交化后的因子截面。主键为 `(ticker, date)`，适配月度调仓研究。

### `news_labeled`
存储多智能体标注后的新闻结论。核心字段包括：
- `calibrated_score`: 模型校准得分 [0.0, 1.0]，其中 0.5 为中性基准。
- `reason`: 标注理由（中文）

## 5. 快速开始
```python
from alpharanker.database.db_manager import QuantDBManager

db = QuantDBManager()
# 执行查询
df = db.query_features('2015-01-01', '2015-12-31', index_group='ZZ500')
```
