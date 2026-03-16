# Alpha Genome: 投研管线逻辑审计与重构报告 (v3.0)

**Goal**: 针对 Alpha Genome v3.0 全链路进行深度审计，消除前瞻偏差（Leakage）、逻辑错位（Alignment）及配置冗余，确保研究结论的科学严谨性。

**Verdict**: **有效 (Fixed)**。通过 6 项核心代码重构，已彻底消除用户指出的所有高危逻辑漏洞。

## Data & Methodology
- **审计范围**: 特征工程 (`build_enhanced_features_cn.py`)、权重逻辑 (`cap_aware_weights.py`)、评估系统 (`eval_factor_leaderboard.py`)。
- **重构手段**: 
    - 引入后复权 (`adjustflag=3`) 消除前瞻偏差。
    - 显式排序分组修复 `shift(-1)` 错位风险。
    - 将回归重要性升级为 LambdaRank 排序重要性。

## Key Metrics (Fix Summary)
| 漏洞类型 | 修复状态 | 影响范围 | 修复方案 |
| :--- | :--- | :--- | :--- |
| **前瞻偏差** | 已修复 | Data Fetching | 切换为后复权 (Backward Adjustment) |
| **标签错位** | 已修复 | Feature Eng | 显式 `sort` + `groupby` 替代全局 `shift` |
| **幽灵因子** | 已清理 | Weights Config | 物理剔除 ROE/np_growth 无效权重 |
| **方向错误** | 已校准 | Registry | 动量方向校准为 -1 (反转效应) |
| **数据污染** | 已拦截 | Exp Tracking | JSON NaN 递归清洗转换 |
| **入口分裂** | 已整合 | UI/UX | 统一主入口 `main_hub.py` |

## Insights
- **前瞻偏差的隐蔽性**: 前复权虽然在实盘分析中直观，但在回测历史时会因“未来送转信息”导致历史价格失真。切换为后复权是确保 Point-in-time 严谨性的唯一途径。
- **排序一致性**: 因子排行榜如果使用 Regression 目标而在主模型使用 LambdaRank 目标，会导致评估准则分裂。统一为排序目标 (NDCG/Gain) 才能实现真正的“优中选优”。

## Action
- [Code] 已完成 `build_enhanced_features_cn.py`, `cap_aware_weights.py`, `main_hub.py` 等 6 个核心文件的重构。
- [Data] 维持后台“后复权”增量抓取，待数据 100% 覆盖后重新生成特征。
- [Repo] 同步更新此报告至 `reports/` 目录。
