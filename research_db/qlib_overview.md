# 📦 Microsoft Qlib 仓库研究概览

> **克隆路径：** `C:\Users\lbw15\Desktop\QLib\qlib`  
> **克隆方式：** `--depth=1`（浅克隆，最新快照）  
> **官方仓库：** https://github.com/microsoft/qlib  
> **研究日期：** 2026-03-22

---

## 一、Qlib 是什么？

**Qlib** 是微软亚洲研究院开源的**AI量化投资平台**，专注于：

- 🧠 将 **机器学习 / 深度学习** 与量化投资流程深度结合
- 📊 覆盖从**数据处理 → 因子挖掘 → 模型训练 → 回测 → 在线服务**的完整链路
- 🌍 支持 A 股、美股等多市场数据（官方内置 A 股数据工具）

---

## 二、仓库目录结构

```
qlib/
├── qlib/                    ← 核心库
│   ├── data/                ← 数据层：Handler、Dataset、Feature Expression
│   ├── model/               ← 模型基类与集成：base.py, trainer.py, ens/
│   ├── strategy/            ← 交易策略：信号 → 持仓决策
│   ├── backtest/            ← 回测引擎：模拟撮合、持仓管理
│   ├── contrib/             ← 社区贡献：各类模型、在线服务、策略
│   │   ├── model/           ← LightGBM, LSTM, GRU, Transformer 等模型实现
│   │   ├── evaluate.py      ← IC、ICIR、超额收益等评估函数
│   │   ├── strategy/        ← TopkDropout、WeightOptimizer 等策略
│   │   └── online/          ← 在线服务框架
│   ├── workflow/            ← 工作流：R（实验记录）、Task 管理
│   ├── rl/                  ← 强化学习模块（订单执行优化）
│   ├── utils/               ← 工具函数：日历、缓存、序列化
│   ├── config.py            ← 全局配置系统（C 对象）
│   └── __init__.py          ← 入口，暴露 init()
│
├── examples/                ← 示例
│   ├── benchmarks/          ← 多模型基准对比（LightGBM, LSTM, SFM 等）
│   ├── tutorial/            ← 入门教程
│   ├── workflow_by_code.py  ← 纯代码工作流示例
│   ├── workflow_by_code.ipynb
│   ├── portfolio/           ← 组合优化示例
│   └── rl/                  ← RL 订单执行示例
│
├── docs/                    ← 文档（Sphinx）
│   ├── component/           ← 各组件说明
│   └── introduction/        ← 架构介绍
│
└── tests/                   ← 测试用例
```

---

## 三、核心模块详解

### 3.1 数据层 (`qlib/data/`)

Qlib 使用**表达式引擎**来定义因子，例如：

```python
from qlib.data import D

# 用表达式定义 20 日动量因子
momentum_20 = D.features(instruments, ["Ref($close, 20) / $close - 1"])

# 读取 A 股数据
df = D.features(["sh600519"], ["$close", "$volume"], start_time="2020-01-01")
```

**Handler 机制**：将原始数据清洗、对齐、归一化封装为标准接口，可与模型无缝对接。

### 3.2 模型层 (`qlib/contrib/model/`)

内置 10+ 种模型：

| 模型 | 类型 | 特点 |
|:---|:---|:---|
| `LGBModel` | LightGBM | 最稳定，适合入门 |
| `LSTMModel` | 深度学习 | 序列建模 |
| `GRUModel` | 深度学习 | 更快的序列模型 |
| `TCNModel` | 深度学习 | 时序卷积网络 |
| `TabnetModel` | Attention | 可解释特征选择 |
| `TFTModel` | Transformer | Temporal Fusion Transformer |
| `HIST` | 图神经网络 | 股票间关系建模 |

### 3.3 策略层 (`qlib/strategy/` + `qlib/contrib/strategy/`)

- **TopkDropout**：选 Top K 股票，随机替换部分持仓以降低换手率（最常用）
- **WeightOptimization**：基于协方差矩阵的 MVO 组合优化

### 3.4 回测引擎 (`qlib/backtest/`)

- 支持**日频 / 分钟频**回测
- 内置**模拟撮合**：order → trade 的完整模拟
- 提供 `analyze_riskreturn_report()`、`generate_report_normal()` 等标准报告生成

### 3.5 工作流 (`qlib/workflow/`)

Qlib 的 **R（Record）系统**类似 MLflow，自动记录：
- 模型超参数
- 评估指标（IC、ICIR、超额收益）
- 回测结果

---

## 四、与 Liumon 的结合点

| Liumon 当前能力 | Qlib 可增强点 | 优先级 |
|:---|:---|:---:|
| baostock + parquet 数据管道 | Qlib Dataset 规范化接口 | ⭐⭐ |
| LightGBM LambdaRank | Qlib `qlib.contrib.model.LGBModel` + 标准 IC 评估 | ⭐⭐⭐ |
| 自写因子 Python 函数 | Qlib 表达式引擎统一管理因子 | ⭐⭐ |
| 自写 backtest.py | Qlib 回测引擎（更精确的撮合） | ⭐⭐⭐ |
| 静态权重选股 | TopkDropout 策略 | ⭐⭐ |

---

## 五、快速上手步骤

```bash
# 1. 安装（需要先激活 venv）
cd C:\Users\lbw15\Desktop\QLib\qlib
pip install numpy cython
pip install -e ".[dev]"

# 2. 下载 A 股数据（Qlib 官方工具）
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data \
    --region cn

# 3. 运行示例工作流
python examples/workflow_by_code.py
```

---

## 六、关键文件速查

| 文件 | 作用 |
|:---|:---|
| [`examples/workflow_by_code.py`](file:///C:\Users\lbw15\Desktop\QLib\qlib\examples\workflow_by_code.py) | 最小工作流示例（入门首选）|
| [`qlib/contrib/evaluate.py`](file:///C:\Users\lbw15\Desktop\QLib\qlib\qlib\contrib\evaluate.py) | IC / ICIR / 超额 / 夏普计算 |
| [`qlib/config.py`](file:///C:\Users\lbw15\Desktop\QLib\qlib\qlib\config.py) | 全局配置系统 |
| [`README.md`](file:///C:\Users\lbw15\Desktop\QLib\qlib\README.md) | 官方文档（含架构图）|

---

*研究文档 · LIUMON Alpha Genome Research Lab · 2026-03-22*
