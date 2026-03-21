# Liumon 使用指南

## 🚀 快速开始

### 方式 1：使用 Python CLI（推荐）

```bash
# 激活虚拟环境
.\venv\Scripts\activate

# 查看帮助
python cli.py help

# 测试核心功能
python cli.py test

# 抓取最新数据
python cli.py fetch

# 生成选股报告
python cli.py predict

# 运行完整流水线（数据抓取 + 选股）
python cli.py run

# 运行回测
python cli.py backtest

# 训练模型
python cli.py train
```

### 方式 2：使用启动器（更简单）

**双击运行：** `liumon.bat`

然后输入命令，例如：
```
test        测试功能
run         每日运行
predict     生成报告
```

---

## 📅 20 天测试计划

### 每天操作（5 分钟）

```bash
# 方式 A：命令行
python cli.py run

# 方式 B：双击
liumon.bat run
```

### 查看报告

```
reports/report_YYYYMMDD.md
```

---

## 📂 项目结构

```
Liumon/
├── cli.py                    # 统一CLI入口 ⭐
├── liumon.bat               # 启动器（唯一的bat文件）
├── liumon/                  # 核心代码
│   ├── data/               # 数据抓取
│   ├── core/               # 核心引擎
│   └── backtest/           # 回测系统
├── scripts/                 # 独立脚本
│   ├── train.py           # 模型训练
│   ├── backtest.py        # 回测
│   └── daily_report.py    # 每日报告
├── tests/                   # 测试
│   └── test_*.py
├── models/                  # 模型文件
├── data/                    # 数据文件
└── reports/                 # 生成的报告
```

---

## 🔧 Windows 定时任务（可选）

如果你想自动运行，配置任务计划程序：

**程序：** `C:\Users\lbw15\Desktop\Liumon\venv\Scripts\python.exe`  
**参数：** `cli.py run`  
**起始于：** `C:\Users\lbw15\Desktop\Liumon`  
**时间：** 每天 16:00

---

## 💡 常见操作

### 手动运行每日任务

```bash
python cli.py run
```

### 测试数据抓取

```bash
python cli.py fetch
```

### 只生成报告（不抓数据）

```bash
python cli.py predict
```

### 运行完整测试

```bash
python cli.py test
```

---

## 🎯 命令速查表

| 命令 | 功能 | 耗时 |
|------|------|------|
| `test` | 测试核心功能 | 10s |
| `fetch` | 抓取最新数据 | 2-5min |
| `predict` | 生成选股报告 | 30s |
| `run` | 完整流水线 | 3-6min |
| `backtest` | 运行回测 | 5-10min |
| `train` | 训练模型 | 10-30min |

---

## 📊 输出

### 报告位置
```
reports/report_YYYYMMDD.md
```

### 日志位置
```
reports/daily.log
```

---

更多详细信息请查看 `TEST_PLAN_20_DAYS.md`
