#!/bin/bash

# =====================================================================
# LIUMON 自动化每日流水线启动脚本
# 描述: 启动数据抓取、模型预测并生成每日格式化报告。
# 适用于每日定时任务 (cron jobs)。
# =====================================================================

set -e

# 获取脚本所在的当前目录为项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

REPORT_DIR="$PROJECT_ROOT/reports"
LOG_FILE="$REPORT_DIR/daily.log"

# 如果 reports 目录不存在则创建
mkdir -p "$REPORT_DIR"

echo "============================================================" >> "$LOG_FILE"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] 启动 LIUMON 每日选股流水线..." >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

# 运行 Python 脚本并将输出记录到日志文件
if python3 scripts/daily_report.py >> "$LOG_FILE" 2>&1; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 流水线执行成功并完成。" >> "$LOG_FILE"
    echo "流水线执行成功。日志见: $LOG_FILE"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ❌ 流水线执行失败！请检查日志。" >> "$LOG_FILE"
    echo "流水线执行失败。请检查日志: $LOG_FILE"
    exit 1
fi
