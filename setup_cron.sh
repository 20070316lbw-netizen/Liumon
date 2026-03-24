#!/bin/bash

# 获取项目根目录绝对路径
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/daily_fetch_and_report_job.py"

# 检查 daily_fetch_and_report_job.py 是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到 $SCRIPT_PATH"
    exit 1
fi
chmod +x "$SCRIPT_PATH"

# 如果 reports 目录不存在则创建
REPORT_DIR="$PROJECT_ROOT/reports"
mkdir -p "$REPORT_DIR"
LOG_FILE="$REPORT_DIR/daily_fetch_and_report_job.log"

# 定义定时任务命令: 每天16:00执行一次
CRON_CMD="0 16 * * * cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT /usr/bin/env python3 $SCRIPT_PATH >> $LOG_FILE 2>&1"

# 检查当前 crontab 是否已经包含了该任务
if crontab -l 2>/dev/null | grep -q "$SCRIPT_PATH"; then
    echo "定时任务已经存在！当前设置如下："
    crontab -l | grep "$SCRIPT_PATH"
else
    # 将现有 crontab 导出到临时文件，然后追加新任务
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ 成功添加定时任务："
    echo "$CRON_CMD"
    echo "脚本将每天自动抓取数据，并在 reports/ 文件夹内严谨地生成当日报告。"
fi
