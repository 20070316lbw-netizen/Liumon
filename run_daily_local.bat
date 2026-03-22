@echo off
REM ======================================================
REM LIUMON 自动化每日流水线启动脚本 (Windows)
REM 描述: 启动数据抓取、模型预测并生成每日格式化报告。
REM 适用于每日定时任务 (Task Scheduler)。
REM ======================================================

setlocal enabledelayedexpansion

REM 获取当前脚本所在目录
set "PROJECT_ROOT=%~dp0"
REM 去除最后的反斜杠
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
cd /d "%PROJECT_ROOT%"

set "PYTHONPATH=%PROJECT_ROOT%"

set "REPORT_DIR=%PROJECT_ROOT%\reports"
set "LOG_FILE=%REPORT_DIR%\daily.log"

REM 如果 reports 目录不存在则创建
if not exist "%REPORT_DIR%" (
    mkdir "%REPORT_DIR%"
)

REM 获取当前时间 (格式化为 YYYY-MM-DD HH:MM:SS)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set "date_str=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%"
set "time_str=%datetime:~8,2%:%datetime:~10,2%:%datetime:~12,2%"

echo ============================================================ >> "%LOG_FILE%"
echo [%date_str% %time_str%] 启动 LIUMON 每日选股流水线... >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

REM 激活虚拟环境 (如果存在)
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM 运行数据抓取和模型预测 (live.py)
echo [%date_str% %time_str%] 正在运行 live.py (数据获取、特征工程、模型预测) >> "%LOG_FILE%"
python scripts\live.py >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%date_str% %time_str%] ❌ live.py 执行失败！请检查日志。 >> "%LOG_FILE%"
    echo live.py 执行失败。请检查日志: "%LOG_FILE%"
    exit /b 1
)

REM 获取当前时间
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set "date_str=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%"
set "time_str=%datetime:~8,2%:%datetime:~10,2%:%datetime:~12,2%"

REM 生成每日格式化报告
echo [%date_str% %time_str%] 正在运行 daily_report.py (生成每日格式化报告) >> "%LOG_FILE%"
python scripts\daily_report.py >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%date_str% %time_str%] ❌ daily_report.py 执行失败！请检查日志。 >> "%LOG_FILE%"
    echo daily_report.py 执行失败。请检查日志: "%LOG_FILE%"
    exit /b 1
)

REM 获取当前时间
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set "date_str=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%"
set "time_str=%datetime:~8,2%:%datetime:~10,2%:%datetime:~12,2%"

echo [%date_str% %time_str%] ✅ 流水线执行成功并完成。生成报告已存入文件夹。 >> "%LOG_FILE%"
echo 流水线执行成功并完成。日志见: "%LOG_FILE%"

endlocal
exit /b 0
