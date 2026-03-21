@echo off
REM ======================================================
REM Liumon CLI Launcher
REM 这是唯一需要的 .bat 文件，其他都用 Python
REM ======================================================

cd /d C:\Users\lbw15\Desktop\Liumon

REM 激活虚拟环境
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM 调用 Python CLI
python cli.py %*

pause
