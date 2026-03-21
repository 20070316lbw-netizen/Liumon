@echo off
REM ======================================================
REM Liumon 启动器
REM ======================================================

cd /d C:\Users\lbw15\Desktop\Liumon

REM 激活虚拟环境
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM 启动主程序
python main.py

pause
