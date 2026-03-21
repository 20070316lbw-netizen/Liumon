@echo off
REM ======================================================
REM 快速查看数据
REM ======================================================

cd /d C:\Users\lbw15\Desktop\Liumon

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python view_data.py

pause
