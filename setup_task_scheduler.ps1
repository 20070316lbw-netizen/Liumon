# ======================================================
# LIUMON Task Scheduler Auto-Configuration
# ======================================================

# Check admin rights
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "ERROR: Please run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    pause
    exit 1
}

# Configuration
$TaskName = "Liumon_DailyStock"
$Description = "Liumon Quant Stock Selection - Daily at 16:00"
$ScriptPath = "C:\Users\lbw15\Desktop\Liumon\run_daily_local.bat"
$WorkingDir = "C:\Users\lbw15\Desktop\Liumon"
$TriggerTime = "16:00"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LIUMON Task Scheduler Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if script exists
if (-Not (Test-Path $ScriptPath)) {
    Write-Host "ERROR: Cannot find script $ScriptPath" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[1/4] Checking existing task..." -ForegroundColor Yellow
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "WARNING: Task '$TaskName' already exists" -ForegroundColor Yellow
    $Confirm = Read-Host "Delete and recreate? (Y/N)"
    if ($Confirm -eq "Y" -or $Confirm -eq "y") {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "SUCCESS: Old task deleted" -ForegroundColor Green
    } else {
        Write-Host "Operation cancelled" -ForegroundColor Yellow
        pause
        exit 0
    }
}

Write-Host ""
Write-Host "[2/4] Creating trigger (Daily at $TriggerTime)..." -ForegroundColor Yellow
$Trigger = New-ScheduledTaskTrigger -Daily -At $TriggerTime

Write-Host "[3/4] Creating action (Execute run_daily_local.bat)..." -ForegroundColor Yellow
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$ScriptPath`"" -WorkingDirectory $WorkingDir

Write-Host "[4/4] Registering task to Task Scheduler..." -ForegroundColor Yellow
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName `
                       -Description $Description `
                       -Trigger $Trigger `
                       -Action $Action `
                       -Settings $Settings `
                       -RunLevel Highest `
                       -Force

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "SUCCESS: Task created!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Task Details:" -ForegroundColor Cyan
Write-Host "  - Task Name: $TaskName" -ForegroundColor White
Write-Host "  - Schedule: Daily at $TriggerTime" -ForegroundColor White
Write-Host "  - Script: $ScriptPath" -ForegroundColor White
Write-Host ""
Write-Host "How to check:" -ForegroundColor Cyan
Write-Host "  1. Open Task Scheduler (Win+R, type: taskschd.msc)" -ForegroundColor White
Write-Host "  2. Click 'Task Scheduler Library' on the left" -ForegroundColor White
Write-Host "  3. Find '$TaskName' task" -ForegroundColor White
Write-Host ""
Write-Host "How to test:" -ForegroundColor Cyan
Write-Host "  Right-click task -> Run" -ForegroundColor White
Write-Host ""

pause
