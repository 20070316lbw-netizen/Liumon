# 清理脚本 - 删除多余的 bat 文件

## 保留的文件

✅ **liumon.bat** - 唯一的启动器
✅ **cli.py** - Python CLI 工具
✅ **USAGE.md** - 使用指南

## 要删除的文件

可以删除这些 bat 文件（功能已整合到 cli.py）：

```bash
rm setup_env.bat
rm test_core.bat
rm test_pipeline.bat
rm test_data_fetch.bat
rm run_manual.bat
rm run_daily_local.bat
rm install_missing_deps.bat
rm launch_ghostty.bat
```

或者在 PowerShell 中一次性删除：

```powershell
Remove-Item setup_env.bat, test_core.bat, test_pipeline.bat, test_data_fetch.bat, run_manual.bat, run_daily_local.bat, install_missing_deps.bat, launch_ghostty.bat
```

## 同时删除的辅助文件

```bash
rm ghostty_config_example
rm GHOSTTY_GUIDE.md
rm DEPLOYMENT_GUIDE.md
rm QUICK_START_GUIDE.md
```

保留：
- ✅ TEST_PLAN_20_DAYS.md（测试计划）
- ✅ USAGE.md（使用指南）
- ✅ README_zh.md（项目说明）

## Git 提交

```bash
# 删除文件
rm setup_env.bat test_*.bat run_*.bat install_*.bat launch_*.bat
rm ghostty_config_example GHOSTTY_GUIDE.md DEPLOYMENT_GUIDE.md QUICK_START_GUIDE.md

# 添加新文件
git add cli.py liumon.bat USAGE.md

# 提交
git commit -m "重构：统一为Python CLI工具，简化项目结构"

# 推送
git push
```
