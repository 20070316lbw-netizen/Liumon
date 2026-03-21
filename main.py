"""
Liumon - 量化交易系统主程序
现代化交互式 CLI
"""

import sys
import os
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    from rich.progress import track
    from InquirerPy import prompt
    from InquirerPy.base.control import Choice
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  推荐安装 rich 和 inquirerpy 以获得更好的体验")
    print("   pip install rich inquirerpy")

console = Console() if RICH_AVAILABLE else None

def check_dependencies():
    """检查依赖是否安装"""
    missing = []
    
    try:
        import baostock
    except ImportError:
        missing.append('baostock')
    
    try:
        import akshare
    except ImportError:
        missing.append('akshare')
    
    if missing:
        if RICH_AVAILABLE:
            console.print(f"\n⚠️  缺少依赖: {', '.join(missing)}", style="bold yellow")
            console.print("请运行: pip install " + " ".join(missing), style="yellow")
        else:
            print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
            print(f"请运行: pip install {' '.join(missing)}")
        
        response = input("\n是否现在安装？ (y/N): ")
        if response.lower() == 'y':
            import subprocess
            for pkg in missing:
                subprocess.run([sys.executable, "-m", "pip", "install", pkg])
            print("\n✅ 依赖安装完成，请重新启动程序")
            sys.exit(0)
        else:
            print("\n部分功能可能无法使用")
            return False
    
    return True

def print_banner():
    """打印欢迎横幅"""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]LIUMON[/bold cyan]\n"
            "[dim]量化选股系统 v1.0[/dim]\n"
            "[dim]Alpha Genome Research Lab[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n" + "="*60)
        print("  LIUMON - 量化选股系统 v1.0")
        print("  Alpha Genome Research Lab")
        print("="*60 + "\n")

def show_menu():
    """显示交互式菜单"""
    if not RICH_AVAILABLE:
        return show_simple_menu()
    
    questions = [
        {
            'type': 'list',
            'name': 'action',
            'message': '请选择操作:',
            'choices': [
                Choice('run', name='🚀 运行完整流水线 (推荐)'),
                Choice('test', name='🧪 测试核心功能'),
                Choice('fetch', name='📥 抓取最新数据'),
                Choice('predict', name='📊 生成选股报告'),
                Choice('reports', name='📁 查看历史报告'),
                Choice('train', name='🎯 训练模型'),
                Choice('exit', name='❌ 退出'),
            ],
            'default': 'run'
        }
    ]
    
    result = prompt(questions)
    return result['action']

def show_simple_menu():
    """简单文本菜单（无依赖）"""
    print("\n请选择操作:")
    print("  1. 运行完整流水线 (推荐)")
    print("  2. 测试核心功能")
    print("  3. 抓取最新数据")
    print("  4. 生成选股报告")
    print("  5. 查看历史报告")
    print("  6. 训练模型")
    print("  0. 退出")
    
    choice = input("\n输入选项 (默认 1): ").strip() or '1'
    
    mapping = {
        '1': 'run', '2': 'test', '3': 'fetch',
        '4': 'predict', '5': 'reports', '6': 'train',
        '0': 'exit'
    }
    
    return mapping.get(choice, 'run')

def cmd_test():
    """测试核心功能"""
    if RICH_AVAILABLE:
        console.rule("[bold cyan]核心功能测试[/bold cyan]")
    
    from test_core_functions import main as test_main
    test_main()

def cmd_fetch():
    """抓取最新数据"""
    if RICH_AVAILABLE:
        console.rule("[bold cyan]数据抓取[/bold cyan]")
    else:
        print("\n" + "="*60)
        print("  数据抓取")
        print("="*60 + "\n")
    
    try:
        # 检查依赖
        try:
            import baostock
        except ImportError:
            print("❌ 缺少 baostock 库，请先安装: pip install baostock")
            return
        
        # 抓取A股数据
        if RICH_AVAILABLE:
            console.print("[1/2] 抓取 A 股行情数据...", style="bold yellow")
        else:
            print("[1/2] 抓取 A 股行情数据...")
        
        sys.path.insert(0, str(PROJECT_ROOT / 'liumon' / 'data'))
        from liumon.data import data_fetch_cn
        data_fetch_cn.main()
        
        # 抓取宏观数据
        if RICH_AVAILABLE:
            console.print("\n[2/2] 抓取宏观状态数据...", style="bold yellow")
        else:
            print("\n[2/2] 抓取宏观状态数据...")
        
        from liumon.data import data_fetch_macro
        data_fetch_macro.main()
        
        if RICH_AVAILABLE:
            console.print("\n✅ 数据抓取完成", style="bold green")
        else:
            print("\n✅ 数据抓取完成")
            
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n❌ 数据抓取失败: {e}", style="bold red")
        else:
            print(f"\n❌ 数据抓取失败: {e}")
        import traceback
        traceback.print_exc()

def cmd_predict():
    """生成选股报告"""
    if RICH_AVAILABLE:
        console.rule("[bold cyan]选股预测[/bold cyan]")
    else:
        print("\n" + "="*60)
        print("  选股预测")
        print("="*60 + "\n")
    
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from scripts import daily_report
    daily_report.main()
    
    # 显示报告路径
    reports_dir = PROJECT_ROOT / 'reports'
    if reports_dir.exists():
        reports = sorted(reports_dir.glob('report_*.md'), reverse=True)
        if reports:
            latest_report = reports[0]
            if RICH_AVAILABLE:
                console.print(f"\n📄 最新报告: [link]{latest_report}[/link]", style="bold green")
            else:
                print(f"\n📄 最新报告: {latest_report}")

def cmd_run():
    """运行完整流水线"""
    if RICH_AVAILABLE:
        console.rule("[bold cyan]完整流水线[/bold cyan]")
    else:
        print("\n" + "="*60)
        print("  完整流水线")
        print("="*60 + "\n")
    
    try:
        # 1. 抓取数据
        cmd_fetch()
        
        # 2. 生成报告
        print("\n")
        cmd_predict()
        
        if RICH_AVAILABLE:
            console.print("\n" + "="*60, style="bold green")
            console.print("✅ 流水线执行成功", style="bold green")
            console.print("="*60, style="bold green")
        else:
            print("\n" + "="*60)
            print("✅ 流水线执行成功")
            print("="*60)
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n❌ 流水线执行失败: {e}", style="bold red")
        else:
            print(f"\n❌ 流水线执行失败: {e}")
        import traceback
        traceback.print_exc()

def cmd_train():
    """训练模型"""
    if RICH_AVAILABLE:
        console.rule("[bold cyan]模型训练[/bold cyan]")
    else:
        print("\n" + "="*60)
        print("  模型训练")
        print("="*60 + "\n")
    
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from scripts import train
    train.main()

def cmd_reports():
    """查看历史报告"""
    if RICH_AVAILABLE:
        console.rule("[bold cyan]历史报告[/bold cyan]")
    
    reports_dir = PROJECT_ROOT / 'reports'
    if not reports_dir.exists():
        print("❌ 报告目录不存在")
        return
    
    reports = sorted(reports_dir.glob('report_*.md'), reverse=True)
    
    if not reports:
        print("❌ 暂无历史报告")
        return
    
    if RICH_AVAILABLE:
        table = Table(title="历史报告列表")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("日期", style="magenta")
        table.add_column("文件名", style="green")
        
        for i, report in enumerate(reports[:20], 1):
            date = report.stem.replace('report_', '')
            table.add_row(str(i), date, report.name)
        
        console.print(table)
    else:
        print("\n历史报告列表:")
        print("-" * 60)
        for i, report in enumerate(reports[:20], 1):
            date = report.stem.replace('report_', '')
            print(f"  {i:2d}. {date} - {report.name}")
        print("-" * 60)
    
    print(f"\n📂 报告目录: {reports_dir}")

def main():
    """主程序入口"""
    try:
        print_banner()
        
        # 检查依赖
        check_dependencies()
        
        # 交互模式
        while True:
            action = show_menu()
            
            if action == 'exit':
                if RICH_AVAILABLE:
                    console.print("\n👋 再见！", style="bold yellow")
                else:
                    print("\n👋 再见！")
                break
            
            # 执行命令
            commands = {
                'test': cmd_test,
                'fetch': cmd_fetch,
                'predict': cmd_predict,
                'run': cmd_run,
                'train': cmd_train,
                'reports': cmd_reports,
            }
            
            if action in commands:
                try:
                    commands[action]()
                    
                    # 询问是否继续
                    if RICH_AVAILABLE:
                        cont = input("\n按 Enter 返回主菜单，输入 q 退出: ")
                    else:
                        cont = input("\n按 Enter 继续，输入 q 退出: ")
                    
                    if cont.lower() == 'q':
                        break
                        
                except KeyboardInterrupt:
                    print("\n\n⚠️  操作已取消")
                    continue
                except Exception as e:
                    print(f"\n❌ 执行失败: {e}")
                    import traceback
                    traceback.print_exc()
                    input("\n按 Enter 继续...")
    
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
