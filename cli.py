"""
Liumon CLI - 统一命令行工具
用法: python cli.py <command> [options]

命令:
  test         测试核心功能
  fetch        抓取最新数据
  predict      生成选股报告
  backtest     运行回测
  train        训练模型
  run          运行完整流水线（fetch + predict）
"""

import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def print_banner(title):
    """打印标题横幅"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def cmd_test():
    """测试核心功能"""
    from test_core_functions import main as test_main
    test_main()

def cmd_fetch():
    """抓取最新数据"""
    print_banner("LIUMON 数据抓取")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 抓取A股数据
    print("[1/2] 抓取A股行情数据...")
    sys.path.append(os.path.join(os.path.dirname(__file__), 'liumon', 'data'))
    import data_fetch_cn
    data_fetch_cn.main()
    
    # 抓取宏观数据
    print("\n[2/2] 抓取宏观状态数据...")
    import data_fetch_macro
    data_fetch_macro.main()
    
    print(f"\n✅ 数据抓取完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def cmd_predict():
    """生成选股报告"""
    print_banner("LIUMON 选股预测")
    sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
    import daily_report
    daily_report.main()

def cmd_backtest():
    """运行回测"""
    print_banner("LIUMON 策略回测")
    sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
    import backtest
    backtest.main()

def cmd_train():
    """训练模型"""
    print_banner("LIUMON 模型训练")
    sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
    import train
    train.main()

def cmd_run():
    """运行完整流水线"""
    print_banner("LIUMON 完整流水线")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # 1. 抓取数据
        cmd_fetch()
        
        # 2. 生成报告
        print("\n")
        cmd_predict()
        
        print("\n" + "="*60)
        print("✅ 流水线执行成功")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 流水线执行失败: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def show_help():
    """显示帮助信息"""
    print(__doc__)
    print("\n示例:")
    print("  python cli.py test      # 测试核心功能")
    print("  python cli.py fetch     # 抓取数据")
    print("  python cli.py predict   # 生成选股报告")
    print("  python cli.py run       # 运行完整流水线")
    print("  python cli.py backtest  # 运行回测")
    print("  python cli.py train     # 训练模型")

def main():
    """主入口"""
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    commands = {
        'test': cmd_test,
        'fetch': cmd_fetch,
        'predict': cmd_predict,
        'backtest': cmd_backtest,
        'train': cmd_train,
        'run': cmd_run,
        'help': show_help,
        '--help': show_help,
        '-h': show_help,
    }
    
    if command in commands:
        try:
            commands[command]()
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断操作")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 执行失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"❌ 未知命令: {command}")
        print("运行 'python cli.py help' 查看帮助")
        sys.exit(1)

if __name__ == "__main__":
    main()
