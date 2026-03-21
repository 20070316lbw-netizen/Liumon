"""
数据库查看工具
查看 Liumon 项目中的数据状态
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CN_DIR, PRICE_DIR

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

def print_section(title):
    """打印分节标题"""
    if RICH_AVAILABLE:
        console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)

def check_features_data():
    """检查特征数据"""
    print_section("特征数据 (cn_features_enhanced.parquet)")
    
    features_path = Path(CN_DIR) / 'cn_features_enhanced.parquet'
    
    if not features_path.exists():
        print(f"❌ 文件不存在: {features_path}")
        return
    
    try:
        df = pd.read_parquet(features_path)
        
        # 基本信息
        print(f"\n📊 基本信息:")
        print(f"   文件大小: {features_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"   总数据量: {len(df):,} 行")
        print(f"   特征数量: {len(df.columns)} 列")
        
        # 日期范围
        min_date = df['date'].min()
        max_date = df['date'].max()
        today = pd.Timestamp.now().date()
        delay_days = (pd.Timestamp(today) - pd.Timestamp(max_date)).days
        
        print(f"\n📅 日期信息:")
        print(f"   最早日期: {min_date.date()}")
        print(f"   最新日期: {max_date.date()}")
        print(f"   今天日期: {today}")
        print(f"   数据延迟: {delay_days} 天")
        
        # 最近交易日统计
        print(f"\n📈 最近 10 个交易日:")
        latest_dates = df['date'].drop_duplicates().sort_values(ascending=False).head(10)
        
        if RICH_AVAILABLE:
            table = Table()
            table.add_column("#", justify="right", style="cyan")
            table.add_column("日期", style="magenta")
            table.add_column("股票数", justify="right", style="green")
            
            for i, date in enumerate(latest_dates, 1):
                count = len(df[df['date'] == date])
                table.add_row(str(i), str(date.date()), f"{count:,}")
            
            console.print(table)
        else:
            print("-" * 60)
            for i, date in enumerate(latest_dates, 1):
                count = len(df[df['date'] == date])
                print(f"   {i:2d}. {date.date()} - {count:,} 只股票")
            print("-" * 60)
        
        # 列信息
        print(f"\n📋 特征列 ({len(df.columns)}):")
        cols = list(df.columns)
        for i in range(0, len(cols), 5):
            print(f"   {', '.join(cols[i:i+5])}")
        
        # 数据质量
        print(f"\n🔍 数据质量:")
        null_counts = df.isnull().sum()
        high_null = null_counts[null_counts > len(df) * 0.1]
        
        if len(high_null) > 0:
            print(f"   ⚠️ 以下列缺失超过 10%:")
            for col, count in high_null.items():
                print(f"      - {col}: {count:,} ({count/len(df)*100:.1f}%)")
        else:
            print(f"   ✅ 所有列数据完整性良好 (缺失率 < 10%)")
        
        return df
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()

def check_macro_data():
    """检查宏观数据"""
    print_section("宏观数据 (macro_regime.parquet)")
    
    macro_path = Path(CN_DIR) / 'macro_regime.parquet'
    
    if not macro_path.exists():
        print(f"❌ 文件不存在: {macro_path}")
        return
    
    try:
        df = pd.read_parquet(macro_path)
        
        print(f"\n📊 基本信息:")
        print(f"   文件大小: {macro_path.stat().st_size / 1024:.2f} KB")
        print(f"   总数据量: {len(df):,} 行")
        
        # 日期范围
        df['date'] = pd.to_datetime(df['date'])
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        print(f"\n📅 日期信息:")
        print(f"   最早日期: {min_date.date()}")
        print(f"   最新日期: {max_date.date()}")
        
        # 市场状态统计
        if 'regime' in df.columns:
            bull_count = (df['regime'] == 1).sum()
            bear_count = (df['regime'] == 0).sum()
            
            print(f"\n📈 市场状态统计:")
            print(f"   牛市天数: {bull_count:,} ({bull_count/len(df)*100:.1f}%)")
            print(f"   熊市天数: {bear_count:,} ({bear_count/len(df)*100:.1f}%)")
        
        # 最近状态
        print(f"\n📊 最近 10 天市场状态:")
        recent = df.sort_values('date', ascending=False).head(10)
        
        if RICH_AVAILABLE:
            table = Table()
            table.add_column("日期", style="cyan")
            table.add_column("状态", style="magenta")
            
            for _, row in recent.iterrows():
                regime = "🐂 Bull" if row['regime'] == 1 else "🐻 Bear"
                table.add_row(str(row['date'].date()), regime)
            
            console.print(table)
        else:
            for _, row in recent.iterrows():
                regime = "Bull" if row['regime'] == 1 else "Bear"
                print(f"   {row['date'].date()} - {regime}")
        
        return df
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")

def check_price_data():
    """检查价格数据"""
    print_section("价格数据 (prices)")
    
    price_dir = Path(PRICE_DIR)
    
    if not price_dir.exists():
        print(f"❌ 目录不存在: {price_dir}")
        return
    
    # 统计价格文件
    price_files = list(price_dir.glob('*.parquet'))
    
    if not price_files:
        print(f"⚠️ 没有找到价格数据文件")
        return
    
    print(f"\n📁 价格文件统计:")
    print(f"   文件数量: {len(price_files)}")
    
    total_size = sum(f.stat().st_size for f in price_files)
    print(f"   总大小: {total_size / (1024*1024):.2f} MB")
    
    # 随机检查几个文件
    print(f"\n📊 随机抽样 5 个文件:")
    import random
    samples = random.sample(price_files, min(5, len(price_files)))
    
    for f in samples:
        try:
            df = pd.read_parquet(f)
            print(f"   {f.name}: {len(df)} 行")
        except:
            print(f"   {f.name}: 读取失败")

def check_models():
    """检查模型文件"""
    print_section("模型文件")
    
    models_dir = Path(PROJECT_ROOT) / 'models'
    
    if not models_dir.exists():
        print(f"❌ 目录不存在: {models_dir}")
        return
    
    model_files = list(models_dir.glob('*.pkl'))
    
    if not model_files:
        print(f"⚠️ 没有找到模型文件")
        return
    
    print(f"\n📦 模型文件:")
    for f in model_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"   {f.name}: {size_mb:.2f} MB")

def check_reports():
    """检查报告文件"""
    print_section("历史报告")
    
    reports_dir = Path(PROJECT_ROOT) / 'reports'
    
    if not reports_dir.exists():
        print(f"❌ 目录不存在: {reports_dir}")
        return
    
    report_files = sorted(reports_dir.glob('report_*.md'), reverse=True)
    
    if not report_files:
        print(f"⚠️ 没有找到报告文件")
        return
    
    print(f"\n📄 报告数量: {len(report_files)}")
    print(f"\n最近 10 份报告:")
    
    if RICH_AVAILABLE:
        table = Table()
        table.add_column("#", justify="right", style="cyan")
        table.add_column("日期", style="magenta")
        table.add_column("文件名", style="green")
        
        for i, f in enumerate(report_files[:10], 1):
            date = f.stem.replace('report_', '')
            table.add_row(str(i), date, f.name)
        
        console.print(table)
    else:
        for i, f in enumerate(report_files[:10], 1):
            date = f.stem.replace('report_', '')
            print(f"   {i:2d}. {date} - {f.name}")

def main():
    """主函数"""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]LIUMON 数据查看器[/bold cyan]\n"
            "[dim]Data Inspector[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n" + "="*60)
        print("  LIUMON 数据查看器")
        print("="*60)
    
    # 检查各类数据
    check_features_data()
    print()
    check_macro_data()
    print()
    check_price_data()
    print()
    check_models()
    print()
    check_reports()
    
    print("\n" + "="*60)
    print("✅ 数据检查完成")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
