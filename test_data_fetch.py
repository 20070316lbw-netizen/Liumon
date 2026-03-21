"""
Liumon 数据抓取测试脚本
目的：测试 20 天连续数据获取的稳定性
"""

import os
import sys
from datetime import datetime, date
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import CN_DIR, PRICE_DIR

def test_data_fetch():
    """测试数据抓取"""
    print("\n" + "="*60)
    print("  测试数据抓取功能")
    print("="*60)
    
    # 运行数据抓取
    print("\n开始抓取数据...")
    print("这可能需要几分钟，请耐心等待...")
    print("\n" + "-"*60)
    
    # 导入数据抓取模块
    sys.path.append(os.path.join(os.path.dirname(__file__), 'liumon', 'data'))
    import data_fetch_cn
    
    try:
        # 运行主函数
        data_fetch_cn.main()
        print("-"*60)
        print("✅ 数据抓取完成")
        return True
    except Exception as e:
        print("-"*60)
        print(f"❌ 数据抓取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_quality():
    """检查数据质量"""
    print("\n" + "="*60)
    print("  数据质量检查")
    print("="*60)
    
    features_path = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
    
    if not os.path.exists(features_path):
        print("❌ 特征数据不存在")
        return False
    
    try:
        df = pd.read_parquet(features_path)
        
        # 获取最新日期
        latest_date = df['date'].max()
        today = date.today()
        
        print(f"\n📊 数据统计:")
        print(f"   - 总数据量: {len(df):,} 行")
        print(f"   - 最新日期: {latest_date}")
        print(f"   - 今天日期: {today}")
        print(f"   - 数据延迟: {(pd.to_datetime(today) - pd.to_datetime(latest_date)).days} 天")
        
        # 检查最近 20 天的数据
        latest_df = df[df['date'] >= (latest_date - pd.Timedelta(days=30))]
        dates = latest_df['date'].unique()
        dates.sort()
        
        print(f"\n📅 最近的交易日 (最多显示 20 个):")
        for i, d in enumerate(dates[-20:], 1):
            stock_count = len(latest_df[latest_df['date'] == d])
            print(f"   {i:2d}. {d.date()} - {stock_count} 只股票")
        
        # 检查数据完整性
        print(f"\n🔍 数据完整性:")
        null_counts = df.isnull().sum()
        high_null = null_counts[null_counts > len(df) * 0.1]
        
        if len(high_null) > 0:
            print(f"   ⚠️ 以下列缺失数据超过 10%:")
            for col, count in high_null.items():
                print(f"      - {col}: {count:,} ({count/len(df)*100:.1f}%)")
        else:
            print(f"   ✅ 所有列数据完整性良好")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
        return False

def main():
    """主测试流程"""
    print("\n" + "#"*60)
    print("  LIUMON 数据抓取测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)
    
    # 测试 1：数据抓取
    fetch_success = test_data_fetch()
    
    # 测试 2：数据质量检查
    if fetch_success:
        quality_success = check_data_quality()
    else:
        print("\n⚠️ 数据抓取失败，跳过质量检查")
        quality_success = False
    
    # 总结
    print("\n" + "="*60)
    print("  测试总结")
    print("="*60)
    
    if fetch_success and quality_success:
        print("✅ 数据抓取和质量检查都通过！")
        print("\n建议:")
        print("1. 每天在电脑开机时手动运行一次（双击 run_daily_local.bat）")
        print("2. 连续测试 20 天，观察数据稳定性")
        print("3. 20 天后再考虑配置定时任务")
    else:
        print("⚠️ 测试未完全通过")
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 重试数据抓取")
        print("3. 查看详细错误信息")
    
    print("="*60)

if __name__ == "__main__":
    main()
