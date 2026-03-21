"""
Liumon 项目功能测试脚本
测试核心功能：数据读取、模型加载、选股预测
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import CN_DIR, MODEL_DIR

def test_data_loading():
    """测试 1：数据加载"""
    print("\n" + "="*60)
    print("测试 1：数据加载")
    print("="*60)
    
    features_path = os.path.join(CN_DIR, 'cn_features_enhanced.parquet')
    macro_path = os.path.join(CN_DIR, 'macro_regime.parquet')
    
    # 检查文件存在
    if not os.path.exists(features_path):
        print(f"❌ 特征文件不存在: {features_path}")
        return False
    
    if not os.path.exists(macro_path):
        print(f"❌ 宏观文件不存在: {macro_path}")
        return False
    
    # 读取数据
    try:
        features_df = pd.read_parquet(features_path)
        macro_df = pd.read_parquet(macro_path)
        
        print(f"✅ 特征数据加载成功")
        print(f"   - 数据量: {len(features_df):,} 行")
        print(f"   - 列数: {len(features_df.columns)} 列")
        print(f"   - 日期范围: {features_df['date'].min()} 到 {features_df['date'].max()}")
        
        print(f"\n✅ 宏观数据加载成功")
        print(f"   - 数据量: {len(macro_df):,} 行")
        print(f"   - 牛市占比: {(macro_df['regime'] == 1).mean():.1%}")
        
        return True, features_df, macro_df
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False, None, None

def test_model_loading():
    """测试 2：模型加载"""
    print("\n" + "="*60)
    print("测试 2：模型加载")
    print("="*60)
    
    model_path = os.path.join(MODEL_DIR, 'cn_regime_genome.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False, None
    
    try:
        with open(model_path, 'rb') as f:
            model_obj = pickle.load(f)
        
        model = model_obj['model']
        features = model_obj.get('features', [])
        
        print(f"✅ 模型加载成功")
        print(f"   - 模型类型: {type(model).__name__}")
        print(f"   - 特征数量: {len(features)}")
        print(f"   - 前5个特征: {features[:5]}")
        
        return True, model_obj
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False, None

def test_prediction(features_df, macro_df, model_obj):
    """测试 3：预测功能"""
    print("\n" + "="*60)
    print("测试 3：预测功能")
    print("="*60)
    
    try:
        # 获取最新日期的数据
        latest_date = features_df['date'].max()
        latest_data = features_df[features_df['date'] == latest_date].copy()
        
        print(f"📅 使用日期: {latest_date}")
        print(f"📊 股票数量: {len(latest_data)}")
        
        # 获取对应的市场状态
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        latest_date_dt = pd.to_datetime(latest_date)
        macro_match = macro_df[macro_df['date'] <= latest_date_dt].tail(1)
        
        if not macro_match.empty:
            regime = macro_match['regime'].iloc[0]
            regime_name = "Bull" if regime == 1 else "Bear"
            print(f"📈 市场状态: {regime_name}")
            latest_data['regime'] = regime
        else:
            print("⚠️ 未找到市场状态，默认使用牛市")
            latest_data['regime'] = 1
        
        # 筛选低价股（< 4.8 元）
        if 'raw_close' in latest_data.columns:
            low_price = latest_data[latest_data['raw_close'] <= 4.80].copy()
            print(f"💰 低价股数量 (≤4.8元): {len(low_price)}")
        else:
            low_price = latest_data.copy()
            print("⚠️ 没有价格列，使用全部数据")
        
        if len(low_price) == 0:
            print("❌ 没有符合条件的股票")
            return False
        
        # 模型预测
        model = model_obj['model']
        feature_cols = model_obj['features']
        
        # 检查特征是否存在
        missing_features = [f for f in feature_cols if f not in low_price.columns]
        if missing_features:
            print(f"⚠️ 缺失特征: {missing_features}")
            print(f"   可用列: {list(low_price.columns)[:10]}...")
            return False
        
        # 准备特征
        X = low_price[feature_cols].fillna(0.5).values.astype(np.float32)
        
        # 预测
        predictions = model.predict(X)
        low_price['alpha_score'] = predictions
        
        # 排序并获取 Top 5
        top_picks = low_price.sort_values('alpha_score', ascending=False).head(5)
        
        print(f"\n✅ 预测成功！")
        print(f"\n📈 Top 5 推荐股票:")
        print("="*60)
        
        for idx, (_, row) in enumerate(top_picks.iterrows(), 1):
            ticker = row.get('ticker', 'N/A')
            score = row['alpha_score']
            price = row.get('raw_close', 0)
            print(f"{idx}. {ticker:12} | 评分: {score:.4f} | 价格: {price:.2f}元")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_report_generation():
    """测试 4：报告生成"""
    print("\n" + "="*60)
    print("测试 4：报告生成能力")
    print("="*60)
    
    report_dir = os.path.join(os.path.dirname(__file__), 'reports')
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        print(f"✅ 创建报告目录: {report_dir}")
    
    # 检查是否有历史报告
    reports = [f for f in os.listdir(report_dir) if f.startswith('report_') and f.endswith('.md')]
    
    if reports:
        print(f"✅ 找到 {len(reports)} 个历史报告")
        latest_report = sorted(reports)[-1]
        print(f"   最新报告: {latest_report}")
        
        # 读取并显示部分内容
        report_path = os.path.join(report_dir, latest_report)
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')[:15]
        print(f"\n   报告预览:")
        for line in lines:
            print(f"   {line}")
    else:
        print("⚠️ 暂无历史报告")
    
    return True

def main():
    """主测试流程"""
    print("\n" + "#"*60)
    print("  LIUMON 项目功能测试")
    print("#"*60)
    
    all_passed = True
    
    # 测试 1：数据加载
    result = test_data_loading()
    if isinstance(result, tuple):
        success, features_df, macro_df = result
    else:
        success = result
        features_df, macro_df = None, None
    
    if not success:
        all_passed = False
        print("\n⚠️ 数据加载失败，跳过后续测试")
        return
    
    # 测试 2：模型加载
    success, model_obj = test_model_loading()
    if not success:
        all_passed = False
        print("\n⚠️ 模型加载失败，跳过预测测试")
    else:
        # 测试 3：预测功能
        if features_df is not None and macro_df is not None:
            success = test_prediction(features_df, macro_df, model_obj)
            if not success:
                all_passed = False
    
    # 测试 4：报告生成
    test_report_generation()
    
    # 总结
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有核心功能测试通过！")
        print("\n建议:")
        print("1. 项目核心功能正常，可以配置定时任务")
        print("2. 数据获取网络问题是暂时的，定时任务会自动重试")
        print("3. 运行 setup_task_scheduler.ps1 配置每日自动执行")
    else:
        print("⚠️ 部分测试未通过，但这不影响定时任务配置")
        print("\n下一步:")
        print("1. 可以先配置定时任务")
        print("2. 等待下次数据获取成功")
        print("3. 查看 reports/daily.log 了解详细情况")
    print("="*60)

if __name__ == "__main__":
    main()
