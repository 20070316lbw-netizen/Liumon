"""
Kronos 本地预测 API
====================
将底层复杂的 Transformer 预测模型包装为一个干净简洁的纯函数，
供顶层智能体 (Agent) 和 CLI 直接调用。

用法：
    from kronos.api import predict_market_trend
    prediction_df = predict_market_trend(historical_df, pred_len=30)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import torch
import warnings

# 屏蔽模型加载可能产生的一些权重不匹配等日志
warnings.filterwarnings('ignore', category=UserWarning)

# 全局缓存储存实例化后的模型，避免重复加载
_kronos_predictor = None

class StatisticalPredictor:
    """
    一个基于线性趋势和滚动波动率的纯量化预测平替类。
    当 Kronos Transformer 加载失败（如 401/404）时，作为稳健回退方案生效。
    """
    def __init__(self):
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False):
        # 1. 计算线性趋势 (使用最后 20 天作为参考)
        window = min(len(df), 20)
        recent_df = df.iloc[-window:]
        
        results = {}
        # 为开高低收计算预测
        for col in self.price_cols:
            if col not in df.columns:
                continue
            y = recent_df[col].values
            x = np.arange(len(y))
            # 线性拟合
            slope, intercept = np.polyfit(x, y, 1)
            # 预测
            future_x = np.arange(len(y), len(y) + pred_len)
            base_pred = intercept + slope * future_x
            
            # 添加基于历史波动的微量噪音
            volatility = recent_df[col].pct_change().std()
            if np.isnan(volatility): volatility = 0.01
            noise = np.random.normal(0, volatility * recent_df[col].iloc[-1] * 0.3, pred_len)
            results[col] = base_pred + noise
            
        # 2. 成交量处理
        avg_vol = df[self.vol_col].mean() if self.vol_col in df.columns else 0.0
        results[self.vol_col] = np.full(pred_len, avg_vol)
        results[self.amt_vol] = np.full(pred_len, avg_vol * results.get('close', [1.0])[0])
        
        return pd.DataFrame(results, index=y_timestamp)

def _get_predictor():
    """懒加载 KronosPredictor 或 StatisticalPredictor 实例"""
    global _kronos_predictor
    if _kronos_predictor is None:
        try:
            from kronos.model.kronos import Kronos, KronosTokenizer, KronosPredictor
            
            print("[Kronos] Initializing prediction model (this might take a few seconds)...")
            
            try:
                # 官方仓库：NeoQuasar/Kronos-Tokenizer-2k + NeoQuasar/Kronos-mini
                tokenizer = KronosTokenizer.from_pretrained("C:/Users/lbw15/Desktop/Dev_Workspace/models/kronos/tokenizer")
                model = Kronos.from_pretrained("C:/Users/lbw15/Desktop/Dev_Workspace/models/kronos/model")
                _kronos_predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
                print("[Kronos] Real model loaded successfully! [Mode: Kronos-mini on CPU]")
            except Exception as e:
                print(f"[Kronos] Load failed: {e}")
                print("[Kronos] Falling back to Statistical Quant Strategy (Robust Mode)...")
                _kronos_predictor = StatisticalPredictor()
            
        except Exception as e:
            print(f"[Kronos] Critical initialization error: {e}")
            print("[Kronos] Critical fallback to Statistical Quant Strategy...")
            _kronos_predictor = StatisticalPredictor()
                
    return _kronos_predictor


def predict_market_trend(
    df: pd.DataFrame, 
    pred_len: int = 30,
    temperature: float = 1.0,
    top_p: float = 0.9,
    sample_count: int = 1
) -> pd.DataFrame:
    """接收历史 K 线数据，输出未来预测走势。"""
    
    if df.empty:
        raise ValueError("Historical data is empty.")
        
    # --- V7.4 格式标准化 (Index -> Series) ---
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    predictor = _get_predictor()
    
    # 强制克隆原始索引以便后续使用
    x_timestamp_series = pd.Series(df.index, name='date')
    
    last_date = df.index[-1]
    y_timestamp_index = pd.date_range(start=last_date + timedelta(days=1), periods=pred_len, freq='B')
    y_timestamp_series = pd.Series(y_timestamp_index, name='date')
    
    # 准备传给模型的 DF：重置索引并生成 'date' 列
    df_for_model = df.copy()
    if 'date' not in df_for_model.columns:
        df_for_model = df_for_model.reset_index().rename(columns={df.index.name if df.index.name else 'index': 'date'})

    # --- V13.0+ 自适应 Ensemble 采样 (Boundary-Aware) ---
    # 先跑 3 轮，若结果 Z 落在决策边界 ±0.1 的纠结区间内，追加 2 轮精准采样
    initial_count = 3
    valid_predictions = []
    boundaries = [0.5, 1.0, 1.5]  # Z-Score 决策边界
    epsilon = 0.1                  # 纠结区间宽度
    noise_std = 0.0309             # 预采样边界判定用的噪声估计
    
    print(f"[Kronos] Predicting next {pred_len} steps from {last_date.date()} [Mode: Adaptive Ensemble]")
    
    # 第一阶段：必跑 3 轮
    for i in range(initial_count):
        print(f"  Sampling {i+1}/{initial_count}...")
        try:
            sample_df = predictor.predict(
                df=df_for_model.set_index('date'), 
                x_timestamp=x_timestamp_series, 
                y_timestamp=y_timestamp_series,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False
            )
            if sample_df is not None and not sample_df.empty:
                valid_predictions.append(sample_df)
        except Exception as e:
            print(f"  ❌ Sampling error: {e}")
            
    if not valid_predictions:
        print("[Kronos] All initial samplings failed.")
        return None

    # 计算初步结果用于边界判定
    temp_df = pd.concat(valid_predictions).groupby(level=0).mean()
    start_p = temp_df.iloc[0]['close']
    end_p = temp_df.iloc[-1]['close']
    r_3 = (end_p / start_p) - 1.0
    z_3 = abs(r_3 / noise_std)
    
    # 边界纠结判定 (Boundary Contention Check)
    near_boundary = any(abs(z_3 - b) < epsilon for b in boundaries)
    
    if near_boundary and len(valid_predictions) == initial_count:
        print(f"[Kronos] Boundary detected (Z={z_3:.2f}). Running 2 additional samples for precision...")
        for i in range(2):
            print(f"  Extra Sampling {i+4}/5...")
            try:
                sample_df = predictor.predict(
                    df=df_for_model.set_index('date'), 
                    x_timestamp=x_timestamp_series, 
                    y_timestamp=y_timestamp_series,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                    verbose=False
                )
                if sample_df is not None and not sample_df.empty:
                    valid_predictions.append(sample_df)
            except Exception as e:
                print(f"  ❌ Extra sampling error: {e}")
    else:
        print(f"[Kronos] Signal clear (Z={z_3:.2f}). Skipping extra samplings.")

    # 最终结果合成
    prediction_df = pd.concat(valid_predictions).groupby(level=0).mean()
    
    # --- V13.4 5-Step Logic: 计算收益均值与波动 (Step 2 & 3) ---
    all_returns = []
    # 抽取区间预测
    max_prices = []
    min_prices = []
    
    for df in valid_predictions:
        s_p = df.iloc[0]['close']
        e_p = df.iloc[-1]['close']
        all_returns.append((e_p / s_p) - 1.0)
        max_prices.append(df['high'].max())
        min_prices.append(df['low'].min())
    
    mean_return = float(np.mean(all_returns))
    std_return = float(np.std(all_returns))
    
    # 获取相对振幅 (区间长度 / 基准价格)
    # 取均值来消除单次采样的极端 outliers
    avg_max = float(np.mean(max_prices))
    avg_min = float(np.mean(min_prices))
    start_price = float(temp_df.iloc[0]['close'])
    predicted_range_pct = (avg_max - avg_min) / start_price
    
    # 注入元数据 (Step 4 & 5 的基础)
    prediction_df.attrs = {
        'mean_return': mean_return,
        'std_return': std_return,
        'model_uncertainty': std_return,  # 保持向下兼容
        'predicted_max': avg_max,
        'predicted_min': avg_min,
        'predicted_range_pct': predicted_range_pct
    }
    
    print(f"[Kronos] Adaptive Ensemble completed ({len(valid_predictions)} samples).")
    print(f"         Mean Return: {mean_return:.2%}, Std: {std_return:.2%}, Range: {predicted_range_pct:.2%}")
    
    return prediction_df
