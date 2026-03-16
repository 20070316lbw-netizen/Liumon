import io
import pandas as pd
from datetime import datetime, timedelta
from crawlers.data_gateway import gateway
from kronos.api import predict_market_trend

# Kronos 模型训练时的固定上下文窗口长度 (context_length = 84 个交易日)
# 不同市场节假日导致 A 股实际返回交易日数量不同，必须统一
_KRONOS_SEQ_LEN = 84

class KronosEngine:
    """
    底层数学量化引擎的封装层：负责拉取历史OHLCV数据并驱动基础大语言/统计模型生成预测曲线。
    """
    
    @staticmethod
    def get_raw_prediction(ticker: str, target_date: str, pred_len: int = 30) -> dict:
        """
        获取原始预测数据，不含 LLM 文字解析
        @param target_date: YYYY-MM-DD，预测起点
        @return: {"z_score": float, "expected_return": float, "uncertainty": float}
        """
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            # 提取最近 150 天数据，足够裁出 84 个交易日的窗口
            start_dt = target_dt - timedelta(days=150)
            start_date_str = start_dt.strftime("%Y-%m-%d")
            fetch_end_date = target_dt.strftime("%Y-%m-%d")
        except ValueError:
            target_dt = datetime.now()
            start_dt = target_dt - timedelta(days=150)
            start_date_str = start_dt.strftime("%Y-%m-%d")
            fetch_end_date = target_dt.strftime("%Y-%m-%d")
            
        raw_data_str = gateway.get_stock_data(ticker, start_date_str, fetch_end_date)
        
        lines = raw_data_str.strip().split('\n')
        csv_lines = [line for line in lines if not line.startswith('#') and line.strip()]
        
        if not csv_lines or "Error" in raw_data_str or "No data" in raw_data_str:
             raise ValueError(f"Failed to fetch historical data for {ticker}. Detail: {raw_data_str}")
             
        csv_str = '\n'.join(csv_lines)
        df = pd.read_csv(io.StringIO(csv_str), index_col="Date", parse_dates=True)
        
        df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Volume": "volume"
        }, inplace=True)
        
        # ── 【Tensor 修复 v2】固定输入序列长度至 _KRONOS_SEQ_LEN ──────────
        # 不同股票/市场的实际交易日数量不一致，predictor 期望固定维度。
        # 超出则截取最近 N 行；不足则用最早一行向前填充（保守占位）。
        if len(df) > _KRONOS_SEQ_LEN:
            df = df.iloc[-_KRONOS_SEQ_LEN:]
        elif len(df) < _KRONOS_SEQ_LEN:
            pad_rows = _KRONOS_SEQ_LEN - len(df)
            pad_df = pd.DataFrame(
                [df.iloc[0].to_dict()] * pad_rows,
                index=pd.date_range(
                    end=df.index[0] - pd.Timedelta(days=1),
                    periods=pad_rows,
                    freq='B'
                )
            )
            df = pd.concat([pad_df, df])
        # ─────────────────────────────────────────────────────────────────
        
        # 调用底层统一预测接口（基于集成采样，包含 z-score 边界判定逻辑）
        prediction_df = predict_market_trend(df, pred_len=pred_len)
        
        if prediction_df is None or prediction_df.empty:
             raise RuntimeError("Kronos engine returned empty prediction.")
             
        mean_ret = prediction_df.attrs.get('mean_return', 0.0)
        std_ret = prediction_df.attrs.get('std_return', 0.0309)  # 默认降级波动率
        pred_range = prediction_df.attrs.get('predicted_range_pct', 0.0)
        pred_max = prediction_df.attrs.get('predicted_max', 0.0)
        pred_min = prediction_df.attrs.get('predicted_min', 0.0)
        
        # 计算 Regime Strength (原 Z-Score)，设定一个最低噪声地板防止极高杠杆
        noise_floor = 0.005 
        regime_strength = float(mean_ret / max(std_ret, noise_floor))
        
        return {
            "expected_return": float(mean_ret),
            "uncertainty": float(std_ret),
            "z_score": regime_strength,
            "regime_strength": regime_strength,
            "predicted_range_pct": float(pred_range),
            "predicted_max": float(pred_max),
            "predicted_min": float(pred_min)
        }
