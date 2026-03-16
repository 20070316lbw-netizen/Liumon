import numpy as np

class RiskManager:
    """
    Liumon 风险管理模块
    功能：波动率控制、回撤保护、风险预算管理
    """
    def __init__(self, target_vol=0.15, max_drawdown_limit=0.20):
        self.target_vol = target_vol
        self.max_drawdown_limit = max_drawdown_limit

    def calculate_position_scale(self, current_vol):
        """
        基于波动率目标的仓位缩放 (Target Volatility Scaling)
        """
        if current_vol == 0:
            return 1.0
        scale = self.target_vol / current_vol
        return min(1.2, max(0.0, scale))  # 杠杆上限 1.2

    def check_drawdown_protection(self, current_drawdown):
        """
        最大回撤硬拦截逻辑
        """
        if current_drawdown > self.max_drawdown_limit:
            return 0.0  # 强制空仓保护
        return 1.0
