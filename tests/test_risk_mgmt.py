import pytest
from liumon.core.risk_mgmt import RiskManager

def test_risk_scaling():
    rm = RiskManager(target_vol=0.15)
    
    # 低波动率应放大仓位
    scale_low = rm.calculate_position_scale(0.05)
    assert scale_low > 1.0
    
    # 高波动率应缩小仓位
    scale_high = rm.calculate_position_scale(0.30)
    assert scale_high < 1.0
    
    # 杠杆上限保护
    scale_extreme = rm.calculate_position_scale(0.001)
    assert scale_extreme <= 1.2

def test_drawdown_protection():
    rm = RiskManager(max_drawdown_limit=0.20)
    
    # 正常回撤
    assert rm.check_drawdown_protection(0.05) == 1.0
    
    # 极端回撤保护触发
    assert rm.check_drawdown_protection(0.25) == 0.0
