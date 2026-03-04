"""
回归任务评估逻辑
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np

def evaluate(y_true, y_pred):
    """
    对回归任务进行评估
    :param y_true: 真实数值
    :param y_pred: 预测数值
    :return: 包含各种评估指标的字典
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    
    report_dict = {
        "mse": round(mse, 4),             # 均方误差 (Mean Squared Error)
        "rmse": round(rmse, 4),           # 均方根误差 (Root Mean Squared Error)
        "mae": round(mae, 4),             # 平均绝对误差 (Mean Absolute Error)
        "r2_score": round(r2, 4),         # 决定系数 (R²)，越接近1越好
        "explained_variance": round(evs, 4) # 解释方差分，类似R²
    }
    
    # 还可以加一些基本的统计信息，比如误差的最大值、最小值
    errors = y_pred - y_true
    report_dict["error_stats"] = {
        "max_error": round(float(np.max(np.abs(errors))), 4),
        "mean_error": round(float(np.mean(errors)), 4)
    }
    
    return report_dict
