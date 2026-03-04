"""
分类任务评估逻辑
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

def evaluate(y_true, y_pred):
    """
    对分类任务进行评估
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 包含各种评估指标的字典
    """
    
    # 检查是二分类还是多分类
    unique_classes = np.unique(y_true)
    is_multiclass = len(unique_classes) > 2
    
    # 设定 average 参数。多分类用 'macro'，二分类用 'binary' 或 'macro' 都可以，这里统一 'macro' 更稳妥，或者对二分类用默认
    avg_method = 'macro' if is_multiclass else 'binary'
    
    report_dict = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=avg_method, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average=avg_method, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, average=avg_method, zero_division=0), 4),
        "is_multiclass": bool(is_multiclass),
        "classes": [str(c) for c in unique_classes]
    }
    
    # 混淆矩阵 (将 ndarray 转为 list 以便 json 序列化)
    cm = confusion_matrix(y_true, y_pred)
    report_dict["confusion_matrix"] = cm.tolist()
    
    # 详细的分类报告（每个类别的 p, r, f1）
    detailed_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # 整理一下 detailed_report，保留到小数点后 4 位
    cleaned_detailed = {}
    for key, value in detailed_report.items():
        if isinstance(value, dict):
            cleaned_detailed[key] = {k: round(v, 4) if isinstance(v, float) else v for k, v in value.items()}
        else:
            cleaned_detailed[key] = round(value, 4)
            
    report_dict["detailed_report"] = cleaned_detailed
    
    return report_dict
