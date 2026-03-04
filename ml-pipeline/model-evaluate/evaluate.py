"""
通用模型评估模块 — Pipeline 里的评估节点

职责：读取模型文件和测试数据集，自动根据任务类型（分类/回归）执行对应的评估逻辑，并输出报告。

输入：
1. 模型文件 (model.pkl)
2. 测试数据 (test_data.npz)
3. 训练元数据 (train_metadata.json) - 用于获取 task_type
"""
import argparse
import json
import os
import joblib
import numpy as np

import evaluate_classification
import evaluate_regression

def parse_args():
    parser = argparse.ArgumentParser(description="通用模型评估模块")
    
    # 只需要告诉它上游模型放在哪，它自己去那个目录找 model.pkl, test_data.npz 等
    parser.add_argument("--model_dir", type=str, required=True,
                        help="包含模型、测试数据和元数据的目录 (即训练节点的 output_dir)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="评估报告输出目录（留空则直接输出到 model_dir 下）")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_dir = args.model_dir
    output_dir = args.output_dir if args.output_dir else model_dir
    
    print("========== 开始模型评估 ==========")
    print(f"读取目录: {model_dir}")
    
    # 1. 检查并读取元数据（确认任务类型）
    meta_path = os.path.join(model_dir, "train_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到元数据文件: {meta_path}，无法判断任务类型")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    task_type = metadata.get("task_type")
    algorithm = metadata.get("algorithm", "unknown")
    print(f"检测到算法: {algorithm}")
    print(f"检测到任务类型: {task_type}")
    
    # 2. 读取模型
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    print("加载模型...")
    model = joblib.load(model_path)
    
    # 3. 读取测试集
    test_data_path = os.path.join(model_dir, "test_data.npz")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"找不到测试集文件: {test_data_path}")
    print("加载测试数据...")
    test_data = np.load(test_data_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    print(f"测试集大小: {X_test.shape[0]} 条样本")
    
    # 4. 让模型做题（预测）
    print("开始预测...")
    y_pred = model.predict(X_test)
    
    # 5. 路由到对应的评估逻辑
    print("开始计算评估指标...")
    if task_type == "classification":
        report = evaluate_classification.evaluate(y_test, y_pred)
    elif task_type == "regression":
        report = evaluate_regression.evaluate(y_test, y_pred)
    else:
        raise ValueError(f"暂不支持的任务类型: {task_type}")
        
    # 6. 整合并输出报告
    final_report = {
        "algorithm": algorithm,
        "task_type": task_type,
        "test_samples": X_test.shape[0],
        "metrics": report
    }
    
    print("\n========== 评估结果 ==========")
    for k, v in report.items():
        if not isinstance(v, (dict, list)):
            print(f"{k}: {v}")
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluate_report.json")
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    print(f"\n========== 评估报告已保存 ==========")
    print(f"路径: {report_path}")

if __name__ == "__main__":
    main()
