# 支持向量机 (SVM)
# 原理：在数据点之间找一条（或一个面）把不同类别分开，且让分界线离两边都尽量远。类比：两群人站操场上，你在中间拉一条绳子把他们分开，绳子要离两边最近的人都尽量远

import argparse
import joblib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.svm import SVC

ALGORITHM_NAME = "svm"
TASK_TYPE = "classification"


def load_csv_dataset(path, target_column):
    """从 CSV 文件加载数据"""
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"CSV 中找不到目标列 '{target_column}'，现有列: {list(df.columns)}")
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values
    feature_names = [c for c in df.columns if c != target_column]
    return X, y, feature_names


def parse_args():
    parser = argparse.ArgumentParser(description="SVM 训练脚本")
    parser.add_argument("--train_path", type=str, required=True, help="上游传来的训练集 CSV 路径")
    parser.add_argument("--test_path", type=str, required=True, help="上游传来的测试集 CSV 路径")
    parser.add_argument("--target_column", type=str, default="target", help="CSV 里目标列的列名")
    parser.add_argument("--output_dir", type=str, default="/mnt/admin/output", help="模型和元数据输出目录")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"], help="SVM 核函数")
    return parser.parse_args()


def create_model(hyper_params):
    return SVC(**hyper_params, random_state=42)


def main():
    args = parse_args()
    hyper_params = {"kernel": args.kernel}

    print("=" * 40)
    print(f"========== 启动 {ALGORITHM_NAME} 训练 ==========")
    print("=" * 40)
    print("【基础参数配置】")
    print(f"  - 训练集路径 (train_path): {args.train_path}")
    print(f"  - 测试集路径 (test_path) : {args.test_path}")
    print(f"  - 目标列名   (target_col): {args.target_column}")
    print(f"  - 输出目录   (output_dir): {args.output_dir}")
    
    if hyper_params:
        print("\n【算法超参数配置】")
        for k, v in hyper_params.items():
            print(f"  - {k}: {v}")

    print("\n========== [1/4] 开始加载数据 ==========")
    X_train, y_train, feature_names = load_csv_dataset(args.train_path, args.target_column)
    X_test, y_test, _ = load_csv_dataset(args.test_path, args.target_column)

    print(f"  - 训练集样本数: {X_train.shape[0]} 行, 特征维度: {X_train.shape[1]} 列")
    print(f"  - 测试集样本数: {X_test.shape[0]} 行, 特征维度: {X_test.shape[1]} 列")

    print("\n========== [2/4] 开始构建并训练模型 ==========")
    model = create_model(hyper_params)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - start, 3)
    print(f"  - 训练完成！耗时: {elapsed} 秒")

    print("\n========== [3/4] 模型简单验证 ==========")
    score = model.score(X_test, y_test)
    metric_name = "accuracy" if TASK_TYPE == "classification" else "r2"
    print(f"  - 测试集 {metric_name}: {round(score, 4)}")

    print("\n========== [4/4] 保存模型和元数据 ==========")
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"  - 模型已保存至: {model_path}")

    test_data_path = os.path.join(args.output_dir, "test_data.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    print(f"  - 测试数据(npz)已保存至: {test_data_path}")

    metadata = {
        "algorithm": ALGORITHM_NAME,
        "task_type": TASK_TYPE,
        "hyper_params": hyper_params,
        "data_source": f"train:{args.train_path}, test:{args.test_path}",
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "features": X_train.shape[1],
        "feature_names": list(feature_names) if feature_names is not None else None,
        f"test_{metric_name}": round(score, 4),
        "train_time_seconds": elapsed,
        "model_path": model_path,
        "test_data_path": test_data_path,
    }
    meta_path = os.path.join(args.output_dir, "train_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
