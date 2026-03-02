# 梯度提升
# 原理：先训练一棵弱树，看它哪里错了，再训练第二棵树专门纠正第一棵的错误，反复叠加。类比：考试错题本 —— 每次只练上次做错的题，越练越准

import argparse
import joblib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

ALGORITHM_NAME = "gradient_boosting"
TASK_TYPE = "classification"


def load_builtin_dataset(name):
    """加载 sklearn 内置数据集"""
    from sklearn import datasets
    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "digits": datasets.load_digits,
        "diabetes": datasets.load_diabetes,
    }
    if name not in loaders:
        raise ValueError(f"未知内置数据集: {name}，可选: {list(loaders.keys())}")
    data = loaders[name]()
    return data.data, data.target, getattr(data, "feature_names", None)


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
    parser = argparse.ArgumentParser(description="梯度提升训练脚本")
    parser.add_argument("--data_path", type=str, default="",
                        help="CSV 数据文件路径（留空则用内置数据集）")
    parser.add_argument("--builtin_dataset", type=str, default="iris",
                        help="内置数据集名称（仅在 data_path 为空时生效）")
    parser.add_argument("--target_column", type=str, default="target",
                        help="CSV 里目标列的列名")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="测试集比例（0~1）")
    parser.add_argument("--output_dir", type=str, default="/mnt/admin/output",
                        help="模型和元数据输出目录")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="弱学习器数量")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="每棵树的最大深度")
    return parser.parse_args()


def create_model(args):
    return GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )


def main():
    args = parse_args()

    if args.data_path:
        X, y, feature_names = load_csv_dataset(args.data_path, args.target_column)
    else:
        X, y, feature_names = load_builtin_dataset(args.builtin_dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    model = create_model(args)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - start, 3)

    score = model.score(X_test, y_test)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pkl")
    joblib.dump(model, model_path)

    test_data_path = os.path.join(args.output_dir, "test_data.npz")
    np.savez(test_data_path, X_test=X_test, y_test=y_test)

    metadata = {
        "algorithm": ALGORITHM_NAME,
        "task_type": TASK_TYPE,
        "hyper_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
        "data_source": args.data_path or f"builtin:{args.builtin_dataset}",
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "features": X.shape[1],
        "feature_names": list(feature_names) if feature_names is not None else None,
        "test_accuracy": round(score, 4),
        "train_time_seconds": elapsed,
        "model_path": model_path,
        "test_data_path": test_data_path,
    }
    meta_path = os.path.join(args.output_dir, "train_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
