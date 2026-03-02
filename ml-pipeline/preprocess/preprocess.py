"""
数据预处理模块 — Pipeline 里的第一个节点
职责：读原始数据 → 清洗 → 标准化/归一化 → 划分训练集/测试集 → 写到 PVC

类比后端：就像 ETL 里的 T（Transform）—— 把"脏数据"洗干净、格式统一，
         下游（训练节点）直接读处理好的数据就行，不用关心原始数据长什么样。

输入：原始 CSV 文件（或内置数据集）
输出：train.csv、test.csv、preprocess_metadata.json（写到 PVC 的 output_dir）
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ========== 缺失值处理 ==========
# 现实中的数据经常有空值（就像数据库里某些字段是 NULL），必须先处理掉才能训练

def handle_missing_values(df, strategy):
    """
    处理缺失值（空值/NaN）

    strategy 可选：
    - "drop"    : 直接删掉含空值的行（简单粗暴，数据多时可用）
    - "mean"    : 用该列的平均值填充（适合数值型，如"身高"列空了就填平均身高）
    - "median"  : 用该列的中位数填充（比平均值更抗极端值干扰）
    - "mode"    : 用该列出现最多的值填充（适合类别型，如"性别"列空了就填出现最多的）
    - "none"    : 不处理（数据本身没空值时选这个）
    """
    if strategy == "none":
        return df
    if strategy == "drop":
        return df.dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "mode":
        df = df.fillna(df.mode().iloc[0])

    return df


# ========== 标准化 / 归一化 ==========
# 不同特征的量纲可能差很大（如"年龄"是0~100，"收入"是0~1000000），
# 很多算法对数值范围敏感，标准化后效果更好

def scale_features(df, target_column, method):
    """
    对特征列做标准化/归一化（不动目标列）

    method 可选：
    - "standard" : 标准化 — 让数据均值=0、标准差=1（最常用，适合大多数算法）
                   公式：(x - 平均值) / 标准差
                   类比：考试分数标准化，不同科目的分数可以比较了
    - "minmax"   : 归一化 — 把数据压缩到 0~1 之间
                   公式：(x - 最小值) / (最大值 - 最小值)
                   类比：把温度从"摄氏度"映射到 0%~100% 的百分比
    - "none"     : 不处理
    """
    if method == "none":
        return df, {}

    feature_cols = [c for c in df.columns if c != target_column]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    scale_params = {}

    if method == "standard":
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val == 0:
                std_val = 1
            df[col] = (df[col] - mean_val) / std_val
            scale_params[col] = {"method": "standard", "mean": mean_val, "std": std_val}

    elif method == "minmax":
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1
            df[col] = (df[col] - min_val) / range_val
            scale_params[col] = {"method": "minmax", "min": min_val, "max": max_val}

    return df, scale_params


# ========== 异常值处理 ==========
# 数据里偶尔会有明显不合理的值（如年龄=999、收入=-100），会干扰模型训练

def handle_outliers(df, target_column, method):
    """
    处理异常值

    method 可选：
    - "clip"  : 把超出 [Q1-1.5*IQR, Q3+1.5*IQR] 范围的值截断到边界
                类比：考试成绩超过100分的按100算，低于0分的按0算
                Q1 = 第25百分位，Q3 = 第75百分位，IQR = Q3 - Q1（四分位距）
    - "none"  : 不处理
    """
    if method == "none":
        return df

    feature_cols = [c for c in df.columns if c != target_column]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if method == "clip":
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower, upper)

    return df


# ========== 内置数据集 → CSV ==========

def load_builtin_as_df(name):
    """把 sklearn 内置数据集转成 DataFrame（方便统一走 CSV 那套流程）"""
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
    feature_names = getattr(data, "feature_names", [f"feature_{i}" for i in range(data.data.shape[1])])
    df = pd.DataFrame(data.data, columns=feature_names)
    df["target"] = data.target
    return df


# ========== 主流程 ==========

def parse_args():
    parser = argparse.ArgumentParser(description="数据预处理模块")

    parser.add_argument("--data_path", type=str, default="",
                        help="原始 CSV 文件路径（留空则用内置数据集）")
    parser.add_argument("--builtin_dataset", type=str, default="iris",
                        help="内置数据集名称（仅在 data_path 为空时生效）")
    parser.add_argument("--target_column", type=str, default="target",
                        help="目标列（标签列）的列名")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="测试集比例（0~1），如 0.2 表示 20%% 做测试")
    parser.add_argument("--missing_strategy", type=str, default="median",
                        choices=["drop", "mean", "median", "mode", "none"],
                        help="缺失值处理策略")
    parser.add_argument("--scale_method", type=str, default="standard",
                        choices=["standard", "minmax", "none"],
                        help="标准化方法")
    parser.add_argument("--outlier_method", type=str, default="none",
                        choices=["clip", "none"],
                        help="异常值处理方法")
    parser.add_argument("--output_dir", type=str, default="/mnt/admin/preprocess",
                        help="处理后的数据输出目录")

    return parser.parse_args()


def main():
    args = parse_args()

    print("========== 数据预处理 ==========")

    # 1. 加载数据
    if args.data_path:
        print(f"从 CSV 加载: {args.data_path}")
        df = pd.read_csv(args.data_path)
    else:
        print(f"使用内置数据集: {args.builtin_dataset}")
        df = load_builtin_as_df(args.builtin_dataset)

    print(f"原始数据: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"目标列: {args.target_column}")
    print(f"缺失值统计:\n{df.isnull().sum().to_string()}")

    # 2. 处理缺失值
    before_rows = len(df)
    df = handle_missing_values(df, args.missing_strategy)
    after_rows = len(df)
    print(f"\n缺失值处理 ({args.missing_strategy}): {before_rows} 行 → {after_rows} 行")

    # 3. 处理异常值
    df = handle_outliers(df, args.target_column, args.outlier_method)
    print(f"异常值处理 ({args.outlier_method}): 完成")

    # 4. 标准化/归一化
    df, scale_params = scale_features(df, args.target_column, args.scale_method)
    print(f"标准化 ({args.scale_method}): 处理了 {len(scale_params)} 列")

    # 5. 划分训练集/测试集
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
    print(f"\n划分数据集: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条")

    # 6. 保存到输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"训练集已保存: {train_path}")
    print(f"测试集已保存: {test_path}")

    # 7. 保存预处理元数据（记录做了什么处理，方便后续复现或推理时用同样的参数）
    metadata = {
        "data_source": args.data_path or f"builtin:{args.builtin_dataset}",
        "target_column": args.target_column,
        "original_rows": before_rows,
        "original_columns": df.shape[1],
        "after_cleaning_rows": after_rows,
        "missing_strategy": args.missing_strategy,
        "outlier_method": args.outlier_method,
        "scale_method": args.scale_method,
        "scale_params": {k: {sk: float(sv) for sk, sv in v.items() if sk != "method"} | {"method": v["method"]}
                         for k, v in scale_params.items()},
        "test_size": args.test_size,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "feature_columns": [c for c in df.columns if c != args.target_column],
        "train_path": train_path,
        "test_path": test_path,
    }
    meta_path = os.path.join(args.output_dir, "preprocess_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存: {meta_path}")

    print("========== 预处理完成 ==========")


if __name__ == "__main__":
    main()
