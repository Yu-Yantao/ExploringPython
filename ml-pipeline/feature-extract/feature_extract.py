"""
关键特征抽取模块 — Pipeline 里的第二个节点（在预处理之后、训练之前）
职责：读预处理后的数据 → 筛选/构造最有用的特征 → 输出精简后的数据

为什么要做特征抽取？
    原始数据可能有几十甚至上百列特征，但不是每一列都有用。
    类比后端：就像数据库查询时只 SELECT 需要的字段，而不是 SELECT *。
    - 去掉没用的列 → 训练更快、模型更简洁
    - 留下关键列 → 模型效果可能更好（噪声少了）

输入：预处理后的 train.csv、test.csv（由 preprocess.py 产出）
输出：筛选特征后的 train.csv、test.csv、feature_metadata.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


# ========== 特征选择方法 ==========

def variance_filter(X_df, threshold):
    """
    方差过滤 — 去掉「几乎不变」的特征

    原理：如果某一列的值几乎都一样（方差接近 0），那它对区分不同类别没什么帮助
    类比：一张用户表里，「国家」列全是「中国」，这列对预测没用，可以去掉
    参数：threshold = 方差阈值，低于这个值的列会被去掉（默认 0.01）
    """
    variances = X_df.var()
    keep_cols = variances[variances >= threshold].index.tolist()
    removed = variances[variances < threshold].index.tolist()
    return keep_cols, removed, {"method": "variance", "threshold": threshold}


def correlation_filter(X_df, y_series, top_k):
    """
    相关性过滤 — 保留和目标列「关系最大」的 K 个特征

    原理：算每个特征和目标列的相关系数（-1~1），绝对值越大说明关系越强
    类比：预测房价时，「面积」和房价强相关（0.9），「门牌号」和房价几乎无关（0.01），
         那就留「面积」、去掉「门牌号」
    参数：top_k = 保留几个特征
    """
    correlations = {}
    for col in X_df.columns:
        try:
            corr = abs(X_df[col].corr(y_series))
            correlations[col] = corr if not np.isnan(corr) else 0
        except Exception:
            correlations[col] = 0

    sorted_cols = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    keep_cols = [col for col, _ in sorted_cols[:top_k]]
    removed = [col for col, _ in sorted_cols[top_k:]]
    return keep_cols, removed, {"method": "correlation", "top_k": top_k, "correlations": {k: round(v, 4) for k, v in sorted_cols}}


def mutual_info_filter(X_df, y_array, top_k, task_type):
    """
    互信息过滤 — 用信息论衡量特征和目标的关联

    原理：互信息（Mutual Information）衡量的是「知道了这个特征后，对预测目标能减少多少不确定性」
         值越大 = 这个特征越有用
    类比：猜一个人的职业，知道了「学历」能大幅缩小范围（互信息高），
         知道了「鞋码」帮助不大（互信息低）
    参数：top_k = 保留几个特征
          task_type = "classification" 或 "regression"（不同任务用不同的互信息计算方式）
    """
    if task_type == "classification":
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_df.values, y_array, random_state=42)
    else:
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X_df.values, y_array, random_state=42)

    mi_dict = dict(zip(X_df.columns, mi_scores))
    sorted_cols = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
    keep_cols = [col for col, _ in sorted_cols[:top_k]]
    removed = [col for col, _ in sorted_cols[top_k:]]
    return keep_cols, removed, {"method": "mutual_info", "top_k": top_k, "scores": {k: round(v, 4) for k, v in sorted_cols}}


def select_k_best(X_df, y_array, top_k, task_type):
    """
    SelectKBest — sklearn 自带的「选 K 个最好的特征」

    原理：用统计检验（如卡方检验、F检验）给每个特征打分，选分数最高的 K 个
    类比：面试时先笔试筛掉一批人，只留分数最高的 K 个进面试
    参数：top_k = 保留几个特征
          task_type = 分类用卡方/F检验，回归用F检验
    """
    from sklearn.feature_selection import SelectKBest as SKB

    if task_type == "classification":
        from sklearn.feature_selection import f_classif
        score_func = f_classif
    else:
        from sklearn.feature_selection import f_regression
        score_func = f_regression

    k = min(top_k, X_df.shape[1])
    selector = SKB(score_func=score_func, k=k)
    selector.fit(X_df.values, y_array)

    scores = dict(zip(X_df.columns, selector.scores_))
    mask = selector.get_support()
    keep_cols = X_df.columns[mask].tolist()
    removed = X_df.columns[~mask].tolist()
    return keep_cols, removed, {"method": "select_k_best", "top_k": k, "scores": {k: round(v, 4) for k, v in scores.items()}}


def pca_transform(X_df, n_components):
    """
    PCA 降维 — 把很多特征「压缩」成少数几个新特征

    原理：找到数据变化最大的几个方向，把原来的特征投影到这几个方向上，
         产生新的「主成分」特征（PC1, PC2, ...），数量比原来少但保留了大部分信息
    类比：一张100列的表，PCA 可以压缩成5列，这5列是原来100列的「精华摘要」
    注意：PCA 产生的新特征是原始特征的组合，不再是原来的列名了
    参数：n_components = 压缩成几个主成分
    """
    from sklearn.decomposition import PCA

    n = min(n_components, X_df.shape[1])
    pca = PCA(n_components=n, random_state=42)
    transformed = pca.fit_transform(X_df.values)

    new_columns = [f"PC{i+1}" for i in range(n)]
    new_df = pd.DataFrame(transformed, columns=new_columns, index=X_df.index)

    explained = pca.explained_variance_ratio_
    info = {
        "method": "pca",
        "n_components": n,
        "explained_variance_ratio": [round(v, 4) for v in explained],
        "total_explained": round(sum(explained), 4),
    }
    return new_df, info


# ========== 主流程 ==========

METHODS = {
    "variance":     "方差过滤 — 去掉几乎不变的特征",
    "correlation":  "相关性过滤 — 保留和目标关系最大的特征",
    "mutual_info":  "互信息 — 用信息论衡量特征重要性",
    "select_k_best": "SelectKBest — sklearn 统计检验选特征",
    "pca":          "PCA 降维 — 把多个特征压缩成少数主成分",
    "none":         "不做特征选择，保留所有特征",
}


def parse_args():
    parser = argparse.ArgumentParser(description="关键特征抽取模块")

    parser.add_argument("--train_path", type=str, required=True,
                        help="预处理后的训练集 CSV 路径")
    parser.add_argument("--test_path", type=str, required=True,
                        help="预处理后的测试集 CSV 路径")
    parser.add_argument("--target_column", type=str, default="target",
                        help="目标列名")
    parser.add_argument("--method", type=str, default="select_k_best",
                        choices=list(METHODS.keys()),
                        help="特征选择方法")
    parser.add_argument("--top_k", type=int, default=0,
                        help="保留几个特征（0 = 自动取特征总数的一半）")
    parser.add_argument("--variance_threshold", type=float, default=0.01,
                        help="方差过滤阈值（仅 method=variance 时生效）")
    parser.add_argument("--task_type", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="任务类型（影响互信息和 SelectKBest 的计算方式）")
    parser.add_argument("--output_dir", type=str, default="/mnt/admin/features",
                        help="输出目录")

    return parser.parse_args()


def main():
    args = parse_args()

    print("========== 关键特征抽取 ==========")
    print(f"方法: {args.method} — {METHODS[args.method]}")

    # 1. 读取预处理后的数据
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    print(f"训练集: {train_df.shape[0]} 行 x {train_df.shape[1]} 列")
    print(f"测试集: {test_df.shape[0]} 行 x {test_df.shape[1]} 列")

    feature_cols = [c for c in train_df.columns if c != args.target_column]
    X_train = train_df[feature_cols]
    y_train = train_df[args.target_column]
    X_test = test_df[feature_cols]
    y_test = test_df[args.target_column]

    top_k = args.top_k if args.top_k > 0 else max(1, len(feature_cols) // 2)
    print(f"原始特征数: {len(feature_cols)}, 目标保留: {top_k}")

    # 2. 执行特征选择
    if args.method == "none":
        keep_cols = feature_cols
        removed_cols = []
        method_info = {"method": "none"}
        X_train_new = X_train
        X_test_new = X_test

    elif args.method == "pca":
        X_train_new, method_info = pca_transform(X_train, top_k)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(top_k, X_train.shape[1]), random_state=42)
        pca.fit(X_train.values)
        X_test_new = pd.DataFrame(
            pca.transform(X_test.values),
            columns=X_train_new.columns,
            index=X_test.index,
        )
        keep_cols = X_train_new.columns.tolist()
        removed_cols = feature_cols

    else:
        if args.method == "variance":
            keep_cols, removed_cols, method_info = variance_filter(X_train, args.variance_threshold)
        elif args.method == "correlation":
            keep_cols, removed_cols, method_info = correlation_filter(X_train, y_train, top_k)
        elif args.method == "mutual_info":
            keep_cols, removed_cols, method_info = mutual_info_filter(X_train, y_train.values, top_k, args.task_type)
        elif args.method == "select_k_best":
            keep_cols, removed_cols, method_info = select_k_best(X_train, y_train.values, top_k, args.task_type)

        X_train_new = X_train[keep_cols]
        X_test_new = X_test[keep_cols]

    print(f"\n保留特征 ({len(keep_cols)}): {keep_cols}")
    if removed_cols:
        print(f"去掉特征 ({len(removed_cols)}): {removed_cols}")

    # 3. 拼回目标列并保存
    os.makedirs(args.output_dir, exist_ok=True)

    train_out = X_train_new.copy()
    train_out[args.target_column] = y_train.values
    test_out = X_test_new.copy()
    test_out[args.target_column] = y_test.values

    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    print(f"\n训练集已保存: {train_path}")
    print(f"测试集已保存: {test_path}")

    # 4. 保存元数据
    metadata = {
        "method": args.method,
        "method_description": METHODS[args.method],
        "task_type": args.task_type,
        "original_features": len(feature_cols),
        "selected_features": len(keep_cols),
        "kept_columns": keep_cols,
        "removed_columns": removed_cols,
        "method_details": method_info,
        "train_path": train_path,
        "test_path": test_path,
    }
    meta_path = os.path.join(args.output_dir, "feature_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"元数据已保存: {meta_path}")

    print("========== 特征抽取完成 ==========")


if __name__ == "__main__":
    main()
