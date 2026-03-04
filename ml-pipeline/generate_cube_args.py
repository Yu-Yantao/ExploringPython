import os
import json

def get_base_param(type_str, label, default, describe, choice=None, require=0):
    return {
        "type": type_str,
        "item_type": "str",
        "label": label,
        "require": require,
        "choice": choice or [],
        "range": "",
        "default": default,
        "placeholder": "",
        "describe": describe,
        "editable": 1
    }

def get_train_common_params():
    return {
        "--data_path": get_base_param("str", "数据路径", "", "训练用 CSV 路径，留空则使用内置数据集"),
        "--builtin_dataset": get_base_param("str", "内置数据集", "iris", "仅当数据路径为空时生效，如 iris/diabetes等"),
        "--target_column": get_base_param("str", "目标列名", "target", "CSV 中标签列的列名", require=1),
        "--test_size": get_base_param("float", "测试集比例", "0.2", "测试集占比，0~1之间", require=1),
        "--output_dir": get_base_param("str", "输出目录", "/mnt/admin/output", "模型、测试集和元数据的输出目录", require=1)
    }

def generate_jsons():
    base_dir = r"d:\Workplace\Program\Python\ExploringPython\ml-pipeline"
    
    configs = {
        "preprocess": {
            "数据与路径": {
                "--data_path": get_base_param("str", "原始数据路径", "", "原始 CSV 路径，留空则使用内置数据集"),
                "--builtin_dataset": get_base_param("str", "内置数据集", "iris", "仅当数据路径为空时生效"),
                "--target_column": get_base_param("str", "目标列名", "target", "CSV 中标签列的列名", require=1),
                "--test_size": get_base_param("float", "测试集比例", "0.2", "测试集占比，0~1之间", require=1),
                "--output_dir": get_base_param("str", "输出目录", "/mnt/admin/preprocess", "处理后的数据输出目录", require=1)
            },
            "预处理策略": {
                "--missing_strategy": get_base_param("str", "缺失值处理策略", "median", "空值的处理方式", ["drop", "mean", "median", "mode", "none"], 1),
                "--scale_method": get_base_param("str", "标准化方法", "standard", "特征缩放方式", ["standard", "minmax", "none"], 1),
                "--outlier_method": get_base_param("str", "异常值处理", "none", "极端值的处理方式", ["clip", "none"], 1)
            }
        },
        "feature-extract": {
            "数据与路径": {
                "--train_path": get_base_param("str", "训练集路径", "/mnt/admin/preprocess/train.csv", "预处理后的训练集 CSV 路径", require=1),
                "--test_path": get_base_param("str", "测试集路径", "/mnt/admin/preprocess/test.csv", "预处理后的测试集 CSV 路径", require=1),
                "--target_column": get_base_param("str", "目标列名", "target", "CSV 中标签列的列名", require=1),
                "--output_dir": get_base_param("str", "输出目录", "/mnt/admin/features", "提取特征后的数据输出目录", require=1)
            },
            "特征抽取参数": {
                "--method": get_base_param("str", "抽取方法", "select_k_best", "特征选择的方法", ["variance", "correlation", "mutual_info", "select_k_best", "pca", "none"], 1),
                "--top_k": get_base_param("int", "保留特征数", 0, "保留几个特征（0表示自动取一半）", require=1),
                "--variance_threshold": get_base_param("float", "方差阈值", 0.01, "仅方差过滤生效，低于此方差的列会被删除"),
                "--task_type": get_base_param("str", "任务类型", "classification", "分类还是回归任务", ["classification", "regression"], 1)
            }
        },
        "algo-decision-tree": {
            "数据与路径": get_train_common_params(),
            "算法参数": {
                "--max_depth": get_base_param("str", "最大深度", "", "树的最大深度，留空则不限制")
            }
        },
        "algo-random-forest": {
            "数据与路径": get_train_common_params(),
            "算法参数": {
                "--n_estimators": get_base_param("int", "树的数量", 100, "森林中树的数量", require=1),
                "--max_depth": get_base_param("str", "最大深度", "", "树的最大深度，留空则不限制")
            }
        },
        "algo-logistic-regression": {
            "数据与路径": get_train_common_params(),
            "算法参数": {
                "--max_iter": get_base_param("int", "最大迭代次数", 200, "求解器的最大迭代次数", require=1)
            }
        },
        "algo-knn": {
            "数据与路径": get_train_common_params(),
            "算法参数": {
                "--n_neighbors": get_base_param("int", "邻居数量 (K)", 5, "考虑周围多少个邻居的分类", require=1)
            }
        },
        "algo-svm": {
            "数据与路径": get_train_common_params(),
            "算法参数": {
                "--kernel": get_base_param("str", "核函数", "rbf", "用来处理非线性分类的函数类型", ["linear", "rbf", "poly", "sigmoid"], 1)
            }
        },
        "algo-naive-bayes": {
            "数据与路径": get_train_common_params(),
            "算法参数": {}
        },
        "algo-gradient-boosting": {
            "数据与路径": get_train_common_params(),
            "算法参数": {
                "--n_estimators": get_base_param("int", "迭代轮数", 100, "提升树的数量", require=1),
                "--max_depth": get_base_param("int", "最大深度", 3, "每棵提升树的最大深度", require=1)
            }
        },
        "algo-adaboost": {
            "数据与路径": get_train_common_params(),
            "算法参数": {
                "--n_estimators": get_base_param("int", "迭代轮数", 50, "提升器的最大数量", require=1)
            }
        },
        "algo-linear-regression": {
            "数据与路径": get_train_common_params(),
            "算法参数": {}
        },
        "algo-ridge": {
            "数据与路径": get_train_common_params(),
            "算法参数": {
                "--alpha": get_base_param("float", "正则化强度", 1.0, "惩罚项的力度，防止过拟合", require=1)
            }
        }
    }
    
    # 修改回归算法的默认数据集
    configs["algo-linear-regression"]["数据与路径"]["--builtin_dataset"]["default"] = "diabetes"
    configs["algo-ridge"]["数据与路径"]["--builtin_dataset"]["default"] = "diabetes"

    for module_dir, config in configs.items():
        dir_path = os.path.join(base_dir, module_dir)
        if os.path.exists(dir_path):
            # 删除空的算法参数组（比如朴素贝叶斯和线性回归）
            if "算法参数" in config and not config["算法参数"]:
                del config["算法参数"]
                
            json_path = os.path.join(dir_path, "cube_args.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"Generated {json_path}")

if __name__ == "__main__":
    generate_jsons()
