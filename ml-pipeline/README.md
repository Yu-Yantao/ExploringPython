# ML-Pipeline：机器学习通用流水线模块

基于 cube-studio 平台的机器学习 Pipeline 各节点实现，覆盖数据预处理、特征抽取、模型训练（10 种算法）、模型评估（待开发）、模型发布等完整流程。

**每个节点独立一个目录、独立一个镜像**，互不耦合，方便单独迭代和部署。

## 目录结构

```
ml-pipeline/
├── preprocess/                    # 数据预处理
├── feature-extract/               # 关键特征抽取
├── algo-decision-tree/            # 算法：决策树
├── algo-random-forest/            # 算法：随机森林
├── algo-logistic-regression/      # 算法：逻辑回归
├── algo-knn/                      # 算法：K近邻
├── algo-svm/                      # 算法：支持向量机
├── algo-naive-bayes/              # 算法：朴素贝叶斯
├── algo-gradient-boosting/        # 算法：梯度提升
├── algo-adaboost/                 # 算法：AdaBoost
├── algo-linear-regression/        # 算法：线性回归
├── algo-ridge/                    # 算法：岭回归
├── model-evaluate/                # 通用模型评估模块
├── model-serving/                 # 通用模型发布服务
├── build_and_push_all.sh          # 一键构建推送所有镜像
├── cleanup_old_images.sh          # 清理旧镜像
└── README.md
```

每个子目录下都有：`train.py`（或对应脚本）、`requirements.txt`、`Dockerfile`。

## Pipeline 完整流程

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  数据预处理   │ →  │  特征抽取     │ →  │  模型训练     │ →  │  模型评估     │ →  │  模型发布     │
│ preprocess/  │    │feature-extract│   │  algo-xxx/   │    │  (待开发)    │    │  server.py   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       ↓                   ↓                   ↓                                      ↑
  train.csv           train.csv            model.pkl ─────────────────────────────────┘
  test.csv            test.csv             test_data.npz
```

各节点之间通过 **PVC 上的文件** 传递数据，路径约定一致即可。

---

## 一、数据预处理模块（preprocess/）

镜像名：`ml-preprocess:v1`

读取原始数据 → 处理缺失值 → 处理异常值 → 标准化/归一化 → 划分训练集/测试集 → 输出到 PVC。

| 参数                   | 默认值                   | 说明                                                    |
|----------------------|-----------------------|-------------------------------------------------------|
| `--data_path`        | 空                     | 原始 CSV 路径（留空用内置数据集）                                   |
| `--builtin_dataset`  | iris                  | 内置数据集：iris / wine / breast_cancer / digits / diabetes |
| `--target_column`    | target                | 目标列名                                                  |
| `--test_size`        | 0.2                   | 测试集比例                                                 |
| `--missing_strategy` | median                | 缺失值：drop / mean / median / mode / none                |
| `--scale_method`     | standard              | 标准化：standard / minmax / none                          |
| `--outlier_method`   | none                  | 异常值：clip / none                                       |
| `--output_dir`       | /mnt/admin/preprocess | 输出目录                                                  |

输出：`train.csv`、`test.csv`、`preprocess_metadata.json`

---

## 二、关键特征抽取模块（feature-extract/）

镜像名：`ml-feature-extract:v1`

读取预处理后的数据 → 筛选最有用的特征 → 输出精简后的数据。

| 方法          | `--method` 值    | 说明                   |
|-------------|-----------------|----------------------|
| 方差过滤        | `variance`      | 去掉几乎不变的特征            |
| 相关性过滤       | `correlation`   | 保留和目标关系最大的 K 个特征     |
| 互信息         | `mutual_info`   | 用信息论衡量特征重要性          |
| SelectKBest | `select_k_best` | sklearn 统计检验选 K 个最好的 |
| PCA 降维      | `pca`           | 把多个特征压缩成少数主成分        |
| 不处理         | `none`          | 保留所有特征               |

输出：`train.csv`、`test.csv`、`feature_metadata.json`

---

## 三、模型训练模块（10 种算法，各自独立镜像）

### 分类算法（8 种）

| 序号 | 目录                          | 镜像名                           | 算法       | 特有参数                           |
|----|-----------------------------|-------------------------------|----------|--------------------------------|
| 1  | `algo-decision-tree/`       | `algo-decision-tree:v1`       | 决策树      | `--max_depth`                  |
| 2  | `algo-random-forest/`       | `algo-random-forest:v1`       | 随机森林     | `--n_estimators` `--max_depth` |
| 3  | `algo-logistic-regression/` | `algo-logistic-regression:v1` | 逻辑回归     | `--max_iter`                   |
| 4  | `algo-knn/`                 | `algo-knn:v1`                 | K近邻      | `--n_neighbors`                |
| 5  | `algo-svm/`                 | `algo-svm:v1`                 | 支持向量机    | `--kernel`                     |
| 6  | `algo-naive-bayes/`         | `algo-naive-bayes:v1`         | 朴素贝叶斯    | （无额外参数）                        |
| 7  | `algo-gradient-boosting/`   | `algo-gradient-boosting:v1`   | 梯度提升     | `--n_estimators` `--max_depth` |
| 8  | `algo-adaboost/`            | `algo-adaboost:v1`            | AdaBoost | `--n_estimators`               |

### 回归算法（2 种）

| 序号 | 目录                        | 镜像名                         | 算法   | 特有参数      |
|----|---------------------------|-----------------------------|------|-----------|
| 9  | `algo-linear-regression/` | `algo-linear-regression:v1` | 线性回归 | （无额外参数）   |
| 10 | `algo-ridge/`             | `algo-ridge:v1`             | 岭回归  | `--alpha` |

### 所有算法的通用参数

| 参数                  | 默认值               | 说明                           |
|---------------------|-------------------|------------------------------|
| `--data_path`       | 空                 | CSV 路径（留空用内置数据集）             |
| `--builtin_dataset` | iris / diabetes   | 内置数据集（分类默认iris，回归默认diabetes） |
| `--target_column`   | target            | 目标列名                         |
| `--test_size`       | 0.2               | 测试集比例                        |
| `--output_dir`      | /mnt/admin/output | 输出目录                         |

### 输出文件（所有算法统一）

- `model.pkl` — 训练好的模型
- `test_data.npz` — 测试集（供评估节点使用）
- `train_metadata.json` — 算法名、超参、得分、耗时等

---

## 四、模型评估模块（model-evaluate/）

镜像名：`ml-model-evaluate:v1`

这是一个**通用**的评估节点。它内部实现了基于 `task_type`（分类/回归）的路由机制：
1. 它会读取上游训练节点传过来的 `train_metadata.json`，识别当前的算法是解决什么问题的。
2. 自动加载 `model.pkl` 和预留的测试集 `test_data.npz`。
3. 执行预测，并调用对应的评估逻辑（分类算准确率/F1，回归算 MSE/MAE 等）。
4. 输出统一的 `evaluate_report.json`。

| 参数                  | 默认值               | 说明                 |
|---------------------|-------------------|--------------------|
| `--model_dir`       | `/mnt/admin/output` | 必填！包含模型和测试集的目录路径 |
| `--output_dir`      | 空                 | 留空则默认把报告输出到 `model_dir` 下 |

---

## 五、模型发布（model-serving/）

镜像名：`ml-model-serving:v1`

通用模型推理服务，基于 FastAPI 构建。能够加载上述 10 种算法训练出的任意 `model.pkl` 文件，并对外提供 HTTP 预测接口。

在 cube-studio 部署时，需将模型的实际路径（PVC 挂载路径）通过环境变量或参数传入。

### API 接口示例

部署成功后，可发送 POST 请求到 `/predict` 接口进行预测：

```json
{
    "features": [[5.1, 3.5, 1.4, 0.2]]
}
```

---

## 镜像构建与推送

### 一键构建推送所有镜像

```bash
# 把 ml-pipeline 目录传到 master 节点后：
cd ml-pipeline
chmod +x build_and_push_all.sh
./build_and_push_all.sh
```

脚本会自动登录 Harbor，依次构建并推送全部 12 个镜像（2 个通用 + 10 个算法）。

### 清理旧镜像

```bash
chmod +x cleanup_old_images.sh
./cleanup_old_images.sh
```

清理之前上传的旧合体镜像 `ml-train:v1`（本地 + Harbor）。

---

## 在 cube-studio 中使用

1. 为每个模块建一个 **任务模板**，使用各自的镜像
2. 所有节点挂载同一块 PVC（`kubeflow-user-workspace`）
3. 在 Pipeline 编排界面中，按顺序连接节点：

```
预处理 → 特征抽取 → 算法训练（选一个） → 模型评估 → 模型发布
```

### cube-studio 启动参数示例

- 预处理：`--builtin_dataset iris --scale_method standard --output_dir /mnt/admin/preprocess`
- 特征抽取：`--train_path /mnt/admin/preprocess/train.csv --test_path /mnt/admin/preprocess/test.csv --method select_k_best --top_k 3 --output_dir /mnt/admin/features`
- 随机森林训练：`--data_path /mnt/admin/features/train.csv --target_column target --n_estimators 100 --output_dir /mnt/admin/output`

## 完整 Pipeline 本地测试

```bash
# 1. 预处理
cd preprocess
python preprocess.py --builtin_dataset iris --scale_method standard --output_dir ../output/preprocess

# 2. 特征抽取
cd ../feature-extract
python feature_extract.py --train_path ../output/preprocess/train.csv --test_path ../output/preprocess/test.csv --method select_k_best --top_k 3 --output_dir ../output/features

# 3. 训练（选一个算法）
cd ../algo-random-forest
python train.py --data_path ../output/features/train.csv --target_column target --n_estimators 100 --output_dir ../output/train
```
