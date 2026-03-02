import joblib
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import os

# 1. 加载数据（不用下载，直接加载）
print("正在加载鸢尾花数据...")
iris = datasets.load_iris()
X = iris.data  # 特征（花瓣长宽等）
y = iris.target  # 标签（花的品种）

# 2. 定义算法（决策树：最容易理解的算法，像在那做判断题）
print("正在初始化算法模型...")
clf = DecisionTreeClassifier()

# 3. 训练模型
print("开始训练...")
clf.fit(X, y)

# 4. 保存模型（inspect 确认平台把 PVC 挂到 /mnt/admin，必须写这里才会进 PVC）
output_dir = "/mnt/admin/output"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "iris_model.pkl")
joblib.dump(clf, model_path)

print(f"训练完成！模型已保存到: {model_path}")
print("恭喜你，你的第一个机器学习流程跑通了！")
