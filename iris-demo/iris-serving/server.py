import logging
import os

import joblib
from fastapi import FastAPI, Request
from sklearn import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# cube-studio 会通过这个环境变量告诉你模型文件在哪
model_path = os.getenv("KUBEFLOW_MODEL_PATH", "/mnt/admin/output/iris_model.pkl")
logger.info(f"正在加载模型: {model_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}，请确认训练节点已成功运行")

model = joblib.load(model_path)
logger.info("模型加载成功")

# 鸢尾花的品种名称（0/1/2 对应的中文）
iris = datasets.load_iris()
SPECIES_NAMES = {i: name for i, name in enumerate(iris.target_names)}

app = FastAPI(title="鸢尾花分类推理服务")

model_name = os.getenv("KUBEFLOW_MODEL_NAME", "iris-demo")
model_version = os.getenv("KUBEFLOW_MODEL_VERSION", "v1")


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name, "version": model_version}


@app.post(f"/v1/models/{model_name}/versions/{model_version}/predict")
async def predict(request: Request):
    """
    cube-studio 标准预测接口

    请求体示例:
    {
        "features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
    }

    每条数据 4 个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
    """
    data = await request.json()
    features = data.get("features", [])

    if not features:
        return {"error": "请传入 features 字段，格式: [[5.1, 3.5, 1.4, 0.2]]"}

    predictions = model.predict(features).tolist()
    species = [SPECIES_NAMES.get(p, str(p)) for p in predictions]

    return {
        "predictions": predictions,
        "species": species,
        "model": model_name,
        "version": model_version,
    }


@app.post("/predict")
async def simple_predict(request: Request):
    """简化版接口，方便测试"""
    data = await request.json()
    features = data.get("features", [])

    if not features:
        return {"error": "请传入 features 字段，格式: [[5.1, 3.5, 1.4, 0.2]]"}

    predictions = model.predict(features).tolist()
    species = [SPECIES_NAMES.get(p, str(p)) for p in predictions]

    return {"predictions": predictions, "species": species}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
