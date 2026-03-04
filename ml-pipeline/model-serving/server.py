"""
通用模型推理服务（Model Serving）

职责：读取指定的 model.pkl，暴露 HTTP 接口供外部调用进行预测。
支持动态模型名和版本，兼容 cube-studio 的部署要求。
"""
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# 从环境变量获取配置（cube-studio 部署时会通过环境变量或启动命令传参，但在 serving 阶段最常用的是环境变量）
# 如果环境变量没有，则使用默认值
MODEL_PATH = os.getenv("KUBEFLOW_MODEL_PATH", "/mnt/admin/output/model.pkl")
MODEL_NAME = os.getenv("KUBEFLOW_MODEL_NAME", "default-model")
MODEL_VERSION = os.getenv("KUBEFLOW_MODEL_VERSION", "v1")
PORT = int(os.getenv("PORT", 8080))

print(f"========== 启动推理服务 ==========")
print(f"模型名称: {MODEL_NAME}")
print(f"模型版本: {MODEL_VERSION}")
print(f"模型路径: {MODEL_PATH}")

# 1. 加载模型
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    # 不在这里直接退出，让服务起来，调用 health 接口时能知道不健康
    model = None

# 2. 初始化 FastAPI
app = FastAPI(title=f"模型推理服务 - {MODEL_NAME}")

@app.get("/health")
async def health():
    """健康检查接口（供 Kubernetes / cube-studio 调用）"""
    if model is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "模型未加载"})
    return {"status": "ok", "model_name": MODEL_NAME, "version": MODEL_VERSION}

@app.post("/predict")
async def predict_simple(request: Request):
    """
    通用预测接口
    接收 JSON 格式的特征数据，例如：
    {
        "features": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
    }
    或者字典格式（如果有 feature_names）：
    {
        "features": [{"age": 30, "duration": 487}, {"age": 39, "duration": 346}]
    }
    """
    if model is None:
        return JSONResponse(status_code=503, content={"error": "模型未准备好"})

    try:
        data = await request.json()
        if "features" not in data:
            return JSONResponse(status_code=400, content={"error": "请求体中缺少 'features' 字段"})
        
        features = data["features"]
        
        # 如果传入的是字典列表，转换为 DataFrame，这样可以兼容具有特定列名要求的 sklearn Pipeline/ColumnTransformer
        if isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
            X = pd.DataFrame(features)
        else:
            # 否则转为 numpy 数组
            X = np.array(features)
        
        # 调用模型的 predict 方法
        predictions = model.predict(X)
        
        # 将 numpy 类型转换为 Python 原生类型以便 JSON 序列化
        predictions = predictions.tolist()
        
        return {"predictions": predictions}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# KFServing / KServe 标准接口格式（为了更好的兼容性）
@app.post(f"/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}:predict")
async def predict_kfserving(request: Request):
    """兼容 KFServing 规范的预测接口"""
    return await predict_simple(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
