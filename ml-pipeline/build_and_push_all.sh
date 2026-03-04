#!/bin/bash
# ============================================================
# 一键构建并推送所有 ML Pipeline 镜像到 Harbor
# 使用方式：在 master 节点的 ml-pipeline 目录下执行
#   chmod +x build_and_push_all.sh
#   ./build_and_push_all.sh
# ============================================================

HARBOR_ADDR="192.168.20.16:1103"
PROJECT="cube-studio"
PROXY="http://192.168.202.205:808"
TAG="v1"

MODULES=(
    "preprocess:ml-preprocess"
    "feature-extract:ml-feature-extract"
    "algo-decision-tree:algo-decision-tree"
    "algo-random-forest:algo-random-forest"
    "algo-logistic-regression:algo-logistic-regression"
    "algo-knn:algo-knn"
    "algo-svm:algo-svm"
    "algo-naive-bayes:algo-naive-bayes"
    "algo-gradient-boosting:algo-gradient-boosting"
    "algo-adaboost:algo-adaboost"
    "algo-linear-regression:algo-linear-regression"
    "algo-ridge:algo-ridge"
    "model-evaluate:ml-model-evaluate"
    "model-serving:ml-model-serving"
)

echo "========== 登录 Harbor =========="
docker login -u admin -p Harbor12345 http://${HARBOR_ADDR}
if [ $? -ne 0 ]; then
    echo "Harbor 登录失败，请检查配置"
    exit 1
fi

FAILED=()
SUCCESS=()

for module in "${MODULES[@]}"; do
    DIR="${module%%:*}"
    IMAGE_NAME="${module##*:}"
    FULL_IMAGE="${HARBOR_ADDR}/${PROJECT}/${IMAGE_NAME}:${TAG}"

    echo ""
    echo "========== 构建 ${IMAGE_NAME} =========="

    if [ ! -d "${DIR}" ]; then
        echo "目录 ${DIR} 不存在，跳过"
        FAILED+=("${IMAGE_NAME}")
        continue
    fi

    docker build \
        --build-arg http_proxy=${PROXY} \
        --build-arg https_proxy=${PROXY} \
        -t ${FULL_IMAGE} \
        ./${DIR}

    if [ $? -ne 0 ]; then
        echo "构建 ${IMAGE_NAME} 失败"
        FAILED+=("${IMAGE_NAME}")
        continue
    fi

    echo "---------- 推送 ${IMAGE_NAME} ----------"
    docker push ${FULL_IMAGE}

    if [ $? -ne 0 ]; then
        echo "推送 ${IMAGE_NAME} 失败"
        FAILED+=("${IMAGE_NAME}")
        continue
    fi

    SUCCESS+=("${IMAGE_NAME}")
    echo "========== ${IMAGE_NAME} 完成 =========="
done

echo ""
echo "============================================"
echo "  构建推送完成"
echo "  成功: ${#SUCCESS[@]} 个"
for s in "${SUCCESS[@]}"; do echo "    ✓ ${s}"; done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  失败: ${#FAILED[@]} 个"
    for f in "${FAILED[@]}"; do echo "    ✗ ${f}"; done
fi
echo "============================================"
