#!/bin/bash
# ============================================================
# 清理之前上传的旧镜像（本地 + Harbor）
# 使用方式：在 master 节点执行
#   chmod +x cleanup_old_images.sh
#   ./cleanup_old_images.sh
# ============================================================

HARBOR_ADDR="192.168.20.16:1103"
PROJECT="cube-studio"

# 需要清理的旧镜像列表（拆分前的合体镜像）
OLD_IMAGES=(
    "ml-train:v1"
)

echo "========== 清理旧镜像 =========="

for image in "${OLD_IMAGES[@]}"; do
    IMAGE_NAME="${image%%:*}"
    IMAGE_TAG="${image##*:}"
    FULL_IMAGE="${HARBOR_ADDR}/${PROJECT}/${image}"

    echo ""
    echo "---------- 处理 ${image} ----------"

    # 1. 删除本地镜像
    echo "[本地] 删除 ${FULL_IMAGE}"
    docker rmi ${FULL_IMAGE} 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[本地] 已删除"
    else
        echo "[本地] 不存在或已删除"
    fi

    # 2. 从 Harbor 删除（通过 API）
    echo "[Harbor] 获取 ${IMAGE_NAME}:${IMAGE_TAG} 的 digest..."

    DIGEST=$(curl -s -I \
        -u admin:Harbor12345 \
        -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
        "http://${HARBOR_ADDR}/v2/${PROJECT}/${IMAGE_NAME}/manifests/${IMAGE_TAG}" \
        2>/dev/null | grep -i "Docker-Content-Digest" | awk '{print $2}' | tr -d '\r')

    if [ -n "${DIGEST}" ]; then
        echo "[Harbor] digest: ${DIGEST}"
        echo "[Harbor] 删除中..."

        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            -u admin:Harbor12345 \
            -X DELETE \
            "http://${HARBOR_ADDR}/v2/${PROJECT}/${IMAGE_NAME}/manifests/${DIGEST}")

        if [ "${HTTP_CODE}" = "202" ] || [ "${HTTP_CODE}" = "200" ]; then
            echo "[Harbor] 已删除"
        else
            echo "[Harbor] 删除返回 HTTP ${HTTP_CODE}，可手动在 Harbor 管理界面删除"
        fi
    else
        echo "[Harbor] 未找到该镜像或无法获取 digest，可能已删除"
    fi
done

echo ""
echo "========== 清理完成 =========="
echo "提示：Harbor 删除后如需回收磁盘空间，需在 Harbor 管理界面执行「垃圾回收」"
