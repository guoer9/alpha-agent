#!/bin/bash
# Kubernetes构建和部署脚本

set -e

# 配置
IMAGE_NAME="qwen-vllm"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-your-registry.com}"
NAMESPACE="${NAMESPACE:-default}"

echo "=========================================="
echo "Qwen3-8B vLLM Kubernetes部署"
echo "=========================================="
echo "镜像: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "命名空间: ${NAMESPACE}"
echo "=========================================="

# 步骤1: 构建Docker镜像
echo ""
echo "步骤1: 构建Docker镜像..."
cd "$(dirname "$0")/../.."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# 步骤2: 推送镜像
echo ""
echo "步骤2: 推送镜像到仓库..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# 步骤3: 创建命名空间
echo ""
echo "步骤3: 创建命名空间..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# 步骤4: 部署到Kubernetes
echo ""
echo "步骤4: 部署到Kubernetes..."
kubectl apply -k k8s/base -n ${NAMESPACE}

# 步骤5: 等待部署完成
echo ""
echo "步骤5: 等待部署完成..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/qwen-vllm -n ${NAMESPACE}

# 步骤6: 验证部署
echo ""
echo "步骤6: 验证部署..."
kubectl get pods -l app=qwen-vllm -n ${NAMESPACE}
kubectl get svc qwen-vllm -n ${NAMESPACE}

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "查看日志:"
echo "  kubectl logs -f -l app=qwen-vllm -n ${NAMESPACE}"
echo ""
echo "端口转发测试:"
echo "  kubectl port-forward svc/qwen-vllm 8000:8000 -n ${NAMESPACE}"
echo ""
echo "查看metrics:"
echo "  curl http://localhost:8000/metrics"
echo ""
echo "=========================================="
