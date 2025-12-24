#!/bin/bash
# Kubernetes快速部署脚本

set -e

echo "=========================================="
echo "Qwen3-8B vLLM Kubernetes部署"
echo "=========================================="

# 1. 构建镜像
echo "步骤1: 构建Docker镜像..."
docker build -t qwen-vllm:latest .

# 2. 部署到K8s
echo ""
echo "步骤2: 部署到Kubernetes..."
kubectl apply -k k8s/base

# 3. 等待Pod就绪
echo ""
echo "步骤3: 等待Pod启动..."
kubectl wait --for=condition=ready pod -l app=qwen-vllm --timeout=300s

# 4. 查看状态
echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
kubectl get pods -l app=qwen-vllm
kubectl get svc qwen-vllm

echo ""
echo "测试服务:"
echo "  kubectl port-forward svc/qwen-vllm 8000:8000"
echo "  curl http://localhost:8000/health"
echo ""
echo "查看metrics:"
echo "  curl http://localhost:8000/metrics"
echo ""
echo "查看日志:"
echo "  kubectl logs -f -l app=qwen-vllm"
echo "=========================================="
