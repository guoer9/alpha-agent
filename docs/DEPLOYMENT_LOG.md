# vLLM K8s 部署日志

**部署日期**: 2024-12-24  
**部署环境**: Ubuntu 22.04 / Kubernetes v1.28.15  
**部署状态**: ✅ 成功

---

## 1. 部署概述

将 vLLM 服务成功部署到单节点 Kubernetes 集群，实现容器化运行和 GPU 资源管理。

### 最终状态
- **Pod**: `qwen-vllm-79fcc6b897-6lkxw` (1/1 Running)
- **镜像**: `docker.io/library/qwen-vllm:latest` (8.32GB)
- **服务端口**: 8000
- **运行时**: containerd + NVIDIA Container Runtime

---

## 2. 部署步骤

### 2.1 K8s 集群初始化
```bash
# 使用 kubeadm 初始化集群
sudo kubeadm init --config kubeadm-config.yaml

# 配置 kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
```

**配置文件** (`kubeadm-config.yaml`):
- K8s 版本: v1.28.15
- Pod 网络 CIDR: 10.244.0.0/16
- 镜像仓库: 6e4mx6zwaaozht-k8s.xuanyuan.run
- CRI: containerd

### 2.2 网络配置
- 使用 `hostNetwork: true` 模式 (单节点部署)
- DNS 策略: ClusterFirstWithHostNet

### 2.3 GPU 支持
```bash
# 安装 NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit

# 配置 containerd 使用 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd

# 部署 NVIDIA Device Plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

### 2.4 镜像构建

由于 buildkit 代理配置问题，使用 Docker 构建后导入 containerd:

```bash
# Docker 构建镜像
sudo docker build -t qwen-vllm:latest .

# 导出并导入到 containerd
sudo docker save qwen-vllm:latest -o /tmp/qwen-vllm.tar
sudo ctr -n k8s.io images import /tmp/qwen-vllm.tar
```

**Dockerfile 要点**:
- 基础镜像: ubuntu:22.04
- Python 依赖: torch, transformers, accelerate, bitsandbytes
- pip 镜像源: mirrors.aliyun.com
- 运行命令: gunicorn + gthread worker

### 2.5 K8s 部署
```bash
kubectl apply -f k8s/base/deployment.yaml
kubectl apply -f k8s/base/service.yaml
```

---

## 3. 关键配置

### 3.1 Deployment 配置
```yaml
spec:
  replicas: 1
  template:
    spec:
      hostNetwork: true
      containers:
      - name: qwen-vllm
        image: docker.io/library/qwen-vllm:latest
        imagePullPolicy: Never
        resources:
          requests:
            nvidia.com/gpu: "1"
          limits:
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        hostPath:
          path: /home/zch/qwen_vllm/models/qwen-news-classifier-merged
```

### 3.2 模型挂载
- 使用 hostPath 直接挂载本地模型目录
- 路径: `/home/zch/qwen_vllm/models/qwen-news-classifier-merged`

---

## 4. 遇到的问题及解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ErrImageNeverPull` | 镜像不在 containerd 中 | Docker 构建后用 ctr 导入到 k8s.io 命名空间 |
| buildkit 网络超时 | 代理配置未生效 | 改用 Docker 构建 |
| Pod Pending (GPU) | 缺少 NVIDIA Device Plugin | 安装 nvidia-container-toolkit 和 device-plugin |
| kubelet 镜像拉取失败 | registry.k8s.io 不可达 | 使用专用镜像源 |

---

## 5. 服务验证

```bash
# 健康检查
curl http://localhost:8000/health
# 返回: {"status":"ok", "gpu":{"free_gb":9.64}, ...}

# Metrics 端点
curl http://localhost:8000/metrics

# API 调用
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

---

## 6. 资源使用

| 资源 | 配置 |
|------|------|
| 镜像大小 | 8.32 GB |
| GPU 内存 | ~9.64 GB 可用 |
| CPU 请求 | 4 核 |
| 内存请求 | 16 Gi |

---

## 7. 后续优化建议

1. **网络插件**: 安装 Flannel/Calico 支持多节点集群
2. **Ingress**: 配置 Ingress Controller 统一入口
3. **监控**: 部署 Prometheus Operator 启用 ServiceMonitor
4. **HPA**: 配置自动扩缩容策略
5. **持久化**: 使用 PV/PVC 替代 hostPath

---

## 8. 相关文件

- `/home/zch/qwen_vllm/Dockerfile` - 容器构建文件
- `/home/zch/qwen_vllm/k8s/base/deployment.yaml` - K8s Deployment
- `/home/zch/qwen_vllm/k8s/base/service.yaml` - K8s Service
- `/home/zch/qwen_vllm/kubeadm-config.yaml` - kubeadm 配置
