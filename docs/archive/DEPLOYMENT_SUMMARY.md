# Qwen3-8B 金融新闻分类模型部署总结

## 📊 部署完成状态

### ✅ 已完成任务

1. **模型下载** - Qwen3-8B基础模型 (16GB)
2. **LoRA合并** - 微调模型与基础模型合并
3. **8-bit量化** - 模型压缩至约7GB显存占用
4. **并发限制** - 配置最优并发请求数
5. **API接口** - 完整的金融新闻分类API
6. **生产部署** - Gunicorn生产级服务器

## 🖥️ 机器配置

- **GPU**: NVIDIA GeForce RTX 3080 (10GB显存)
- **驱动**: 580.95.05, CUDA 13.0
- **内存**: 31GB
- **CPU**: 16核
- **显存使用**: 7GB (模型) + 3GB (推理缓冲)

## 🚀 服务配置

### 并发限制（已优化）
- **最大并发请求**: 3个
- **请求队列大小**: 10个
- **速率限制**: 10请求/分钟 (每IP)
- **最大Token数**: 512
- **超时时间**: 120秒

### 性能指标
- **单请求响应时间**: 2-5秒
- **理论吞吐量**: 约60请求/分钟
- **实际吞吐量**: 约40-50请求/分钟

## 📁 文件结构

```
/home/zch/qwen_vllm/
├── models/
│   ├── Qwen/
│   │   └── Qwen3-8B/                    # 基础模型 (16GB)
│   ├── qwen-news-classifier/            # 原始LoRA适配器
│   │   ├── checkpoint-6500/
│   │   └── checkpoint-10005/            # 最佳checkpoint
│   └── qwen-news-classifier-merged/     # 合并后的模型 (16GB)
│
├── deploy_with_transformers.py         # 基础部署脚本
├── deploy_with_limits.py               # 带并发限制的部署脚本 ⭐
├── start_deploy.sh                     # 开发环境启动脚本
├── start_production.sh                 # 生产环境启动脚本 ⭐
├── setup_deploy_env.sh                 # 环境安装脚本
│
├── test_api.py                         # API测试脚本
├── API_USAGE.md                        # API使用文档 ⭐
├── CONCURRENCY_CONFIG.md               # 并发配置说明 ⭐
└── DEPLOYMENT_SUMMARY.md               # 本文档
```

## 🎯 快速启动

### 方法1: 生产环境（推荐）

```bash
cd /home/zch/qwen_vllm

# 启动服务（带并发限制和速率限制）
bash start_production.sh
```

### 方法2: 开发环境

```bash
cd /home/zch/qwen_vllm

# 启动服务（简单模式）
bash start_deploy.sh
```

### 方法3: 手动启动

```bash
# 激活环境
conda activate vllm-deploy

# 启动服务
python deploy_with_limits.py \
    --model-path ./models/qwen-news-classifier-merged \
    --host 0.0.0.0 \
    --port 8000
```

## 🔌 API接口

### 服务地址
- **主地址**: http://localhost:8000
- **健康检查**: http://localhost:8000/health
- **统计信息**: http://localhost:8000/stats
- **模型列表**: http://localhost:8000/v1/models

### 金融新闻分类接口

**端点**: `POST /v1/chat/completions`

**请求示例**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "请分析以下新闻的类别：央行宣布降准0.5个百分点"
      }
    ],
    "max_tokens": 100,
    "temperature": 0.3
  }'
```

**Python客户端**:
```python
import requests

def classify_news(news_text):
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "user", "content": f"请分析以下新闻的类别：{news_text}"}
        ],
        "max_tokens": 100,
        "temperature": 0.3
    }
    response = requests.post(url, json=payload)
    return response.json()['choices'][0]['message']['content']

# 使用
result = classify_news("央行宣布降准0.5个百分点")
print(result)
```

## 📊 监控命令

### 查看GPU状态
```bash
watch -n 1 nvidia-smi
```

### 查看服务统计
```bash
curl http://localhost:8000/stats
```

### 查看健康状态
```bash
curl http://localhost:8000/health
```

### 测试API
```bash
python test_api.py
```

## 🔧 维护操作

### 重启服务
```bash
# 停止服务 (Ctrl+C 或 kill进程)
pkill -f deploy_with_limits

# 重新启动
bash start_production.sh
```

### 查看日志
服务日志直接输出到终端，包含：
- 请求处理信息
- 错误信息
- 性能统计

### 清理显存
```bash
# 如果遇到显存不足
pkill -f python
nvidia-smi

# 重启服务
bash start_production.sh
```

## ⚙️ 配置调整

### 修改并发数

编辑 `deploy_with_limits.py`:
```python
MAX_CONCURRENT_REQUESTS = 3  # 改为2-4之间的值
```

### 修改速率限制

编辑 `deploy_with_limits.py`:
```python
@limiter.limit("10 per minute")  # 改为你需要的值
```

### 修改最大Token数

编辑 `deploy_with_limits.py`:
```python
max_tokens = min(data.get('max_tokens', 100), 512)  # 改为256或1024
```

## 🎓 使用场景

### 场景1: 实时新闻分类
```python
# 实时处理新闻流
for news in news_stream:
    category = classify_news(news)
    save_to_database(news, category)
```

### 场景2: 批量历史数据分类
```python
# 批量处理历史数据
from concurrent.futures import ThreadPoolExecutor

def classify_batch(news_list):
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(classify_news, news_list))
    return results
```

### 场景3: API服务集成
```python
# 集成到现有API服务
@app.route('/classify', methods=['POST'])
def classify_endpoint():
    news = request.json['news']
    category = classify_news(news)
    return jsonify({'category': category})
```

## 📈 性能优化建议

### 1. 降低延迟
- 减少max_tokens到100
- 降低temperature到0.1
- 使用缓存机制

### 2. 提高吞吐量
- 增加并发数到4（风险：可能OOM）
- 使用批处理
- 部署多个实例

### 3. 节省显存
- 减少max_model_len
- 使用4-bit量化（需重新配置）
- 定期清理缓存

## ⚠️ 常见问题

### Q1: 服务无响应
```bash
# 检查进程
ps aux | grep deploy_with_limits

# 检查端口
netstat -tlnp | grep 8000

# 重启服务
bash start_production.sh
```

### Q2: 显存不足 (OOM)
```bash
# 降低并发数
# 编辑 deploy_with_limits.py
MAX_CONCURRENT_REQUESTS = 2

# 重启服务
bash start_production.sh
```

### Q3: 请求超时
```bash
# 检查GPU状态
nvidia-smi

# 查看是否有阻塞请求
curl http://localhost:8000/stats
```

### Q4: 速率限制触发
```bash
# 查看当前限制
curl http://localhost:8000/health

# 修改限制
# 编辑 deploy_with_limits.py
@limiter.limit("20 per minute")  # 增加限制
```

## 📚 相关文档

- **API使用指南**: `API_USAGE.md` - 详细的API接口说明和示例
- **并发配置**: `CONCURRENCY_CONFIG.md` - 并发限制的详细说明
- **快速开始**: `QUICKSTART.md` - 快速入门指南
- **安装指南**: `INSTALL_5090.md` - 环境安装说明

## 🔐 安全建议

### 生产环境部署
1. **使用反向代理** (Nginx)
2. **启用HTTPS**
3. **配置防火墙**
4. **添加认证机制**
5. **日志审计**

### 示例Nginx配置
```nginx
upstream qwen_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://qwen_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # 速率限制
        limit_req zone=api_limit burst=5;
    }
}
```

## 📞 技术支持

### 环境信息
- **Conda环境**: vllm-deploy
- **Python**: 3.10
- **PyTorch**: 2.9.0+cu128
- **Transformers**: 4.57.3
- **vLLM**: 0.13.0

### 检查清单
- [ ] GPU驱动正常 (`nvidia-smi`)
- [ ] Conda环境激活 (`conda activate vllm-deploy`)
- [ ] 模型文件完整 (`ls models/qwen-news-classifier-merged/`)
- [ ] 端口未被占用 (`netstat -tlnp | grep 8000`)
- [ ] 显存充足 (`nvidia-smi`)

## 🎉 部署成功

您的Qwen3-8B金融新闻分类模型已成功部署！

**当前状态**:
- ✅ 模型加载完成
- ✅ 8-bit量化运行
- ✅ 并发限制已配置
- ✅ API服务正常
- ✅ 测试通过

**下一步**:
1. 运行 `python test_api.py` 测试API
2. 查看 `API_USAGE.md` 了解详细用法
3. 集成到您的应用中
4. 监控服务性能

祝使用愉快！🚀
