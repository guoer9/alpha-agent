# 并发配置说明

## 机器配置分析

### 硬件配置
- **GPU**: NVIDIA GeForce RTX 3080
  - 总显存: 10GB (10240 MB)
  - 当前使用: 9.4GB (模型加载)
  - 可用显存: 0.6GB (推理缓冲)

- **内存**: 31GB
  - 可用: 约14GB

- **CPU**: 16核

### 模型配置
- **模型**: Qwen3-8B (80亿参数)
- **量化**: 8-bit (INT8)
- **模型显存占用**: 约7GB
- **推理显存需求**: 每个请求约0.3-0.5GB (KV cache)

## 并发限制设置

### 最大并发请求数: 3

**计算依据**:
```
可用显存 = 10GB - 7GB(模型) = 3GB
每请求显存 ≈ 0.5-1GB (KV cache + 临时变量)
最大并发 = 3GB / 1GB ≈ 3个请求
```

**为什么是3个?**
1. **显存限制**: 10GB显存，模型占7GB，剩余3GB
2. **安全裕度**: 每个请求预留1GB显存（包括KV cache和临时变量）
3. **稳定性**: 避免OOM（内存溢出）错误

### 请求队列大小: 10

**说明**:
- 当并发请求达到3个时，新请求进入队列等待
- 队列最大长度10个，超过则返回503错误
- 队列设计避免请求丢失，提供缓冲

### 速率限制

**每IP限制**:
- **每分钟**: 10个请求
- **每小时**: 100个请求

**全局限制**:
- **最大Token数**: 512 (单次请求)
- **超时时间**: 120秒

## 性能预估

### 单请求性能
- **推理时间**: 2-5秒 (取决于输入长度)
- **吞吐量**: 约12-30个请求/分钟 (理想情况)

### 并发性能
```
最大并发: 3个请求
平均响应时间: 3秒
理论吞吐量: 3 / 3秒 = 1请求/秒 = 60请求/分钟
实际吞吐量: 约40-50请求/分钟 (考虑队列等待)
```

## 配置调优建议

### 场景1: 低延迟优先
```bash
MAX_CONCURRENT=2  # 降低并发，减少等待
WORKERS=1
THREADS=2
```
- 优点: 响应快，延迟低
- 缺点: 吞吐量降低

### 场景2: 高吞吐量优先
```bash
MAX_CONCURRENT=3  # 当前配置
WORKERS=1
THREADS=4
```
- 优点: 吞吐量最大化
- 缺点: 可能有排队等待

### 场景3: 极限测试（不推荐）
```bash
MAX_CONCURRENT=4  # 可能导致OOM
WORKERS=1
THREADS=4
```
- 风险: 显存不足，可能崩溃
- 仅用于测试最大容量

## 监控指标

### 关键指标
1. **GPU显存使用率**: 应保持在90%以下
2. **请求队列长度**: 应保持在5以下
3. **请求失败率**: 应低于1%
4. **平均响应时间**: 应低于5秒

### 监控命令

**查看GPU状态**:
```bash
watch -n 1 nvidia-smi
```

**查看服务统计**:
```bash
curl http://localhost:8000/stats
```

**查看健康状态**:
```bash
curl http://localhost:8000/health
```

## 故障处理

### OOM (显存不足)

**症状**:
- 请求失败，返回CUDA out of memory错误
- GPU进程崩溃

**解决方案**:
1. 降低MAX_CONCURRENT到2
2. 减少max_tokens限制到256
3. 重启服务清理显存

**预防措施**:
```python
# 在deploy_with_limits.py中已设置
MAX_CONCURRENT_REQUESTS = 3  # 保守设置
max_tokens = min(data.get('max_tokens', 100), 512)  # 限制最大token
```

### 请求超时

**症状**:
- 请求长时间无响应
- 客户端超时错误

**解决方案**:
1. 检查GPU是否正常工作
2. 查看是否有请求阻塞
3. 重启服务

### 队列满载

**症状**:
- 返回503错误
- 提示"服务繁忙"

**解决方案**:
1. 增加队列大小（不推荐超过20）
2. 优化客户端重试策略
3. 考虑部署多个实例

## 生产环境建议

### 单机部署（当前配置）
```
最大QPS: 约0.5-1 (每秒0.5-1个请求)
适用场景: 小规模应用，内部测试
```

### 多机部署（扩展方案）
如需更高并发，建议：
1. 部署多个实例（每个实例独立GPU）
2. 使用负载均衡器（Nginx/HAProxy）
3. 实现请求分发和故障转移

**示例架构**:
```
客户端
  ↓
负载均衡器 (Nginx)
  ↓
实例1 (GPU 0) → 3并发
实例2 (GPU 1) → 3并发
实例3 (GPU 2) → 3并发
总并发: 9个请求
```

## 配置文件位置

- **主配置**: `deploy_with_limits.py`
  - `MAX_CONCURRENT_REQUESTS = 3`
  - `request_queue.maxsize = 10`
  - `@limiter.limit("10 per minute")`

- **启动脚本**: `start_production.sh`
  - `MAX_CONCURRENT=3`
  - `WORKERS=1`
  - `THREADS=4`

## 修改配置

### 修改最大并发数

编辑 `deploy_with_limits.py`:
```python
MAX_CONCURRENT_REQUESTS = 3  # 改为你想要的值 (建议2-4)
```

### 修改速率限制

编辑 `deploy_with_limits.py`:
```python
@limiter.limit("10 per minute")  # 改为 "20 per minute" 等
```

### 修改队列大小

编辑 `deploy_with_limits.py`:
```python
request_queue = Queue(maxsize=10)  # 改为你想要的值 (建议5-20)
```

## 测试建议

### 压力测试
```bash
# 使用Apache Bench测试
ab -n 100 -c 5 -p request.json -T application/json http://localhost:8000/v1/chat/completions

# 使用wrk测试
wrk -t4 -c10 -d30s --latency http://localhost:8000/health
```

### 监控测试
```bash
# 终端1: 监控GPU
watch -n 1 nvidia-smi

# 终端2: 运行测试
python test_api.py

# 终端3: 查看统计
watch -n 1 curl -s http://localhost:8000/stats
```

## 总结

当前配置针对您的RTX 3080 10GB显存优化：
- ✅ **最大并发**: 3个请求（安全且高效）
- ✅ **队列缓冲**: 10个请求（避免丢失）
- ✅ **速率限制**: 10请求/分钟（防止滥用）
- ✅ **Token限制**: 512（控制显存使用）

这个配置在**稳定性**和**性能**之间取得了良好平衡，适合生产环境使用。
