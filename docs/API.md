# Qwen-vLLM API 文档

## 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 对话补全 (推荐) |
| `/v1/completions` | POST | 文本补全 |
| `/v1/models` | GET | 模型列表 |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 统计信息 |
| `/api/v1/metrics` | GET | 可观测性API (推荐) |
| `/metrics` | GET | Metrics (JSON) |
| `/metrics/prometheus` | GET | Metrics (Prometheus) |

---

## 1. 对话补全 (推荐)

### 请求

```http
POST /v1/chat/completions
Content-Type: application/json
```

### 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| messages | array | ✅ | - | 对话消息列表 |
| max_tokens | int | ❌ | 100 | 最大生成token数 (上限512) |
| temperature | float | ❌ | 0.7 | 温度 (0-1) |

### 示例

```bash
curl -X POST http://10.9.3.131:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "请分类这条新闻：央行宣布降息25个基点"}
    ],
    "max_tokens": 100,
    "temperature": 0.3
  }'
```

### 响应

```json
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "created": 1703404800,
  "model": "qwen-news-classifier",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "这条新闻属于【财经】类别..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 80,
    "total_tokens": 130
  }
}
```

---

## 2. 文本补全

### 请求

```http
POST /v1/completions
Content-Type: application/json
```

### 示例

```bash
curl -X POST http://10.9.3.131:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请分类：苹果发布新款iPhone\n类别：",
    "max_tokens": 50,
    "temperature": 0.3
  }'
```

### 响应

```json
{
  "id": "cmpl-123456789",
  "object": "text_completion",
  "created": 1703404800,
  "model": "qwen-news-classifier",
  "choices": [
    {
      "text": "科技",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 30,
    "completion_tokens": 5,
    "total_tokens": 35
  }
}
```

---

## 3. 健康检查

### 请求

```bash
curl http://10.9.3.131:8000/health
```

### 响应

```json
{
  "status": "ok",
  "gpu": {
    "allocated_gb": 6.92,
    "reserved_gb": 7.02,
    "free_gb": 2.72
  },
  "limits": {
    "max_concurrent_requests": 3,
    "max_queue_size": 10,
    "rate_limit": "无限制"
  }
}
```

---

## 4. 可观测性API (推荐)

### 请求

```bash
curl http://10.9.3.131:8000/api/v1/metrics
```

### 响应

```json
{
  "timestamp": "2025-12-24T21:11:59.226485",
  "service": "qwen-vllm",
  "version": "1.0.0",
  "status": "healthy",
  
  "uptime": {
    "seconds": 117.3,
    "human": "0h 1m"
  },
  
  "requests": {
    "waiting": 0,
    "running": 0,
    "peak_concurrent": 1,
    "total": 3,
    "successful": 3,
    "failed": 0,
    "success_rate_percent": 100.0,
    "error_rate_percent": 0.0,
    "requests_per_second": 0.026
  },
  
  "tokens": {
    "total_input": 27,
    "total_output": 43,
    "avg_per_request": 14.3
  },
  
  "slo": {
    "ttft": {
      "mean": 0.6978,
      "p50": 0.6143,
      "p95": 0.9588,
      "p99": 0.9588,
      "unit": "seconds"
    },
    "throughput": {
      "decoding_mean": 3.41,
      "decoding_p50": 3.39,
      "total": 3.11,
      "unit": "tokens/sec"
    },
    "sample_size": 3
  },
  
  "latency": {
    "mean": 4.6085,
    "p50": 4.0945,
    "p95": 6.2639,
    "p99": 6.2639,
    "min": 3.467,
    "max": 6.2639,
    "unit": "seconds"
  },
  
  "latency_histogram": {
    "0-100ms": 0,
    "100-500ms": 0,
    "500-1000ms": 0,
    "1-2s": 0,
    "2-5s": 2,
    "5-10s": 1,
    ">10s": 0
  },
  
  "gpu": {
    "name": "NVIDIA GeForce RTX 3080",
    "allocated_gb": 7.64,
    "total_gb": 9.64,
    "free_gb": 2.0,
    "utilization_percent": 79.3
  },
  
  "errors": {
    "total": 0,
    "by_type": {}
  },
  
  "alerts": [],
  "alerts_count": 0
}
```

### 指标说明

| 指标 | 说明 |
|------|------|
| `requests.waiting` | 等待队列中的请求数 |
| `requests.running` | 正在处理的请求数 |
| `requests.peak_concurrent` | 峰值并发数 |
| `slo.ttft.*` | Time to First Token (首token延迟) |
| `slo.throughput.*` | Decoding吞吐量 (tokens/秒) |
| `latency.*` | 请求总延迟统计 |
| `latency_histogram` | 延迟分布直方图 |

---

## 5. Prometheus Metrics

### 请求

```bash
curl http://10.9.3.131:8000/metrics/prometheus
```

### 响应

```
# HELP vllm_num_waiting_requests Number of requests waiting in queue
# TYPE vllm_num_waiting_requests gauge
vllm_num_waiting_requests 0

# HELP vllm_num_running_requests Number of requests currently running
# TYPE vllm_num_running_requests gauge
vllm_num_running_requests 0

# HELP vllm_ttft_p95 P95 time to first token in seconds
# TYPE vllm_ttft_p95 gauge
vllm_ttft_p95 0.68

# HELP vllm_decoding_throughput_mean Mean decoding throughput in tokens/sec
# TYPE vllm_decoding_throughput_mean gauge
vllm_decoding_throughput_mean 3.5
```

---

## 6. 错误响应

### 队列已满 (503)

```json
{
  "error": "服务繁忙，请稍后重试",
  "queue_size": 10,
  "max_queue_size": 10
}
```

### 内部错误 (500)

```json
{
  "error": "CUDA out of memory..."
}
```

---

## Python 调用示例

```python
import requests

BASE_URL = "http://10.9.3.131:8000"

# 1. 新闻分类
def classify_news(text):
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": f"请分类：{text}"}],
            "max_tokens": 100,
            "temperature": 0.3
        }
    )
    return response.json()['choices'][0]['message']['content']

# 2. 获取服务状态
def get_metrics():
    response = requests.get(f"{BASE_URL}/api/v1/metrics")
    return response.json()

# 3. 健康检查
def health_check():
    response = requests.get(f"{BASE_URL}/health")
    return response.json()['status'] == 'ok'

# 使用示例
result = classify_news("央行宣布降息25个基点")
print(result)

metrics = get_metrics()
print(f"TTFT P95: {metrics['slo']['ttft']['p95']}s")
print(f"吞吐量: {metrics['slo']['throughput']['decoding_mean']} tok/s")
```

---

## 服务地址

- **本地**: http://localhost:8000
- **内网**: http://10.9.3.131:8000
