# Qwen-vLLM 可观测性指标文档

## 概览

服务提供完整的可观测性指标，支持：
- JSON API (`/api/v1/metrics`)
- Prometheus格式 (`/metrics/prometheus`)
- 实时监控仪表板

---

## 指标分类

### 1. 请求状态指标

| 指标 | 类型 | 说明 |
|------|------|------|
| `num_waiting_requests` | Gauge | 等待队列中的请求数 |
| `num_running_requests` | Gauge | 正在处理的请求数 |
| `peak_concurrent_requests` | Gauge | 历史峰值并发数 |
| `total_requests` | Counter | 总请求数 |
| `successful_requests` | Counter | 成功请求数 |
| `failed_requests` | Counter | 失败请求数 |
| `success_rate_percent` | Gauge | 成功率 (%) |
| `error_rate_percent` | Gauge | 错误率 (%) |
| `requests_per_second` | Gauge | 每秒请求数 |

### 2. SLO 指标

#### Time to First Token (TTFT)

| 指标 | 说明 |
|------|------|
| `ttft_mean` | 平均首Token延迟 |
| `ttft_p50` | P50 首Token延迟 |
| `ttft_p95` | P95 首Token延迟 |
| `ttft_p99` | P99 首Token延迟 |

#### Decoding Throughput

| 指标 | 说明 |
|------|------|
| `decoding_throughput_mean` | 平均解码吞吐量 (tokens/秒) |
| `decoding_throughput_p50` | P50 解码吞吐量 |
| `total_throughput` | 总吞吐量 |

### 3. 延迟指标

| 指标 | 说明 |
|------|------|
| `latency_mean` | 平均请求延迟 |
| `latency_p50` | P50 延迟 |
| `latency_p95` | P95 延迟 |
| `latency_p99` | P99 延迟 |
| `latency_min` | 最小延迟 |
| `latency_max` | 最大延迟 |

### 4. 延迟直方图

```json
{
  "0-100ms": 0,
  "100-500ms": 5,
  "500-1000ms": 10,
  "1-2s": 8,
  "2-5s": 3,
  "5-10s": 1,
  ">10s": 0
}
```

### 5. Token 统计

| 指标 | 说明 |
|------|------|
| `total_input_tokens` | 总输入token数 |
| `total_output_tokens` | 总输出token数 |
| `avg_tokens_per_request` | 每请求平均token数 |

### 6. GPU 指标

| 指标 | 说明 |
|------|------|
| `gpu.name` | GPU型号 |
| `gpu.allocated_gb` | 已分配显存 (GB) |
| `gpu.total_gb` | 总显存 (GB) |
| `gpu.free_gb` | 可用显存 (GB) |
| `gpu.utilization_percent` | 显存使用率 (%) |

### 7. 错误统计

| 指标 | 说明 |
|------|------|
| `errors.total` | 总错误数 |
| `errors.by_type` | 按类型分类的错误 |

---

## API 端点

### JSON 格式 (推荐)

```bash
curl http://10.9.3.131:8000/api/v1/metrics
```

完整响应示例见 [API.md](API.md)

### Prometheus 格式

```bash
curl http://10.9.3.131:8000/metrics/prometheus
```

---

## 告警规则

### 内置告警

| 告警 | 触发条件 | 级别 |
|------|----------|------|
| 队列过长 | `num_waiting_requests > 5` | warning |
| TTFT过高 | `ttft_p95 > 3.0s` | warning |
| 错误率过高 | `error_rate_percent > 10%` | critical |

### Prometheus 告警配置

```yaml
groups:
  - name: vllm-alerts
    rules:
      - alert: VLLMHighQueueLength
        expr: vllm_num_waiting_requests > 5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "请求队列过长"
          
      - alert: VLLMHighTTFT
        expr: vllm_ttft_p95 > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TTFT延迟过高"
          
      - alert: VLLMHighErrorRate
        expr: (vllm_failed_requests / vllm_total_requests) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "错误率超过10%"
```

---

## 实时监控

### 终端仪表板

```bash
python scripts/deployment/metrics_dashboard.py
```

输出示例：
```
╔════════════════════════════════════════════════════════════════╗
║           vLLM 服务实时监控仪表板                              ║
╠════════════════════════════════════════════════════════════════╣
║  📊 请求状态                                                   ║
║     等待中: 0          运行中: 1          总请求: 15           ║
║     并发使用: [██████░░░░░░░░░░░░░░] 1/3                       ║
╠════════════════════════════════════════════════════════════════╣
║  ⏱️  SLO 指标                                                   ║
║     TTFT: Mean: 0.68s  P50: 0.65s  P95: 0.85s                 ║
║     Throughput: Mean: 3.2 tok/s  Total: 2.9 tok/s             ║
╚════════════════════════════════════════════════════════════════╝
```

### Grafana Dashboard

导入以下Prometheus查询：

```promql
# TTFT P95
vllm_ttft_p95

# 吞吐量
vllm_decoding_throughput_mean

# 请求队列
vllm_num_waiting_requests + vllm_num_running_requests

# 成功率
(vllm_successful_requests / vllm_total_requests) * 100
```

---

## Python 监控示例

```python
import requests
import time

def monitor_service(url="http://10.9.3.131:8000"):
    """监控服务状态"""
    while True:
        try:
            r = requests.get(f"{url}/api/v1/metrics", timeout=5)
            m = r.json()
            
            print(f"状态: {m['status']}")
            print(f"等待: {m['requests']['waiting']} | 运行: {m['requests']['running']}")
            print(f"TTFT P95: {m['slo']['ttft']['p95']:.2f}s")
            print(f"吞吐量: {m['slo']['throughput']['decoding_mean']:.1f} tok/s")
            print(f"GPU: {m['gpu']['utilization_percent']:.1f}%")
            
            # 检查告警
            if m['alerts_count'] > 0:
                for alert in m['alerts']:
                    print(f"⚠️ {alert['message']}")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor_service()
```

---

## 性能基准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| TTFT P95 | < 1.0s | 首Token延迟 |
| Throughput | > 3 tok/s | 解码吞吐量 |
| 成功率 | > 99% | 请求成功率 |
| GPU使用率 | < 90% | 避免OOM |
