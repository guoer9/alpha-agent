# FinBERT 接口调用文档

## 服务信息

| 项目 | 值 |
|------|-----|
| 服务地址 | `http://localhost:8888` |
| 协议 | HTTP/HTTPS |
| 数据格式 | JSON |
| 字符编码 | UTF-8 |

---

## 接口列表

| 接口 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 健康检查 | GET | `/health` | 检查服务状态 |
| 单条分析 | POST | `/predict` | 分析单条文本 |
| 批量分析 | POST | `/predict/batch` | 批量分析多条文本 |
| API文档 | GET | `/docs` | Swagger UI |

---

## 1. 健康检查

检查服务是否正常运行。

### 请求

```http
GET /health
```

### 响应

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

### 示例

```bash
curl http://localhost:8888/health
```

```python
import requests
resp = requests.get("http://localhost:8888/health")
print(resp.json())
```

---

## 2. 单条文本情绪分析

### 请求

```http
POST /predict
Content-Type: application/json
```

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | ✓ | 待分析文本，1-512字符 |

### 请求示例

```json
{
  "text": "股市大涨，投资者信心增强"
}
```

### 响应参数

| 参数 | 类型 | 说明 |
|------|------|------|
| text | string | 原始文本 |
| sentiment | string | 情绪标签(英文): Positive/Negative/Neutral |
| sentiment_zh | string | 情绪标签(中文): 正面/负面/中性 |
| confidence | float | 置信度 (0-1) |
| probabilities | object | 各类别概率 |

### 响应示例

```json
{
  "text": "股市大涨，投资者信心增强",
  "sentiment": "Positive",
  "sentiment_zh": "正面",
  "confidence": 0.9998,
  "probabilities": {
    "Negative": 0.00003,
    "Neutral": 0.00014,
    "Positive": 0.9998
  }
}
```

### 调用示例

**cURL**
```bash
curl -X POST "http://localhost:8888/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "股市大涨，投资者信心增强"}'
```

**Python**
```python
import requests

response = requests.post(
    "http://localhost:8888/predict",
    json={"text": "股市大涨，投资者信心增强"}
)
result = response.json()

print(f"情绪: {result['sentiment_zh']}")
print(f"置信度: {result['confidence']:.2%}")
```

**JavaScript**
```javascript
fetch("http://localhost:8888/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "股市大涨，投资者信心增强" })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 3. 批量文本情绪分析

### 请求

```http
POST /predict/batch
Content-Type: application/json
```

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| texts | array | ✓ | 文本列表 |

### 请求示例

```json
{
  "texts": [
    "利好消息推动股价上涨",
    "公司业绩大幅下滑",
    "市场维持震荡格局"
  ]
}
```

### 响应示例

```json
{
  "results": [
    {
      "text": "利好消息推动股价上涨",
      "sentiment": "Positive",
      "sentiment_zh": "正面",
      "confidence": 0.9987,
      "probabilities": {...}
    },
    {
      "text": "公司业绩大幅下滑",
      "sentiment": "Negative",
      "sentiment_zh": "负面",
      "confidence": 0.9956,
      "probabilities": {...}
    },
    {
      "text": "市场维持震荡格局",
      "sentiment": "Neutral",
      "sentiment_zh": "中性",
      "confidence": 0.8234,
      "probabilities": {...}
    }
  ],
  "total": 3
}
```

### 调用示例

**cURL**
```bash
curl -X POST "http://localhost:8888/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "利好消息推动股价上涨",
      "公司业绩大幅下滑",
      "市场维持震荡格局"
    ]
  }'
```

**Python**
```python
import requests

texts = [
    "利好消息推动股价上涨",
    "公司业绩大幅下滑", 
    "市场维持震荡格局"
]

response = requests.post(
    "http://localhost:8888/predict/batch",
    json={"texts": texts}
)
data = response.json()

for result in data["results"]:
    print(f"{result['text'][:20]}... -> {result['sentiment_zh']} ({result['confidence']:.2%})")
```

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 422 | 参数验证失败 |
| 500 | 服务器内部错误 |
| 503 | 模型未加载 |

### 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

### 常见错误

**参数验证失败 (422)**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**文本过长 (422)**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "ensure this value has at most 512 characters",
      "type": "value_error.any_str.max_length"
    }
  ]
}
```

---

## 完整 Python 示例

```python
"""FinBERT 情绪分析客户端示例"""
import requests
from typing import List, Dict

class FinBERTClient:
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
    
    def health(self) -> Dict:
        """健康检查"""
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()
    
    def predict(self, text: str) -> Dict:
        """单条文本分析"""
        resp = requests.post(
            f"{self.base_url}/predict",
            json={"text": text}
        )
        resp.raise_for_status()
        return resp.json()
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """批量文本分析"""
        resp = requests.post(
            f"{self.base_url}/predict/batch",
            json={"texts": texts}
        )
        resp.raise_for_status()
        return resp.json()["results"]


if __name__ == "__main__":
    client = FinBERTClient()
    
    # 健康检查
    print("服务状态:", client.health())
    
    # 单条分析
    result = client.predict("央行降息利好股市")
    print(f"\n单条分析: {result['sentiment_zh']} ({result['confidence']:.2%})")
    
    # 批量分析
    texts = [
        "公司利润创历史新高",
        "股价暴跌引发恐慌",
        "市场观望情绪浓厚"
    ]
    results = client.predict_batch(texts)
    print("\n批量分析:")
    for r in results:
        print(f"  {r['text'][:15]}... -> {r['sentiment_zh']}")
```

---

## 性能参考

| 指标 | GPU | CPU |
|------|-----|-----|
| 单条延迟 | ~10ms | ~50ms |
| 批量吞吐 | ~500条/秒 | ~100条/秒 |
| 并发建议 | 10-20 | 5-10 |
