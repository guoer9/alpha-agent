#!/usr/bin/env python3
"""
Qwen3-8B金融新闻分类服务
支持8-bit量化，适配10GB显存
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import argparse
import threading
from queue import Queue
import time

try:
    from config import *
except ImportError:
    # 默认配置
    MODEL_PATH = "./models/qwen-news-classifier-merged"
    MAX_CONCURRENT_REQUESTS = 3
    REQUEST_QUEUE_SIZE = 10
    RATE_LIMIT_PER_MINUTE = 10
    MAX_TOKENS = 512
    HOST = "0.0.0.0"
    PORT = 8000

# 导入metrics
from scripts.deployment.metrics import metrics_collector

app = Flask(__name__)

# 速率限制器配置
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{RATE_LIMIT_PER_MINUTE} per minute"],
    storage_uri="memory://"
)

# 全局变量
model = None
tokenizer = None

# 请求队列配置
request_queue = Queue(maxsize=REQUEST_QUEUE_SIZE)
request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# 统计信息
stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "queue_full_count": 0,
    "current_queue_size": 0,
    "peak_queue_size": 0
}

def load_model_8bit(model_path: str):
    """加载8-bit量化模型"""
    global model, tokenizer
    
    print("=" * 60)
    print("加载模型（8-bit量化）")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"量化方式: 8-bit (适配10GB显存)")
    print(f"最大并发请求: {MAX_CONCURRENT_REQUESTS}")
    print(f"请求队列大小: {request_queue.maxsize}")
    print("=" * 60)
    
    # 8-bit量化配置（启用CPU offload）
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    print("\n步骤1: 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print("✓ Tokenizer加载成功")
    
    print("\n步骤2: 加载模型（8-bit量化）...")
    print("这可能需要几分钟...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("✓ 模型加载成功")
    
    # 检查显存使用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n显存使用:")
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已保留: {reserved:.2f} GB")
    
    print("\n" + "=" * 60)
    print("模型加载完成！")
    print("=" * 60)

def generate_text(prompt: str, max_tokens: int, temperature: float):
    """生成文本（带并发控制）"""
    with request_semaphore:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

@app.route('/v1/completions', methods=['POST'])
@limiter.limit("10 per minute")
def completions():
    """OpenAI兼容的completions接口（带速率限制和metrics）"""
    request_id = f"req-{int(time.time() * 1000)}-{hash(str(request.json))}"
    
    # 检查队列是否已满
    if request_queue.full():
        stats["queue_full_count"] += 1
        return jsonify({
            "error": "服务繁忙，请稍后重试",
            "queue_size": request_queue.qsize(),
            "max_queue_size": request_queue.maxsize
        }), 503
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = min(data.get('max_tokens', 100), 512)
        temperature = data.get('temperature', 0.7)
        
        # 开始metrics收集
        input_tokens = len(tokenizer.encode(prompt))
        req_metrics = metrics_collector.start_request(request_id, input_tokens)
        
        # 生成
        generated_text = generate_text(prompt, max_tokens, temperature)
        
        # 计算输出tokens
        output_tokens = len(tokenizer.encode(generated_text)) - input_tokens
        
        # 结束metrics收集
        metrics_collector.end_request(request_id, output_tokens)
        
        stats["successful_requests"] += 1
        stats["total_requests"] += 1
        
        return jsonify({
            "id": "cmpl-" + str(hash(prompt)),
            "object": "text_completion",
            "created": int(time.time()),
            "model": "qwen-news-classifier",
            "choices": [{
                "text": generated_text[len(prompt):],
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        })
    
    except Exception as e:
        stats["failed_requests"] += 1
        stats["total_requests"] += 1
        metrics_collector.end_request(request_id, 0)
        return jsonify({"error": str(e)}), 500

@app.route('/v1/chat/completions', methods=['POST'])
@limiter.limit("10 per minute")
def chat_completions():
    """OpenAI兼容的chat completions接口（带速率限制和metrics）"""
    request_id = f"req-{int(time.time() * 1000)}-{hash(str(request.json))}"
    
    # 检查队列是否已满
    if request_queue.full():
        stats["queue_full_count"] += 1
        return jsonify({
            "error": "服务繁忙，请稍后重试",
            "queue_size": request_queue.qsize(),
            "max_queue_size": request_queue.maxsize
        }), 503
    
    try:
        data = request.json
        messages = data.get('messages', [])
        max_tokens = min(data.get('max_tokens', 100), 512)
        temperature = data.get('temperature', 0.7)
        
        # 应用chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 开始metrics收集
        input_tokens = len(tokenizer.encode(text))
        req_metrics = metrics_collector.start_request(request_id, input_tokens)
        
        # 生成（第一个token时标记）
        generated_text = generate_text(text, max_tokens, temperature)
        
        # 提取生成的内容
        response_text = generated_text[len(text):]
        output_tokens = len(tokenizer.encode(response_text))
        
        # 结束metrics收集
        metrics_collector.end_request(request_id, output_tokens)
        
        stats["successful_requests"] += 1
        stats["total_requests"] += 1
        
        return jsonify({
            "id": "chatcmpl-" + str(hash(str(messages))),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen-news-classifier",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(tokenizer.encode(text)),
                "completion_tokens": max_tokens,
                "total_tokens": len(tokenizer.encode(text)) + max_tokens
            }
        })
    
    except Exception as e:
        stats["failed_requests"] += 1
        stats["total_requests"] += 1
        metrics_collector.end_request(request_id, 0)
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "qwen-news-classifier",
            "object": "model",
            "created": 0,
            "owned_by": "user"
        }]
    })

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - allocated
    else:
        allocated = reserved = free = 0
    
    return jsonify({
        "status": "ok",
        "gpu": {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2)
        },
        "limits": {
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "max_queue_size": request_queue.maxsize,
            "rate_limit": "10 requests/minute per IP"
        }
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """获取服务统计信息"""
    return jsonify({
        "statistics": stats,
        "current_queue_size": request_queue.qsize(),
        "available_slots": MAX_CONCURRENT_REQUESTS - (request_queue.qsize() if request_queue.qsize() < MAX_CONCURRENT_REQUESTS else MAX_CONCURRENT_REQUESTS)
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """获取详细metrics（包括SLO指标）"""
    all_metrics = metrics_collector.get_all_metrics()
    return jsonify(all_metrics)

@app.route('/metrics/prometheus', methods=['GET'])
def prometheus_metrics():
    """Prometheus格式metrics"""
    current = metrics_collector.get_current_metrics()
    slo = metrics_collector.get_slo_metrics()
    
    lines = [
        "# HELP vllm_num_waiting_requests Number of requests waiting in queue",
        "# TYPE vllm_num_waiting_requests gauge",
        f"vllm_num_waiting_requests {current['num_waiting_requests']}",
        "",
        "# HELP vllm_num_running_requests Number of requests currently running",
        "# TYPE vllm_num_running_requests gauge",
        f"vllm_num_running_requests {current['num_running_requests']}",
        "",
        "# HELP vllm_total_requests Total number of requests",
        "# TYPE vllm_total_requests counter",
        f"vllm_total_requests {current['total_requests']}",
        "",
        "# HELP vllm_ttft_mean Mean time to first token in seconds",
        "# TYPE vllm_ttft_mean gauge",
        f"vllm_ttft_mean {slo['ttft_mean']}",
        "",
        "# HELP vllm_ttft_p50 P50 time to first token in seconds",
        "# TYPE vllm_ttft_p50 gauge",
        f"vllm_ttft_p50 {slo['ttft_p50']}",
        "",
        "# HELP vllm_ttft_p95 P95 time to first token in seconds",
        "# TYPE vllm_ttft_p95 gauge",
        f"vllm_ttft_p95 {slo['ttft_p95']}",
        "",
        "# HELP vllm_ttft_p99 P99 time to first token in seconds",
        "# TYPE vllm_ttft_p99 gauge",
        f"vllm_ttft_p99 {slo['ttft_p99']}",
        "",
        "# HELP vllm_decoding_throughput_mean Mean decoding throughput in tokens/sec",
        "# TYPE vllm_decoding_throughput_mean gauge",
        f"vllm_decoding_throughput_mean {slo['decoding_throughput_mean']}",
        "",
        "# HELP vllm_decoding_throughput_p50 P50 decoding throughput in tokens/sec",
        "# TYPE vllm_decoding_throughput_p50 gauge",
        f"vllm_decoding_throughput_p50 {slo['decoding_throughput_p50']}",
        "",
        "# HELP vllm_total_throughput Total throughput in tokens/sec",
        "# TYPE vllm_total_throughput gauge",
        f"vllm_total_throughput {slo['total_throughput']}",
        "",
    ]
    
    return "\n".join(lines), 200, {'Content-Type': 'text/plain; charset=utf-8'}

def main():
    parser = argparse.ArgumentParser(description="部署Qwen模型（8-bit量化，带并发限制）")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/qwen-news-classifier-merged",
        help="模型路径"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Gunicorn worker数量"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    load_model_8bit(args.model_path)
    
    # 启动服务
    print(f"\n启动服务: http://{args.host}:{args.port}")
    print(f"API文档: http://localhost:{args.port}/v1/models")
    print(f"健康检查: http://localhost:{args.port}/health")
    print(f"统计信息: http://localhost:{args.port}/stats")
    print(f"\n并发限制:")
    print(f"  - 最大并发请求: {MAX_CONCURRENT_REQUESTS}")
    print(f"  - 请求队列大小: {request_queue.maxsize}")
    print(f"  - 速率限制: 10请求/分钟 (每IP)")
    print(f"  - 最大token数: 512")
    print("\n按 Ctrl+C 停止服务\n")
    
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
