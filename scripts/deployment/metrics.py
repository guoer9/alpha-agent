"""
vLLM服务核心Metrics监控
包括：请求队列、TTFT、Decoding Throughput等SLO指标
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List
from threading import Lock
import statistics

@dataclass
class RequestMetrics:
    """单个请求的metrics"""
    request_id: str
    start_time: float
    first_token_time: float = 0.0
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def time_to_first_token(self) -> float:
        """TTFT: 从请求到第一个token的时间(秒)"""
        if self.first_token_time > 0:
            return self.first_token_time - self.start_time
        return 0.0
    
    @property
    def total_time(self) -> float:
        """总处理时间(秒)"""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def decoding_throughput(self) -> float:
        """Decoding吞吐量: tokens/秒"""
        if self.end_time > 0 and self.first_token_time > 0:
            decode_time = self.end_time - self.first_token_time
            if decode_time > 0 and self.output_tokens > 1:
                # 减去第一个token，因为TTFT已经计算了
                return (self.output_tokens - 1) / decode_time
        return 0.0


class MetricsCollector:
    """Metrics收集器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.lock = Lock()
        
        # 当前状态
        self.num_waiting_requests = 0
        self.num_running_requests = 0
        
        # 历史记录
        self.completed_requests: List[RequestMetrics] = []
        self.active_requests: Dict[str, RequestMetrics] = {}
        
        # 统计数据
        self.total_requests = 0
        self.total_tokens_generated = 0
        
    def start_request(self, request_id: str, input_tokens: int = 0) -> RequestMetrics:
        """开始一个新请求"""
        with self.lock:
            metrics = RequestMetrics(
                request_id=request_id,
                start_time=time.time(),
                input_tokens=input_tokens
            )
            self.active_requests[request_id] = metrics
            self.num_running_requests += 1
            self.total_requests += 1
            return metrics
    
    def mark_first_token(self, request_id: str):
        """标记第一个token生成"""
        with self.lock:
            if request_id in self.active_requests:
                self.active_requests[request_id].first_token_time = time.time()
    
    def end_request(self, request_id: str, output_tokens: int = 0):
        """结束请求"""
        with self.lock:
            if request_id in self.active_requests:
                metrics = self.active_requests.pop(request_id)
                metrics.end_time = time.time()
                metrics.output_tokens = output_tokens
                
                # 保存到历史记录
                self.completed_requests.append(metrics)
                if len(self.completed_requests) > self.window_size:
                    self.completed_requests.pop(0)
                
                self.num_running_requests -= 1
                self.total_tokens_generated += output_tokens
    
    def add_waiting_request(self):
        """增加等待请求数"""
        with self.lock:
            self.num_waiting_requests += 1
    
    def remove_waiting_request(self):
        """减少等待请求数"""
        with self.lock:
            if self.num_waiting_requests > 0:
                self.num_waiting_requests -= 1
    
    def get_current_metrics(self) -> Dict:
        """获取当前metrics"""
        with self.lock:
            return {
                "num_waiting_requests": self.num_waiting_requests,
                "num_running_requests": self.num_running_requests,
                "total_requests": self.total_requests,
                "total_tokens_generated": self.total_tokens_generated,
            }
    
    def get_slo_metrics(self) -> Dict:
        """获取SLO相关metrics"""
        with self.lock:
            if not self.completed_requests:
                return {
                    "ttft_mean": 0.0,
                    "ttft_p50": 0.0,
                    "ttft_p95": 0.0,
                    "ttft_p99": 0.0,
                    "decoding_throughput_mean": 0.0,
                    "decoding_throughput_p50": 0.0,
                    "decoding_throughput_p95": 0.0,
                    "total_throughput": 0.0,
                }
            
            # 计算TTFT统计
            ttfts = [r.time_to_first_token for r in self.completed_requests if r.time_to_first_token > 0]
            
            # 计算Decoding Throughput统计
            throughputs = [r.decoding_throughput for r in self.completed_requests if r.decoding_throughput > 0]
            
            # 计算总吞吐量
            total_time = sum(r.total_time for r in self.completed_requests)
            total_tokens = sum(r.output_tokens for r in self.completed_requests)
            total_throughput = total_tokens / total_time if total_time > 0 else 0.0
            
            return {
                "ttft_mean": statistics.mean(ttfts) if ttfts else 0.0,
                "ttft_p50": statistics.median(ttfts) if ttfts else 0.0,
                "ttft_p95": self._percentile(ttfts, 95) if ttfts else 0.0,
                "ttft_p99": self._percentile(ttfts, 99) if ttfts else 0.0,
                "decoding_throughput_mean": statistics.mean(throughputs) if throughputs else 0.0,
                "decoding_throughput_p50": statistics.median(throughputs) if throughputs else 0.0,
                "decoding_throughput_p95": self._percentile(throughputs, 95) if throughputs else 0.0,
                "total_throughput": total_throughput,
                "sample_size": len(self.completed_requests),
            }
    
    def get_all_metrics(self) -> Dict:
        """获取所有metrics"""
        current = self.get_current_metrics()
        slo = self.get_slo_metrics()
        return {
            **current,
            "slo": slo
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


# 全局metrics收集器
metrics_collector = MetricsCollector()
