#!/usr/bin/env python3
"""
vLLM服务Metrics实时监控脚本
监控核心指标：请求队列、TTFT、Decoding Throughput
"""

import requests
import time
import sys
from datetime import datetime

class MetricsMonitor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.metrics_url = f"{base_url}/metrics"
        
    def get_metrics(self):
        """获取metrics数据"""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            return response.json()
        except Exception as e:
            return None
    
    def print_metrics(self, metrics):
        """打印格式化的metrics"""
        if not metrics:
            print("❌ 无法获取metrics")
            return
        
        # 清屏
        print("\033[2J\033[H", end="")
        
        # 标题
        print("=" * 70)
        print(f"{'vLLM服务Metrics监控':^70}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^70}")
        print("=" * 70)
        
        # 请求队列状态
        print("\n📊 请求队列状态")
        print("-" * 70)
        waiting = metrics.get('num_waiting_requests', 0)
        running = metrics.get('num_running_requests', 0)
        total = metrics.get('total_requests', 0)
        
        print(f"  等待请求: {waiting:>3} {'⚠️  队列积压' if waiting > 5 else '✓'}")
        print(f"  运行请求: {running:>3} / 3")
        print(f"  总请求数: {total:>3}")
        print(f"  生成Token: {metrics.get('total_tokens_generated', 0):>6}")
        
        # SLO指标
        slo = metrics.get('slo', {})
        if slo and slo.get('sample_size', 0) > 0:
            print("\n⏱️  SLO指标 (Service Level Objectives)")
            print("-" * 70)
            
            # TTFT
            ttft_p50 = slo.get('ttft_p50', 0)
            ttft_p95 = slo.get('ttft_p95', 0)
            ttft_p99 = slo.get('ttft_p99', 0)
            
            print(f"  Time to First Token (TTFT):")
            print(f"    P50: {ttft_p50:>6.3f}s  {'✓' if ttft_p50 < 1.0 else '⚠️'}")
            print(f"    P95: {ttft_p95:>6.3f}s  {'✓' if ttft_p95 < 2.0 else '⚠️'}")
            print(f"    P99: {ttft_p99:>6.3f}s  {'✓' if ttft_p99 < 3.0 else '⚠️'}")
            
            # Decoding Throughput
            dec_mean = slo.get('decoding_throughput_mean', 0)
            dec_p50 = slo.get('decoding_throughput_p50', 0)
            dec_p95 = slo.get('decoding_throughput_p95', 0)
            
            print(f"\n  Decoding Throughput (tokens/sec):")
            print(f"    Mean: {dec_mean:>6.1f}  {'✓' if dec_mean > 20 else '⚠️'}")
            print(f"    P50:  {dec_p50:>6.1f}")
            print(f"    P95:  {dec_p95:>6.1f}")
            
            # Total Throughput
            total_tp = slo.get('total_throughput', 0)
            print(f"\n  Total Throughput: {total_tp:>6.1f} tokens/sec  {'✓' if total_tp > 50 else '⚠️'}")
            
            print(f"\n  样本数量: {slo.get('sample_size', 0)}")
        else:
            print("\n⏱️  SLO指标")
            print("-" * 70)
            print("  暂无数据 (需要至少1个完成的请求)")
        
        # 告警检查
        alerts = []
        if waiting > 5:
            alerts.append(f"⚠️  队列积压: {waiting}个请求等待")
        if slo.get('ttft_p95', 0) > 2.0:
            alerts.append(f"⚠️  TTFT P95过高: {slo.get('ttft_p95', 0):.3f}s")
        if slo.get('decoding_throughput_mean', 0) > 0 and slo.get('decoding_throughput_mean', 0) < 15:
            alerts.append(f"⚠️  吞吐量过低: {slo.get('decoding_throughput_mean', 0):.1f} tokens/s")
        
        if alerts:
            print("\n🚨 告警")
            print("-" * 70)
            for alert in alerts:
                print(f"  {alert}")
        
        print("\n" + "=" * 70)
        print("按 Ctrl+C 停止监控 | 刷新间隔: 5秒")
    
    def run(self, interval=5):
        """运行监控"""
        print("启动vLLM Metrics监控...")
        print(f"服务地址: {self.base_url}")
        print(f"刷新间隔: {interval}秒\n")
        
        try:
            while True:
                metrics = self.get_metrics()
                self.print_metrics(metrics)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            sys.exit(0)
        except Exception as e:
            print(f"\n错误: {e}")
            sys.exit(1)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Metrics监控")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="服务地址"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="刷新间隔(秒)"
    )
    
    args = parser.parse_args()
    
    monitor = MetricsMonitor(args.url)
    monitor.run(args.interval)

if __name__ == "__main__":
    main()
