"""
Performance Analysis with Failure Injection
===========================================
Collects and presents metrics before, during, and after failures:
- Throughput (req/sec): Dip during failure, recovery afterward
- Latency (p95): Spike → return to baseline
- Recovery time: Seconds until normal behavior

Generates graphs:
- Latency vs Time
- Throughput vs Time
- Annotated with failures & recovery points
"""

import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import os
import sys
import subprocess
import signal

# Import load generator and disruption simulator
from input_load_generator_grpc import GrpcLoadGenerator
from disruption_simulator import DisruptionSimulator

BASE_PORT = 50051


@dataclass
class MetricSample:
    """Single metric sample at a point in time"""
    timestamp: float
    latency_ms: Optional[float]
    success: bool
    request_id: int


@dataclass
class FailureEvent:
    """Represents a failure event"""
    timestamp: float
    replica_id: int
    event_type: str  # 'FAILURE' or 'RECOVERY'
    description: str


class PerformanceMonitor:
    """Monitors performance metrics in real-time"""
    
    def __init__(self, window_size_seconds: float = 1.0):
        self.window_size = window_size_seconds
        self.samples: List[MetricSample] = []
        self.failure_events: List[FailureEvent] = []
        self.start_time: Optional[float] = None
        self.lock = threading.Lock()
        
    def add_sample(self, sample: MetricSample):
        """Add a metric sample"""
        with self.lock:
            if self.start_time is None:
                self.start_time = sample.timestamp
            self.samples.append(sample)
    
    def add_failure_event(self, event: FailureEvent):
        """Record a failure or recovery event"""
        with self.lock:
            self.failure_events.append(event)
    
    def get_time_series(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get time series data for plotting
        Returns: (time_points, latency_p95, throughput)
        """
        with self.lock:
            if not self.samples:
                return np.array([]), np.array([]), np.array([])
            
            # Group samples into time windows
            if self.start_time is None:
                return np.array([]), np.array([]), np.array([])
            
            # Find time range
            end_time = max(s.timestamp for s in self.samples)
            num_windows = int((end_time - self.start_time) / self.window_size) + 1
            
            if num_windows == 0:
                return np.array([]), np.array([]), np.array([])
            
            time_points = np.linspace(self.start_time, end_time, num_windows)
            latency_p95 = np.zeros(num_windows)
            throughput = np.zeros(num_windows)
            
            # Calculate metrics for each window
            for i, window_start in enumerate(time_points):
                window_end = window_start + self.window_size
                
                # Get samples in this window
                window_samples = [
                    s for s in self.samples
                    if window_start <= s.timestamp < window_end
                ]
                
                if window_samples:
                    # Calculate throughput (successful requests per second)
                    successful = [s for s in window_samples if s.success]
                    throughput[i] = len(successful) / self.window_size
                    
                    # Calculate p95 latency
                    latencies = [s.latency_ms for s in successful if s.latency_ms is not None]
                    if latencies:
                        latency_p95[i] = np.percentile(latencies, 95)
                    else:
                        latency_p95[i] = np.nan
                else:
                    latency_p95[i] = np.nan
                    throughput[i] = 0.0
            
            return time_points, latency_p95, throughput
    
    def get_baseline_metrics(self, before_failure_time: float) -> Dict[str, float]:
        """Get baseline metrics before first failure"""
        with self.lock:
            baseline_samples = [
                s for s in self.samples
                if s.timestamp < before_failure_time and s.success
            ]
            
            if not baseline_samples:
                return {'latency_p95': 0, 'throughput': 0}
            
            latencies = [s.latency_ms for s in baseline_samples if s.latency_ms is not None]
            if not latencies:
                return {'latency_p95': 0, 'throughput': 0}
            
            # Calculate average throughput
            if baseline_samples:
                time_span = baseline_samples[-1].timestamp - baseline_samples[0].timestamp
                throughput = len(baseline_samples) / time_span if time_span > 0 else 0
            else:
                throughput = 0
            
            return {
                'latency_p95': np.percentile(latencies, 95),
                'throughput': throughput
            }
    
    def calculate_recovery_times(self) -> List[Dict]:
        """Calculate recovery time for each failure"""
        with self.lock:
            if not self.failure_events:
                return []
            
            recovery_times = []
            failures = [e for e in self.failure_events if e.event_type == 'FAILURE']
            
            for failure in failures:
                # Find corresponding recovery
                recovery = next(
                    (e for e in self.failure_events 
                     if e.event_type == 'RECOVERY' and e.timestamp > failure.timestamp),
                    None
                )
                
                if recovery:
                    recovery_time = recovery.timestamp - failure.timestamp
                    
                    # Check when metrics returned to baseline
                    baseline = self.get_baseline_metrics(failure.timestamp)
                    
                    if baseline['latency_p95'] > 0 and baseline['throughput'] > 0:
                        # Find when metrics returned to baseline (within 20% tolerance)
                        recovery_samples = [
                            s for s in self.samples
                            if failure.timestamp <= s.timestamp <= recovery.timestamp + 30
                            and s.success
                        ]
                        
                        if recovery_samples:
                            # Check each window after recovery
                            window_size = 2.0  # 2 second windows
                            window_start_idx = 0
                            
                            while window_start_idx < len(recovery_samples):
                                window_end_idx = min(window_start_idx + int(window_size * 10), len(recovery_samples))
                                window = recovery_samples[window_start_idx:window_end_idx]
                                
                                if len(window) >= 5:
                                    window_latencies = [s.latency_ms for s in window if s.latency_ms is not None]
                                    if window_latencies:
                                        window_p95 = np.percentile(window_latencies, 95)
                                        window_throughput = len(window) / window_size
                                        
                                        # Check if within 20% of baseline (more lenient)
                                        latency_ok = abs(window_p95 - baseline['latency_p95']) / baseline['latency_p95'] < 0.2
                                        throughput_ok = abs(window_throughput - baseline['throughput']) / baseline['throughput'] < 0.2
                                        
                                        if latency_ok and throughput_ok:
                                            metric_recovery_time = window[0].timestamp - failure.timestamp
                                            recovery_times.append({
                                                'failure_time': failure.timestamp,
                                                'recovery_time': recovery_time,
                                                'metric_recovery_time': metric_recovery_time,
                                                'replica_id': failure.replica_id,
                                                'description': failure.description
                                            })
                                            break
                                
                                window_start_idx += int(window_size * 5)  # Overlap windows
                            
                            # If no metric recovery found, use recovery time
                            if not any(rt['failure_time'] == failure.timestamp for rt in recovery_times):
                                recovery_times.append({
                                    'failure_time': failure.timestamp,
                                    'recovery_time': recovery_time,
                                    'metric_recovery_time': recovery_time,
                                    'replica_id': failure.replica_id,
                                    'description': failure.description
                                })
                        else:
                            recovery_times.append({
                                'failure_time': failure.timestamp,
                                'recovery_time': recovery_time,
                                'metric_recovery_time': recovery_time,
                                'replica_id': failure.replica_id,
                                'description': failure.description
                            })
                    else:
                        # No baseline available, use recovery time
                        recovery_times.append({
                            'failure_time': failure.timestamp,
                            'recovery_time': recovery_time,
                            'metric_recovery_time': recovery_time,
                            'replica_id': failure.replica_id,
                            'description': failure.description
                        })
            
            return recovery_times


class FailurePerformanceAnalyzer:
    """Main analyzer that orchestrates load generation, failure injection, and metric collection"""
    
    def __init__(self, 
                 num_replicas: int = 3,
                 requests_per_second: float = 10.0,
                 duration_seconds: float = 120.0,
                 failure_scenarios: Optional[List[Dict]] = None):
        self.num_replicas = num_replicas
        self.requests_per_second = requests_per_second
        self.duration_seconds = duration_seconds
        self.failure_scenarios = failure_scenarios or self._default_failure_scenarios()
        
        self.monitor = PerformanceMonitor()
        self.load_generator: Optional[GrpcLoadGenerator] = None
        self.disruption_simulator: Optional[DisruptionSimulator] = None
        self.load_thread: Optional[threading.Thread] = None
        self.failure_thread: Optional[threading.Thread] = None
        
    def _default_failure_scenarios(self) -> List[Dict]:
        """Default failure scenarios if none provided"""
        return [
            {'type': 'kill', 'replica_id': 0, 'time': 30.0, 'recover': True, 'recover_after': 10.0},
            {'type': 'kill', 'replica_id': 1, 'time': 60.0, 'recover': True, 'recover_after': 8.0},
        ]
    
    def _collect_metrics_from_load_generator(self):
        """Thread function to collect metrics from load generator"""
        if not self.load_generator:
            return
        
        queue = self.load_generator.get_queue()
        
        while self.load_generator.running or not queue.empty():
            try:
                request = queue.get(timeout=0.5)
                
                # Create metric sample
                sample = MetricSample(
                    timestamp=request.timestamp,
                    latency_ms=request.grpc_response.get('measured_latency_ms') if request.grpc_response else None,
                    success=request.grpc_success or False,
                    request_id=request.request_id
                )
                
                self.monitor.add_sample(sample)
                
            except:
                # Queue empty or timeout - continue checking
                if not self.load_generator.running:
                    break
                continue
    
    def _run_failure_scenarios(self):
        """Thread function to inject failures according to scenarios"""
        # Wait for system to stabilize
        time.sleep(5)
        
        # Use absolute time from when this thread starts
        thread_start_time = time.time()
        
        for scenario in self.failure_scenarios:
            # Wait until scenario time (relative to thread start)
            scenario_time = scenario['time']
            current_time = time.time()
            elapsed = current_time - thread_start_time
            
            wait_time = scenario_time - elapsed
            if wait_time > 0:
                print(f"[FAILURE INJECTOR] Waiting {wait_time:.1f}s before injecting failure at {scenario_time}s...")
                time.sleep(wait_time)
            
            replica_id = scenario['replica_id']
            event_type = scenario['type']
            
            # Record failure event
            failure_event = FailureEvent(
                timestamp=time.time(),
                replica_id=replica_id,
                event_type='FAILURE',
                description=f"{event_type} replica {replica_id}"
            )
            self.monitor.add_failure_event(failure_event)
            print(f"[FAILURE INJECTOR] Injected FAILURE: {failure_event.description} at {failure_event.timestamp:.2f}")
            
            # Inject failure
            if event_type == 'kill' and self.disruption_simulator:
                self.disruption_simulator.kill_replica(replica_id)
            
            # Wait for recovery if specified
            if scenario.get('recover', False) and self.disruption_simulator:
                recover_after = scenario.get('recover_after', 10.0)
                print(f"[FAILURE INJECTOR] Waiting {recover_after}s before recovery...")
                time.sleep(recover_after)
                
                # Restart replica
                self.disruption_simulator.restart_replica(replica_id)
                
                # Record recovery event
                recovery_event = FailureEvent(
                    timestamp=time.time(),
                    replica_id=replica_id,
                    event_type='RECOVERY',
                    description=f"recovered replica {replica_id}"
                )
                self.monitor.add_failure_event(recovery_event)
                print(f"[FAILURE INJECTOR] Recorded RECOVERY: {recovery_event.description} at {recovery_event.timestamp:.2f}")
    
    def run_analysis(self):
        """Run the complete performance analysis"""
        print("=" * 80)
        print("Performance Analysis with Failure Injection")
        print("=" * 80)
        print(f"Replicas: {self.num_replicas}")
        print(f"Request rate: {self.requests_per_second} req/s")
        print(f"Duration: {self.duration_seconds} seconds")
        print(f"Failure scenarios: {len(self.failure_scenarios)}")
        print("=" * 80)
        print()
        
        # Setup disruption simulator
        self.disruption_simulator = DisruptionSimulator(self.num_replicas)
        
        # Launch replicas
        print("Launching replicas...")
        self.disruption_simulator.launch_all_replicas()
        time.sleep(3)  # Wait for replicas to be ready
        
        # Setup load generator
        grpc_targets = [f"localhost:{BASE_PORT + i}" for i in range(self.num_replicas)]
        self.load_generator = GrpcLoadGenerator(
            requests_per_second=self.requests_per_second,
            duration_seconds=self.duration_seconds,
            grpc_targets=grpc_targets,
            timeout=5.0
        )
        
        # Start metric collection thread
        self.load_thread = threading.Thread(target=self._collect_metrics_from_load_generator, daemon=True)
        self.load_thread.start()
        
        # Start failure injection thread
        self.failure_thread = threading.Thread(target=self._run_failure_scenarios, daemon=True)
        self.failure_thread.start()
        
        # Run load generator (this blocks until complete)
        print("Starting load generation...")
        self.load_generator.run()
        
        # Wait a bit for any remaining metrics to be collected
        time.sleep(2)
        
        # Wait for threads to finish
        if self.failure_thread.is_alive():
            self.failure_thread.join(timeout=5)
        
        # Collect any remaining metrics from queue
        queue = self.load_generator.get_queue()
        while not queue.empty():
            try:
                request = queue.get_nowait()
                sample = MetricSample(
                    timestamp=request.timestamp,
                    latency_ms=request.grpc_response.get('measured_latency_ms') if request.grpc_response else None,
                    success=request.grpc_success or False,
                    request_id=request.request_id
                )
                self.monitor.add_sample(sample)
            except:
                break
        
        print("\nAnalysis complete!")
        print()
    
    def generate_plots(self, output_dir: str = "plots"):
        """Generate performance plots with failure annotations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get time series data
        time_points, latency_p95, throughput = self.monitor.get_time_series()
        
        if len(time_points) == 0:
            print("No data collected. Cannot generate plots.")
            return
        
        # Normalize time to start from 0
        time_normalized = time_points - time_points[0]
        
        # Get failure events
        failure_events = [e for e in self.monitor.failure_events if e.event_type == 'FAILURE']
        recovery_events = [e for e in self.monitor.failure_events if e.event_type == 'RECOVERY']
        
        # Normalize event times relative to first data point
        if failure_events:
            failure_times = [(e.timestamp - time_points[0]) for e in failure_events]
            print(f"[PLOT] Found {len(failure_times)} failure events at times: {failure_times}")
        else:
            failure_times = []
            print("[PLOT] No failure events found - no annotations will be added")
        
        if recovery_events:
            recovery_times = [(e.timestamp - time_points[0]) for e in recovery_events]
            print(f"[PLOT] Found {len(recovery_times)} recovery events at times: {recovery_times}")
        else:
            recovery_times = []
            print("[PLOT] No recovery events found - no recovery annotations will be added")
        
        # Plot 1: Latency vs Time
        plt.figure(figsize=(14, 6))
        plt.plot(time_normalized, latency_p95, 'b-', linewidth=2, label='P95 Latency')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Latency (ms)', fontsize=12)
        plt.title('Latency vs Time (P95)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Annotate failures
        for i, fail_time in enumerate(failure_times):
            if 0 <= fail_time <= time_normalized[-1]:
                plt.axvline(x=fail_time, color='r', linestyle='--', linewidth=2, alpha=0.7)
                plt.annotate(f'Failure {i+1}', 
                            xy=(fail_time, plt.ylim()[1] * 0.9),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        # Annotate recoveries
        for i, rec_time in enumerate(recovery_times):
            if 0 <= rec_time <= time_normalized[-1]:
                plt.axvline(x=rec_time, color='g', linestyle='--', linewidth=2, alpha=0.7)
                plt.annotate(f'Recovery {i+1}', 
                            xy=(rec_time, plt.ylim()[1] * 0.8),
                            xytext=(10, -20),
                            textcoords='offset points',
                            fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
        
        plt.tight_layout()
        latency_plot_path = os.path.join(output_dir, 'latency_vs_time.png')
        plt.savefig(latency_plot_path, dpi=300, bbox_inches='tight')
        print(f"Latency plot saved to: {latency_plot_path}")
        plt.close()
        
        # Plot 2: Throughput vs Time
        plt.figure(figsize=(14, 6))
        plt.plot(time_normalized, throughput, 'g-', linewidth=2, label='Throughput')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Throughput (req/sec)', fontsize=12)
        plt.title('Throughput vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Annotate failures
        for i, fail_time in enumerate(failure_times):
            if 0 <= fail_time <= time_normalized[-1]:
                plt.axvline(x=fail_time, color='r', linestyle='--', linewidth=2, alpha=0.7)
                plt.annotate(f'Failure {i+1}', 
                            xy=(fail_time, plt.ylim()[1] * 0.9),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        # Annotate recoveries
        for i, rec_time in enumerate(recovery_times):
            if 0 <= rec_time <= time_normalized[-1]:
                plt.axvline(x=rec_time, color='g', linestyle='--', linewidth=2, alpha=0.7)
                plt.annotate(f'Recovery {i+1}', 
                            xy=(rec_time, plt.ylim()[1] * 0.8),
                            xytext=(10, -20),
                            textcoords='offset points',
                            fontsize=10,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                            arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
        
        plt.tight_layout()
        throughput_plot_path = os.path.join(output_dir, 'throughput_vs_time.png')
        plt.savefig(throughput_plot_path, dpi=300, bbox_inches='tight')
        print(f"Throughput plot saved to: {throughput_plot_path}")
        plt.close()
        
        # Combined plot with both metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Latency subplot
        ax1.plot(time_normalized, latency_p95, 'b-', linewidth=2, label='P95 Latency')
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_title('Performance Metrics vs Time (with Failure Annotations)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Throughput subplot
        ax2.plot(time_normalized, throughput, 'g-', linewidth=2, label='Throughput')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Throughput (req/sec)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        # Annotate failures on both subplots
        for i, fail_time in enumerate(failure_times):
            if 0 <= fail_time <= time_normalized[-1]:
                ax1.axvline(x=fail_time, color='r', linestyle='--', linewidth=2, alpha=0.7)
                ax2.axvline(x=fail_time, color='r', linestyle='--', linewidth=2, alpha=0.7)
                ax1.annotate(f'Failure {i+1}', 
                            xy=(fail_time, ax1.get_ylim()[1] * 0.9),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
                ax2.annotate(f'Failure {i+1}', 
                            xy=(fail_time, ax2.get_ylim()[1] * 0.9),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
        
        # Annotate recoveries on both subplots
        for i, rec_time in enumerate(recovery_times):
            if 0 <= rec_time <= time_normalized[-1]:
                ax1.axvline(x=rec_time, color='g', linestyle='--', linewidth=2, alpha=0.7)
                ax2.axvline(x=rec_time, color='g', linestyle='--', linewidth=2, alpha=0.7)
                ax1.annotate(f'Recovery {i+1}', 
                            xy=(rec_time, ax1.get_ylim()[1] * 0.8),
                            xytext=(10, -20),
                            textcoords='offset points',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3))
                ax2.annotate(f'Recovery {i+1}', 
                            xy=(rec_time, ax2.get_ylim()[1] * 0.8),
                            xytext=(10, -20),
                            textcoords='offset points',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3))
        
        plt.tight_layout()
        combined_plot_path = os.path.join(output_dir, 'performance_with_failures.png')
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {combined_plot_path}")
        plt.close()
    
    def print_summary(self):
        """Print summary of metrics and recovery times"""
        print("\n" + "=" * 80)
        print("Performance Analysis Summary")
        print("=" * 80)
        
        # Get baseline metrics
        failure_events = [e for e in self.monitor.failure_events if e.event_type == 'FAILURE']
        if failure_events:
            first_failure_time = min(e.timestamp for e in failure_events)
            baseline = self.monitor.get_baseline_metrics(first_failure_time)
            
            print(f"\nBaseline Metrics (before failures):")
            print(f"  P95 Latency: {baseline['latency_p95']:.2f} ms")
            print(f"  Throughput: {baseline['throughput']:.2f} req/sec")
        
        # Get recovery times
        recovery_times = self.monitor.calculate_recovery_times()
        
        if recovery_times:
            print(f"\nRecovery Times:")
            print(f"{'Failure Time':>15} | {'Recovery Time (s)':>20} | {'Metric Recovery (s)':>20} | {'Replica':>10}")
            print("-" * 80)
            for rt in recovery_times:
                print(f"{rt['failure_time']:>15.1f} | {rt['recovery_time']:>20.2f} | {rt['metric_recovery_time']:>20.2f} | {rt['replica_id']:>10}")
            
            avg_recovery = np.mean([rt['recovery_time'] for rt in recovery_times])
            avg_metric_recovery = np.mean([rt['metric_recovery_time'] for rt in recovery_times])
            print(f"\nAverage Recovery Time: {avg_recovery:.2f} seconds")
            print(f"Average Metric Recovery Time: {avg_metric_recovery:.2f} seconds")
        else:
            print("\nNo recovery events recorded")
        
        # Overall statistics
        time_points, latency_p95, throughput = self.monitor.get_time_series()
        if len(time_points) > 0:
            print(f"\nOverall Statistics:")
            valid_latency = latency_p95[~np.isnan(latency_p95)]
            if len(valid_latency) > 0:
                print(f"  Min P95 Latency: {np.min(valid_latency):.2f} ms")
                print(f"  Max P95 Latency: {np.max(valid_latency):.2f} ms")
                print(f"  Avg P95 Latency: {np.mean(valid_latency):.2f} ms")
            
            print(f"  Min Throughput: {np.min(throughput):.2f} req/sec")
            print(f"  Max Throughput: {np.max(throughput):.2f} req/sec")
            print(f"  Avg Throughput: {np.mean(throughput):.2f} req/sec")
        
        print("=" * 80)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.disruption_simulator:
            self.disruption_simulator.cleanup()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Performance Analysis with Failure Injection'
    )
    parser.add_argument(
        '--replicas', '-n',
        type=int,
        default=3,
        help='Number of replicas (default: 3)'
    )
    parser.add_argument(
        '--rate', '-r',
        type=float,
        default=10.0,
        help='Requests per second (default: 10.0)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=120.0,
        help='Duration in seconds (default: 120.0)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots)'
    )
    parser.add_argument(
        '--scenarios', '-s',
        type=str,
        default=None,
        help='JSON file with failure scenarios (optional)'
    )
    
    args = parser.parse_args()
    
    # Load failure scenarios if provided
    failure_scenarios = None
    if args.scenarios:
        with open(args.scenarios, 'r') as f:
            failure_scenarios = json.load(f)
    
    # Create analyzer
    analyzer = FailurePerformanceAnalyzer(
        num_replicas=args.replicas,
        requests_per_second=args.rate,
        duration_seconds=args.duration,
        failure_scenarios=failure_scenarios
    )
    
    try:
        # Run analysis
        analyzer.run_analysis()
        
        # Generate plots
        print("\nGenerating plots...")
        analyzer.generate_plots(output_dir=args.output_dir)
        
        # Print summary
        analyzer.print_summary()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()

