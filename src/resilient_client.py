"""
Resilient Client with Automatic Retry and Recovery
- Exponential backoff with jitter
- Automatic failover between replicas
- Connection health monitoring
- No manual intervention required
- Continuous streaming of outputs
"""

import grpc
import time
import sys
import random
from typing import List, Tuple
import particle_pb2
import particle_pb2_grpc

BASE_PORT = 50051
DURATION = 120  # Extended duration for disruption testing
INITIAL_TIMEOUT = 2.0
MAX_TIMEOUT = 10.0
MAX_RETRIES = 5
BACKOFF_MULTIPLIER = 2
JITTER_RANGE = 0.5  # +/- 0.5 seconds


class ResilientClient:
    def __init__(self, num_replicas: int):
        self.num_replicas = num_replicas
        self.targets = [f"localhost:{BASE_PORT + i}" for i in range(num_replicas)]
        self.channels = []
        self.stubs = []
        self.replica_health = {}  # Track health of each replica
        self.current_timeout = INITIAL_TIMEOUT
        
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Initialize gRPC channels with retry configuration"""
        print(f"[RESILIENT CLIENT] Initializing connections to {self.num_replicas} replicas")
        
        for target in self.targets:
            # Configure channel with automatic retry
            options = [
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', 1),
                ('grpc.http2.max_pings_without_data', 0),
            ]
            
            channel = grpc.insecure_channel(target, options=options)
            stub = particle_pb2_grpc.ParticleSimStub(channel)
            
            self.channels.append(channel)
            self.stubs.append((target, stub))
            self.replica_health[target] = {'failures': 0, 'last_success': None}
            
        print(f"[RESILIENT CLIENT] Configuration:")
        print(f" -> Replicas: {self.num_replicas}")
        print(f" -> Ports: {BASE_PORT} to {BASE_PORT + self.num_replicas - 1}")
        print(f" -> Initial Timeout: {INITIAL_TIMEOUT}s")
        print(f" -> Max Retries: {MAX_RETRIES}")
        print(f" -> Backoff Multiplier: {BACKOFF_MULTIPLIER}x")
        print("-" * 80)
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        base_delay = min(INITIAL_TIMEOUT * (BACKOFF_MULTIPLIER ** attempt), MAX_TIMEOUT)
        jitter = random.uniform(-JITTER_RANGE, JITTER_RANGE)
        return max(0.1, base_delay + jitter)
    
    def _update_health(self, target: str, success: bool):
        """Update replica health tracking"""
        if success:
            self.replica_health[target]['failures'] = 0
            self.replica_health[target]['last_success'] = time.time()
        else:
            self.replica_health[target]['failures'] += 1
    
    def _get_sorted_replicas(self) -> List[Tuple[str, any]]:
        """Sort replicas by health (least failures first)"""
        return sorted(
            self.stubs,
            key=lambda x: self.replica_health[x[0]]['failures']
        )
    
    def send_request_with_retry(self, steps: int = 10) -> Tuple[bool, str]:
        """
        Send request with automatic retry and failover
        Returns: (success, message)
        """
        attempt = 0
        
        while attempt < MAX_RETRIES:
            # Try all replicas in order of health
            sorted_replicas = self._get_sorted_replicas()
            
            for target, stub in sorted_replicas:
                try:
                    # Calculate dynamic timeout based on attempt
                    timeout = self._calculate_backoff(attempt)
                    
                    # Attempt request
                    resp = stub.Step(
                        particle_pb2.StepRequest(steps=steps),
                        timeout=timeout
                    )
                    
                    # Success!
                    self._update_health(target, success=True)
                    self.current_timeout = INITIAL_TIMEOUT  # Reset timeout on success
                    
                    timestamp = time.strftime('%H:%M:%S')
                    message = (f"{timestamp} | {resp.replica_id:<25} | "
                             f"{resp.latency_ms:6.2f}ms | ✅ OK | "
                             f"attempt={attempt+1}")
                    
                    return True, message
                
                except grpc.RpcError as e:
                    self._update_health(target, success=False)
                    
                    # Determine error type
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        error_type = "TIMEOUT"
                    elif e.code() == grpc.StatusCode.UNAVAILABLE:
                        error_type = "UNAVAILABLE"
                    else:
                        error_type = str(e.code())
                    
                    # Don't print every failure, just continue to next replica
                    continue
            
            # All replicas failed this attempt
            attempt += 1
            
            if attempt < MAX_RETRIES:
                backoff_time = self._calculate_backoff(attempt)
                timestamp = time.strftime('%H:%M:%S')
                print(f"{timestamp} | ⚠️  All replicas failed (attempt {attempt}/{MAX_RETRIES}), "
                      f"backing off {backoff_time:.2f}s...")
                time.sleep(backoff_time)
        
        # Max retries exceeded
        return False, f"❌ FAILED after {MAX_RETRIES} attempts - all replicas unreachable"
    
    def run(self, duration: int = DURATION):
        """Run the client for specified duration"""
        print(f"[RESILIENT CLIENT] Starting test run for {duration} seconds")
        print(f"[RESILIENT CLIENT] Streaming outputs continuously...\n")
        
        start_time = time.time()
        request_count = 0
        success_count = 0
        failure_count = 0
        
        while time.time() - start_time < duration:
            success, message = self.send_request_with_retry()
            
            print(message)
            
            request_count += 1
            if success:
                success_count += 1
                time.sleep(0.5)  # Normal delay between successful requests
            else:
                failure_count += 1
                print(f"[RESILIENT CLIENT] Permanent failure detected, continuing...")
                time.sleep(1)  # Longer delay after permanent failure
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"[RESILIENT CLIENT] Test Complete")
        print(f" -> Duration: {duration}s")
        print(f" -> Total Requests: {request_count}")
        print(f" -> Successful: {success_count} ({100*success_count/request_count:.1f}%)")
        print(f" -> Failed: {failure_count} ({100*failure_count/request_count:.1f}%)")
        print("=" * 80)
        
        # Cleanup
        for channel in self.channels:
            channel.close()


def main():
    num_replicas = 3
    if len(sys.argv) > 1:
        num_replicas = int(sys.argv[1])
    
    duration = DURATION
    if len(sys.argv) > 2:
        duration = int(sys.argv[2])
    
    client = ResilientClient(num_replicas)
    client.run(duration)


if __name__ == "__main__":
    main()

