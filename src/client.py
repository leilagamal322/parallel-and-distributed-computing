import grpc
import time
import sys
import random
import particle_pb2
import particle_pb2_grpc

BASE_PORT = 50051
DURATION = 160      
INITIAL_TIMEOUT = 2.0
MAX_TIMEOUT = 10.0
MAX_RETRIES = None  # None = infinite retries within the duration
BACKOFF_MULTIPLIER = 2
JITTER_RANGE = 0.5  # +/- 0.5 seconds for jitter      

def calculate_backoff(attempt):
    """RESILIENCE: Calculate exponential backoff with jitter"""
    base_delay = min(INITIAL_TIMEOUT * (BACKOFF_MULTIPLIER ** attempt), MAX_TIMEOUT)
    jitter = random.uniform(-JITTER_RANGE, JITTER_RANGE)
    return max(0.1, base_delay + jitter)


def main(num_replicas):
    # 1. Dynamically generate target list based on input argument
    targets = [f"localhost:{BASE_PORT + i}" for i in range(num_replicas)]
    stubs = []
    replica_health = {}  # RESILIENCE: Track health of each replica
    min_requests_after_recovery = 20  # Define here for configuration display
    
    print(f"[CLIENT] Configuration:")
    print(f" -> Looking for {num_replicas} replicas")
    print(f" -> Ports: {BASE_PORT} to {BASE_PORT + num_replicas - 1}")
    print(f" -> Duration: {DURATION} seconds")
    print(f" -> Min requests after recovery: {min_requests_after_recovery}")
    if MAX_RETRIES is None:
        print(f" -> Retry Mode: INFINITE (will keep retrying until duration expires)")
    else:
        print(f" -> Max Retries: {MAX_RETRIES} with exponential backoff")
    print("-" * 50)
    
    # Create lazy connections (no immediate connection check)
    # RESILIENCE: Added keepalive options
    for target in targets:
        options = [
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', 1),
        ]
        channel = grpc.insecure_channel(target, options=options)
        stub = particle_pb2_grpc.ParticleSimStub(channel)
        stubs.append((target, stub))
        replica_health[target] = {'failures': 0}  # Track failures

    start_time = time.time()
    request_count = 0
    success_count = 0
    failure_count = 0

    # 2. Main Request Loop
    # RESILIENCE: Enhanced with exponential backoff and health tracking
    # Continue until duration expires AND we've processed some requests after recovery
    requests_since_last_failure = 0
    
    while True:
        elapsed = time.time() - start_time
        
        # Exit conditions:
        # 1. Duration expired AND we've processed enough requests after last recovery
        # 2. OR we haven't had any failures yet and duration expired
        if elapsed >= DURATION:
            if requests_since_last_failure >= min_requests_after_recovery or failure_count == 0:
                break
        success = False
        attempt = 0
        
        # RESILIENCE: Retry with exponential backoff (infinite if MAX_RETRIES is None)
        while (MAX_RETRIES is None or attempt < MAX_RETRIES) and not success:
            # Sort replicas by health (least failures first)
            sorted_stubs = sorted(stubs, key=lambda x: replica_health[x[0]]['failures'])
            
            # Try all replicas in order of health
            for target, stub in sorted_stubs:
                try:
                    # RESILIENCE: Dynamic timeout based on attempt
                    timeout = calculate_backoff(attempt)
                    
                    # Attempt request
                    resp = stub.Step(
                        particle_pb2.StepRequest(steps=10), 
                        timeout=timeout
                    )
                    
                    # RESILIENCE: Update health on success
                    replica_health[target]['failures'] = 0
                    
                    retry_info = f" | retry={attempt+1}" if attempt > 0 else ""
                    
                    # Check if this is a recovery (first success after failures)
                    if attempt > 0 and requests_since_last_failure == 0:
                        print(f"{time.strftime('%H:%M:%S')} | {resp.replica_id:<20} | {resp.latency_ms:.2f} ms | ✅ RECOVERED | retry={attempt+1}")
                    else:
                        print(f"{time.strftime('%H:%M:%S')} | {resp.replica_id:<20} | {resp.latency_ms:.2f} ms | ✅ OK{retry_info}")
                    
                    success = True
                    success_count += 1
                    requests_since_last_failure += 1
                    break # Request succeeded, stop trying other replicas

                except grpc.RpcError as e:
                    # RESILIENCE: Update health on failure
                    replica_health[target]['failures'] += 1
                    # Try next replica
                    continue 
            
            # If all replicas failed this attempt
            if not success:
                attempt += 1
                requests_since_last_failure = 0  # Reset counter on failure
                backoff_time = calculate_backoff(attempt)
                if MAX_RETRIES is None:
                    print(f"[CLIENT] ⚠️  All {num_replicas} replicas failed (attempt {attempt}), backing off {backoff_time:.2f}s... (will keep retrying)")
                elif attempt < MAX_RETRIES:
                    print(f"[CLIENT] ⚠️  All {num_replicas} replicas failed (attempt {attempt}/{MAX_RETRIES}), backing off {backoff_time:.2f}s...")
                time.sleep(backoff_time)
        
        # After all retries (only happens if MAX_RETRIES is not None)
        if not success:
            if MAX_RETRIES is None:
                # This shouldn't happen with infinite retry, but just in case
                print(f"[CLIENT] ⚠️  Request cycle completed without success (attempt {attempt})")
            else:
                print(f"[CLIENT] ❌ FAILED after {MAX_RETRIES} attempts - all replicas unreachable")
            failure_count += 1
            time.sleep(1)
        else:
            # Normal delay between successful requests
            time.sleep(0.5)
            
        request_count += 1

    # Print exit reason
    elapsed = time.time() - start_time
    if requests_since_last_failure >= min_requests_after_recovery:
        print(f"[CLIENT] ℹ️  Duration complete ({elapsed:.0f}s) and processed {requests_since_last_failure} requests after recovery")

    # RESILIENCE: Enhanced summary
    print("-" * 50)
    print(f"[CLIENT] Test Complete")
    print(f" -> Total Requests: {request_count}")
    print(f" -> Successful: {success_count} ({100*success_count/request_count:.1f}%)")
    print(f" -> Failed: {failure_count} ({100*failure_count/request_count:.1f}%)")
    print("-" * 50)

if __name__ == "__main__":
    # Default to 3 if no argument is provided
    count = 3
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
        
    main(count)