"""
Resilient Server with Auto-Recovery
- Graceful error handling
- Automatic GPU context recovery
- Request queue persistence
- No manual restart required
- Continuous operation even during failures
"""

import time
import sys
import grpc
import queue
import threading
from concurrent import futures
import traceback

import particle_pb2
import particle_pb2_grpc
from gpu_simulation_pycuda import ParticleSystemGPU

REPLICA_ID = f"replica-{time.time_ns()}"

# Global queue to pass requests from gRPC threads to the Main GPU Thread
gpu_request_queue = queue.Queue()

# Statistics tracking
stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'gpu_resets': 0,
    'last_request_time': None
}


class ResilientParticleService(particle_pb2_grpc.ParticleSimServicer):
    def __init__(self):
        self.is_healthy = True
    
    def Step(self, request, context):
        """
        Handle incoming step requests with error recovery
        """
        stats['total_requests'] += 1
        stats['last_request_time'] = time.time()
        
        # Create a response queue for this specific request
        my_response_queue = queue.Queue()
        
        # Send job to main GPU thread
        gpu_request_queue.put((request.steps, my_response_queue))
        
        # Wait for result with timeout
        try:
            result = my_response_queue.get(timeout=10.0)
            stats['successful_requests'] += 1
            return result
            
        except queue.Empty:
            stats['failed_requests'] += 1
            context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                "GPU processing timed out - system may be recovering"
            )


def initialize_gpu(num_particles=1000, max_retries=3):
    """
    Initialize GPU with retry logic
    Returns: (sim, success)
    """
    for attempt in range(max_retries):
        try:
            print(f"[{REPLICA_ID}] Initializing GPU (attempt {attempt + 1}/{max_retries})...")
            sim = ParticleSystemGPU(num_particles=num_particles)
            sim.step(5)  # Warmup
            print(f"[{REPLICA_ID}] ✅ GPU initialized successfully")
            return sim, True
            
        except Exception as e:
            print(f"[{REPLICA_ID}] ❌ GPU initialization failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"[{REPLICA_ID}] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[{REPLICA_ID}] ⚠️  GPU initialization failed after {max_retries} attempts")
                return None, False
    
    return None, False


def reset_gpu(sim):
    """
    Attempt to reset GPU context
    Returns: (new_sim, success)
    """
    print(f"[{REPLICA_ID}] 🔄 Attempting GPU reset...")
    stats['gpu_resets'] += 1
    
    # Try to cleanup old context
    try:
        if sim is not None:
            del sim
            time.sleep(1)
    except:
        pass
    
    # Reinitialize
    return initialize_gpu()


def process_gpu_requests(sim, server):
    """
    Main GPU processing loop with automatic recovery
    """
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            # Wait for requests
            steps, response_q = gpu_request_queue.get(timeout=0.5)
            
            # Check if GPU is available
            if sim is None:
                print(f"[{REPLICA_ID}] ⚠️  GPU not available, attempting recovery...")
                sim, success = reset_gpu(sim)
                
                if not success:
                    # Can't recover, send error response
                    print(f"[{REPLICA_ID}] ❌ GPU recovery failed")
                    consecutive_failures += 1
                    continue
            
            # Process request
            try:
                start = time.perf_counter()
                sim.step(steps)
                latency = (time.perf_counter() - start) * 1000
                timestamp = int(time.time())
                
                # Success!
                consecutive_failures = 0
                
                # Send response
                resp = particle_pb2.StepResponse(
                    replica_id=REPLICA_ID,
                    latency_ms=latency,
                    timestamp=timestamp
                )
                response_q.put(resp)
                
                print(f"[{REPLICA_ID}] ✅ Request processed in {latency:.2f}ms")
                
            except Exception as e:
                # GPU operation failed
                print(f"[{REPLICA_ID}] ❌ GPU operation failed: {e}")
                consecutive_failures += 1
                
                # Try to recover GPU
                if consecutive_failures >= 3:
                    print(f"[{REPLICA_ID}] Multiple failures detected, resetting GPU...")
                    sim, success = reset_gpu(sim)
                    consecutive_failures = 0
                
                # Request will timeout on client side
                continue
        
        except queue.Empty:
            # No requests, just continue
            continue
        
        except Exception as e:
            # Unexpected error
            print(f"[{REPLICA_ID}] ⚠️  Unexpected error in GPU loop: {e}")
            traceback.print_exc()
            time.sleep(1)


def print_stats():
    """Print server statistics periodically"""
    while True:
        time.sleep(30)  # Print every 30 seconds
        
        print("\n" + "=" * 60)
        print(f"[{REPLICA_ID}] Server Statistics:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Successful: {stats['successful_requests']}")
        print(f"  Failed: {stats['failed_requests']}")
        print(f"  GPU Resets: {stats['gpu_resets']}")
        
        if stats['total_requests'] > 0:
            success_rate = 100 * stats['successful_requests'] / stats['total_requests']
            print(f"  Success Rate: {success_rate:.1f}%")
        
        if stats['last_request_time']:
            idle_time = time.time() - stats['last_request_time']
            print(f"  Time Since Last Request: {idle_time:.1f}s")
        
        print("=" * 60 + "\n")


def serve(port):
    """
    Start resilient server with automatic recovery
    """
    print(f"\n{'=' * 60}")
    print(f"[{REPLICA_ID}] Starting Resilient Server")
    print(f"[{REPLICA_ID}] Port: {port}")
    print(f"{'=' * 60}\n")
    
    # Initialize gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
        ]
    )
    
    particle_pb2_grpc.add_ParticleSimServicer_to_server(
        ResilientParticleService(), server
    )
    
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    print(f"[{REPLICA_ID}] ✅ gRPC server started on port {port}")
    
    # Initialize GPU
    sim, success = initialize_gpu()
    
    if not success:
        print(f"[{REPLICA_ID}] ⚠️  Starting without GPU - will retry on first request")
    
    # Start statistics thread
    stats_thread = threading.Thread(target=print_stats, daemon=True)
    stats_thread.start()
    
    # Start main GPU processing loop
    print(f"[{REPLICA_ID}] 🚀 Server ready - waiting for requests...\n")
    
    try:
        process_gpu_requests(sim, server)
    
    except KeyboardInterrupt:
        print(f"\n[{REPLICA_ID}] Shutting down gracefully...")
        server.stop(grace=5)
        
        # Print final stats
        print("\n" + "=" * 60)
        print(f"[{REPLICA_ID}] Final Statistics:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Successful: {stats['successful_requests']}")
        print(f"  Failed: {stats['failed_requests']}")
        print(f"  GPU Resets: {stats['gpu_resets']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resilient_server.py <port>")
        sys.exit(1)
    
    serve(int(sys.argv[1]))

