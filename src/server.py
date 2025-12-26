import time
import sys
import grpc
import queue
import threading
from concurrent import futures
import traceback

import particle_pb2
import particle_pb2_grpc

# Try to import GPU simulation, fall back to mock if PyCUDA not available
try:
    from gpu_simulation_pycuda import ParticleSystemGPU
    print("[SERVER] Using PyCUDA GPU simulation")
except ImportError:
    from mock_gpu_simulation import ParticleSystemGPU
    print("[SERVER] PyCUDA not available, using mock GPU simulation (CPU-based)")

REPLICA_ID = f"replica-{time.time_ns()}"

# Global queue to pass requests from gRPC threads to the Main GPU Thread
# Item format: (steps, response_queue)
gpu_request_queue = queue.Queue()

# Statistics tracking for resilience monitoring
stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'gpu_resets': 0,
    'last_request_time': None
}

class ParticleService(particle_pb2_grpc.ParticleSimServicer):
    def Step(self, request, context):
        """
        This runs on a gRPC Worker Thread.
        Instead of touching the GPU directly, we ask the Main Thread to do it.
        ENHANCED: Now with statistics tracking and better error handling
        """
        stats['total_requests'] += 1
        stats['last_request_time'] = time.time()
        
        # 1. Create a temporary queue to receive the specific result for this request
        my_response_queue = queue.Queue()

        # 2. Send the job to the Main Thread
        gpu_request_queue.put((request.steps, my_response_queue))

        # 3. Wait for the Main Thread to finish the work
        # This blocks this worker thread until the GPU is done
        try:
            result = my_response_queue.get(timeout=10.0) # Internal timeout
            stats['successful_requests'] += 1
            return result
        except queue.Empty:
            stats['failed_requests'] += 1
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "GPU processing timed out - system may be recovering")

def initialize_gpu(num_particles=1000, max_retries=3):
    """
    RESILIENCE: Initialize GPU with retry logic
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
    
    print(f"[{REPLICA_ID}] ⚠️  GPU initialization failed after {max_retries} attempts")
    return None, False


def reset_gpu(sim):
    """
    RESILIENCE: Attempt to reset GPU context on failure
    Returns: (new_sim, success)
    """
    print(f"[{REPLICA_ID}] 🔄 Attempting GPU reset...")
    stats['gpu_resets'] += 1
    
    try:
        if sim is not None:
            del sim
            time.sleep(1)
    except:
        pass
    
    return initialize_gpu()


def print_stats():
    """RESILIENCE: Print server statistics periodically"""
    while True:
        time.sleep(30)  # Print every 30 seconds
        
        if stats['total_requests'] > 0:  # Only print if we've received requests
            print("\n" + "=" * 60)
            print(f"[{REPLICA_ID}] Server Statistics:")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Successful: {stats['successful_requests']}")
            print(f"  Failed: {stats['failed_requests']}")
            print(f"  GPU Resets: {stats['gpu_resets']}")
            success_rate = 100 * stats['successful_requests'] / stats['total_requests']
            print(f"  Success Rate: {success_rate:.1f}%")
            print("=" * 60 + "\n")


def serve(port):
    # 1. Start gRPC Server (Non-blocking)
    # RESILIENCE: Added keepalive options for better connection handling
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
        ]
    )
    particle_pb2_grpc.add_ParticleSimServicer_to_server(
        ParticleService(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[{REPLICA_ID}] running on port {port}")

    # 2. Initialize GPU (On Main Thread)
    # RESILIENCE: Now with retry logic
    sim, success = initialize_gpu()
    
    if not success:
        print(f"[{REPLICA_ID}] ⚠️  Starting without GPU - will retry on first request")
    else:
        print(f"[{REPLICA_ID}] GPU Ready. Waiting for requests...")
    
    # RESILIENCE: Start statistics thread
    stats_thread = threading.Thread(target=print_stats, daemon=True)
    stats_thread.start()

    # 3. Main GPU Loop (Replaces server.wait_for_termination)
    # RESILIENCE: Enhanced with automatic GPU recovery
    consecutive_failures = 0
    
    try:
        while True:
            try:
                # Wait for a request from gRPC threads (non-blocking wait)
                steps, response_q = gpu_request_queue.get(timeout=0.5)
                
                # RESILIENCE: Check if GPU is available
                if sim is None:
                    print(f"[{REPLICA_ID}] ⚠️  GPU not available, attempting recovery...")
                    sim, success = reset_gpu(sim)
                    if not success:
                        consecutive_failures += 1
                        continue
                
                # --- CRITICAL SECTION: GPU WORK ---
                try:
                    start = time.perf_counter()
                    
                    sim.step(steps) # Safe because we are on Main Thread
                    
                    latency = (time.perf_counter() - start) * 1000
                    timestamp = int(time.time())
                    
                    consecutive_failures = 0  # Reset on success
                    
                    print(f"[{REPLICA_ID}] handled request in {latency:.2f} ms")
                    
                    # Send result back to the waiting gRPC thread
                    resp = particle_pb2.StepResponse(
                        replica_id=REPLICA_ID,
                        latency_ms=latency,
                        timestamp=timestamp
                    )
                    response_q.put(resp)
                
                except Exception as e:
                    # RESILIENCE: GPU operation failed, try to recover
                    print(f"[{REPLICA_ID}] ❌ GPU operation failed: {e}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= 3:
                        print(f"[{REPLICA_ID}] Multiple failures detected, resetting GPU...")
                        sim, success = reset_gpu(sim)
                        consecutive_failures = 0
                # ----------------------------------

            except queue.Empty:
                # No requests right now, just loop again
                continue
            
            except Exception as e:
                print(f"[{REPLICA_ID}] ⚠️  Unexpected error: {e}")
                traceback.print_exc()
                time.sleep(1)
                
    except KeyboardInterrupt:
        print(f"\n[{REPLICA_ID}] Shutting down gracefully...")
        server.stop(grace=5)
        
        # Print final stats
        if stats['total_requests'] > 0:
            print("\n" + "=" * 60)
            print(f"[{REPLICA_ID}] Final Statistics:")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Successful: {stats['successful_requests']}")
            print(f"  Failed: {stats['failed_requests']}")
            print(f"  GPU Resets: {stats['gpu_resets']}")
            print("=" * 60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python server.py <port>")
        sys.exit(1)

    serve(sys.argv[1])