import time
import sys
import grpc
import queue
import threading
from concurrent import futures

import particle_pb2
import particle_pb2_grpc
from gpu_simulation_pycuda import ParticleSystemGPU

REPLICA_ID = f"replica-{time.time_ns()}"

# Global queue to pass requests from gRPC threads to the Main GPU Thread
# Item format: (steps, response_queue)
gpu_request_queue = queue.Queue()

class ParticleService(particle_pb2_grpc.ParticleSimServicer):
    def Step(self, request, context):
        """
        This runs on a gRPC Worker Thread.
        Instead of touching the GPU directly, we ask the Main Thread to do it.
        """
        # 1. Create a temporary queue to receive the specific result for this request
        my_response_queue = queue.Queue()

        # 2. Send the job to the Main Thread
        gpu_request_queue.put((request.steps, my_response_queue))

        # 3. Wait for the Main Thread to finish the work
        # This blocks this worker thread until the GPU is done
        try:
            result = my_response_queue.get(timeout=10.0) # Internal timeout
            return result
        except queue.Empty:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "GPU processing timed out")

def serve(port):
    # 1. Start gRPC Server (Non-blocking)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    particle_pb2_grpc.add_ParticleSimServicer_to_server(
        ParticleService(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[{REPLICA_ID}] running on port {port}")

    # 2. Initialize GPU (On Main Thread)
    # PyCUDA context is now bound to THIS thread
    print(f"[{REPLICA_ID}] Initializing GPU...")
    sim = ParticleSystemGPU(num_particles=1000)
    sim.step(5) # Warmup
    print(f"[{REPLICA_ID}] GPU Ready. Waiting for requests...")

    # 3. Main GPU Loop (Replaces server.wait_for_termination)
    # This loop keeps the Main Thread alive and processing GPU work
    try:
        while True:
            try:
                # Wait for a request from gRPC threads (non-blocking wait)
                steps, response_q = gpu_request_queue.get(timeout=0.5)
                
                # --- CRITICAL SECTION: GPU WORK ---
                start = time.perf_counter()
                
                sim.step(steps) # Safe because we are on Main Thread
                
                latency = (time.perf_counter() - start) * 1000
                timestamp = int(time.time())
                
                print(f"[{REPLICA_ID}] handled request in {latency:.2f} ms")
                
                # Send result back to the waiting gRPC thread
                resp = particle_pb2.StepResponse(
                    replica_id=REPLICA_ID,
                    latency_ms=latency,
                    timestamp=timestamp
                )
                response_q.put(resp)
                # ----------------------------------

            except queue.Empty:
                # No requests right now, just loop again
                continue
                
    except KeyboardInterrupt:
       print(f"[SERVER] replica={REPLICA_ID} on port={port} CRASHED")
        server.stop(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python server.py <port>")
        sys.exit(1)

    serve(sys.argv[1])