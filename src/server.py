# server.py
import time
import sys
import grpc
import queue
from concurrent import futures
import traceback

import particle_pb2
import particle_pb2_grpc

try:
    from gpu_simulation_pycuda import ParticleSystemGPU
    print("[SERVER] Using PyCUDA GPU simulation")
except ImportError:
    from mock_gpu_simulation import ParticleSystemGPU
    print("[SERVER] PyCUDA not available, using mock GPU simulation (CPU-based)")


class ParticleService(particle_pb2_grpc.ParticleSimServicer):
    def __init__(self, gpu_queue):
        self.gpu_queue = gpu_queue

    def Step(self, request, context):
        response_queue = queue.Queue()
        self.gpu_queue.put((request.steps, response_queue))

        try:
            return response_queue.get(timeout=5.0)
        except queue.Empty:
            context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                "GPU timeout"
            )


def initialize_gpu(num_particles=1000):
    print("[SERVER] Initializing GPU")
    sim = ParticleSystemGPU(num_particles=num_particles)
    sim.step(5)
    print("[SERVER] GPU ready")
    return sim


def serve(port):
    replica_name = f"replica@{port}"
    gpu_queue = queue.Queue()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    particle_pb2_grpc.add_ParticleSimServicer_to_server(
        ParticleService(gpu_queue), server
    )

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[SERVER {replica_name}] started")

    sim = initialize_gpu()

    try:
        while True:
            try:
                steps, response_q = gpu_queue.get(timeout=0.5)

                start = time.perf_counter()
                sim.step(steps)
                latency = (time.perf_counter() - start) * 1000

                response_q.put(
                    particle_pb2.StepResponse(
                        replica_id=replica_name,
                        latency_ms=latency,
                        timestamp=int(time.time()),
                    )
                )

            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print(f"[SERVER {replica_name}] shutting down")
        server.stop(5)


if __name__ == "__main__":
    serve(sys.argv[1])
