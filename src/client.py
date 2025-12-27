import grpc
import time
import sys

import particle_pb2
import particle_pb2_grpc

BASE_PORT = 50051
DURATION = 160
TIMEOUT = 1.5


def main(num_replicas):
    replicas = [f"localhost:{BASE_PORT + i}" for i in range(num_replicas)]
    channels = {r: grpc.insecure_channel(r) for r in replicas}
    stubs = {r: particle_pb2_grpc.ParticleSimStub(channels[r]) for r in replicas}

    unhealthy = set()
    start_time = time.time()

    print("[CLIENT] started", flush=True)
    print(f"[CLIENT] replicas = {replicas}", flush=True)
    print("-" * 80, flush=True)

    while time.time() - start_time < DURATION:
        timestamp = time.strftime("%H:%M:%S")
        served = False

        for replica in replicas:
            if replica in unhealthy:
                continue

            try:
                resp = stubs[replica].Step(
                    particle_pb2.StepRequest(steps=10),
                    timeout=TIMEOUT
                )

                status = "RECOVERED" if unhealthy else "SUCCESS"
                unhealthy.clear()

                print(
                    f"[{timestamp}] "
                    f"replica={replica} "
                    f"latency={resp.latency_ms:.2f}ms "
                    f"{status}",
                    flush=True
                )

                served = True
                break

            except grpc.RpcError:
                print(
                    f"[{timestamp}] replica={replica} FAILED",
                    flush=True
                )
                unhealthy.add(replica)

        time.sleep(0.5)

    print("\n[CLIENT] finished — streaming uninterrupted", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(n)
