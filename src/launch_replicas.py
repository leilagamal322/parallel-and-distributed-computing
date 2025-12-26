# launch_replicas.py
import subprocess
import sys
import time

BASE_PORT = 50051

def launch_replicas(num_replicas):
    processes = []

    print(f"\nLaunching {num_replicas} replicas...\n")

    for i in range(num_replicas):
        port = BASE_PORT + i
        print(f"→ Replica {i + 1} on port {port}")

        p = subprocess.Popen(
            ["python", "server.py", str(port)]
        )
        processes.append(p)

        time.sleep(0.5)  # avoid GPU init race

    print("\nAll replicas running.")
    print("Press CTRL+C to stop all replicas.\n")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all replicas...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python launch_replicas.py <num_replicas>")
        sys.exit(1)

    launch_replicas(int(sys.argv[1]))
