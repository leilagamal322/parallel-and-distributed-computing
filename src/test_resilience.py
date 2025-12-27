import subprocess
import time
import signal
import os
import sys
import threading

NUM_REPLICAS = 3
BASE_PORT = 50051


def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90 + "\n")


def stream_output(prefix, stream):
    for line in stream:
        print(f"{prefix} {line.rstrip()}")



def launch_replicas():
    print_header("STEP 1 — Launching replicated gRPC servers")

    subprocess.Popen(
        [sys.executable, "launch_replicas.py", str(NUM_REPLICAS)]
    )

    time.sleep(6)

    pids = subprocess.check_output(
        ["pgrep", "-f", "server.py"]
    ).decode().strip().split("\n")

    pids = [int(p) for p in pids]

    for i, pid in enumerate(pids, start=1):
        print(f"Replica #{i} started (PID={pid}, port={BASE_PORT + i - 1})")

    return pids


def launch_client():
    print_header("STEP 2 — Launching client (continuous requests)")

    client = subprocess.Popen(
    [sys.executable, "-u", "client.py", str(NUM_REPLICAS)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1)


    threading.Thread(
        target=stream_output,
        args=("[CLIENT]", client.stdout),
        daemon=True
    ).start()

    return client


def service_crash(pids):
    print_header("FAILURE 1 — Service crash (kill one replica)")

    time.sleep(10)

    pid = pids[0]
    print(f"Killing Replica #1 (PID={pid})")

    os.kill(pid, signal.SIGTERM)


def force_timeout(pids):
    print_header("FAILURE 2 — Network disruption (temporary freeze)")

    time.sleep(10)

    pid = pids[1]
    print(f"Freezing Replica #2 (PID={pid})")

    os.kill(pid, signal.SIGSTOP)

    time.sleep(10)

    print(f"Resuming Replica #2 (PID={pid})")
    os.kill(pid, signal.SIGCONT)


def cleanup():
    print_header("CLEANUP")

    subprocess.call(["pkill", "-f", "server.py"])
    subprocess.call(["pkill", "-f", "client.py"])
    print("All processes stopped")


if __name__ == "__main__":
    try:
        pids = launch_replicas()
        launch_client()

        service_crash(pids)
        force_timeout(pids)

        print("\nDemo finished — client remained available during failures\n")
        time.sleep(10)

    finally:
        cleanup()
