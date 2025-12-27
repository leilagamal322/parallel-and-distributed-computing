"""
Phase 2 — MPI Strong Scaling Runner (FINAL)
"""

import subprocess
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

MPI_SCRIPT = os.path.join("src", "mpi_performance_experiments.py")


def run_experiment(num_processes: int) -> float:
    """
    Run MPI strong scaling experiment and extract avg time.
    """
    cmd = [
        "mpirun",
        "--oversubscribe",
        "-n", str(num_processes),
        sys.executable,
        MPI_SCRIPT,
        "--strong-scaling"     # ✅ THIS WAS MISSING
    ]

    print(f"\nRunning: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    for line in result.stdout.splitlines():
        if "Avg time:" in line:
            # Avg time: XXXX ms/frame
            return float(line.split(":")[1].split()[0])

    print("Failed to extract timing.")
    return None


def plot_strong_scaling(results):
    os.makedirs("plots", exist_ok=True)

    procs = sorted(results.keys())
    times = np.array([results[p] for p in procs])

    speedup = times[0] / times
    efficiency = speedup / np.array(procs)

    plt.figure()
    plt.plot(procs, speedup, marker="o", label="Speedup")
    plt.plot(procs, procs, "--", label="Ideal")
    plt.xlabel("MPI Processes")
    plt.ylabel("Speedup")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/strong_scaling_speedup.png")
    plt.close()

    plt.figure()
    plt.plot(procs, efficiency, marker="o")
    plt.xlabel("MPI Processes")
    plt.ylabel("Efficiency")
    plt.grid(True)
    plt.savefig("plots/strong_scaling_efficiency.png")
    plt.close()

    print("\nPlots saved in ./plots/")


def main():
    processes = [1, 2, 4, 8]

    print("=" * 70)
    print("PHASE 2 — MPI STRONG SCALING EXPERIMENT")
    print("=" * 70)

    results = {}

    for p in processes:
        t = run_experiment(p)
        if t is not None:
            results[p] = t

    if not results:
        print("\nNo results collected.")
        return

    plot_strong_scaling(results)
    print("\nPhase 2 DONE.")


if __name__ == "__main__":
    main()
