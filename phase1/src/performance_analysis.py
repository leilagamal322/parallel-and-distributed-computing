import os
import sys
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from baseline_cpu_simulation import benchmark_cpu
from openmp_simulation import benchmark_openmp

THREADS = [1, 2, 4, 8]


def compare_performance(particle_counts, num_frames=100):
    """
    Run CPU baseline once, and OpenMP for multiple thread counts.
    """
    cpu_results = benchmark_cpu(particle_counts, num_frames)
    openmp_results = {}

    for t in THREADS:
        openmp_results[t] = benchmark_openmp(
            particle_counts,
            num_frames=num_frames,
            num_threads=t
        )

    return cpu_results, openmp_results


def plot_speedup_vs_threads(cpu, openmp, particle_count):
    threads = []
    speedups = []

    for t in THREADS:
        threads.append(t)
        speedups.append(cpu[particle_count] / openmp[t][particle_count])

    plt.figure()
    plt.plot(threads, speedups, marker="o")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title(f"Speedup vs Threads ({particle_count} particles)")
    plt.grid(True)
    plt.savefig("plots/speedup_vs_threads.png")
    plt.close()


def plot_efficiency_vs_threads(cpu, openmp, particle_count):
    threads = []
    efficiency = []

    for t in THREADS:
        speedup = cpu[particle_count] / openmp[t][particle_count]
        threads.append(t)
        efficiency.append(speedup / t)

    plt.figure()
    plt.plot(threads, efficiency, marker="o")
    plt.xlabel("Number of Threads")
    plt.ylabel("Efficiency")
    plt.title(f"Efficiency vs Threads ({particle_count} particles)")
    plt.grid(True)
    plt.savefig("plots/efficiency_vs_threads.png")
    plt.close()
