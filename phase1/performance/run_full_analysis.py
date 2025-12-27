import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from performance_analysis import (
    compare_performance,
    plot_speedup_vs_threads,
    plot_efficiency_vs_threads
)

os.makedirs("plots", exist_ok=True)

PARTICLES = [100, 500, 1000, 5000, 10000]
FRAMES = 100
REFERENCE_PARTICLE = 5000

print("=" * 80)
print("PHASE 1 — FULL PERFORMANCE ANALYSIS (THREAD SCALING)")
print("=" * 80)

cpu, openmp = compare_performance(PARTICLES, FRAMES)

print("\nCPU baseline:")
for p, t in cpu.items():
    print(f"{p:6d} particles → {t:.4f} ms")

print("\nOpenMP results:")
for threads, results in openmp.items():
    print(f"\nThreads = {threads}")
    for p, t in results.items():
        print(f"{p:6d} particles → {t:.4f} ms")

plot_speedup_vs_threads(cpu, openmp, REFERENCE_PARTICLE)
plot_efficiency_vs_threads(cpu, openmp, REFERENCE_PARTICLE)

print("\nGenerated:")
print(" - plots/speedup_vs_threads.png")
print(" - plots/efficiency_vs_threads.png")
print("\nPhase 1 DONE.")
