"""Simple benchmark comparison"""
import sys
sys.path.insert(0, 'src')

print("=" * 80)
print("Performance Comparison: CPU vs OpenMP")
print("=" * 80)

# Import benchmarks
from baseline_cpu_simulation import benchmark_cpu
from openmp_simulation import benchmark_openmp

# Test parameters
particle_counts = [100, 500, 1000, 5000]
num_frames = 100

print(f"\nTesting with {particle_counts} particles")
print(f"Frames per test: {num_frames}\n")

# Run CPU benchmark
print("=" * 80)
print("CPU (Sequential) Benchmark")
print("=" * 80)
cpu_results = benchmark_cpu(particle_counts, num_frames)

# Run OpenMP benchmark
print("\n" + "=" * 80)
print("OpenMP (Parallel) Benchmark")
print("=" * 80)
openmp_results = benchmark_openmp(particle_counts, num_frames, num_threads=4)

# Compare results
print("\n" + "=" * 80)
print("Performance Comparison Summary")
print("=" * 80)
print(f"{'Particles':>10} | {'CPU (ms)':>12} | {'OpenMP (ms)':>14} | {'Speedup':>10}")
print("-" * 80)

for p in particle_counts:
    cpu_time = cpu_results.get(p, 0)
    openmp_time = openmp_results.get(p, 0) if openmp_results.get(p) else None
    
    if openmp_time and openmp_time > 0:
        speedup = cpu_time / openmp_time
        print(f"{p:>10} | {cpu_time:>12.4f} | {openmp_time:>14.4f} | {speedup:>10.2f}x")
    else:
        print(f"{p:>10} | {cpu_time:>12.4f} | {'N/A':>14} | {'N/A':>10}")

print("=" * 80)
print("\nAnalysis complete!")

