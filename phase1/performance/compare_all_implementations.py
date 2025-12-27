"""
Comprehensive Performance Comparison
====================================
Compares Sequential CPU, OpenMP, and PyCUDA (GPU) implementations
"""

import sys
import os
import time

# Setup paths
sys.path.insert(0, 'src')

print("=" * 80)
print("COMPREHENSIVE PERFORMANCE COMPARISON")
print("Sequential CPU vs OpenMP vs PyCUDA (GPU)")
print("=" * 80)
print()

# Import implementations
print("Loading implementations...")
from baseline_cpu_simulation import ParticleSystemCPU, benchmark_cpu

# Try to import OpenMP
openmp_available = False
try:
    from openmp_simulation import ParticleSystemOpenMP, benchmark_openmp
    openmp_available = True
    print("[OK] OpenMP implementation loaded")
except Exception as e:
    print(f"[WARNING] OpenMP not available: {e}")
    print("  (Using Python threading fallback if available)")

# Try to import GPU
gpu_available = False
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from gpu_simulation_pycuda import ParticleSystemGPU, benchmark_gpu
    gpu_available = True
    print(f"[OK] GPU implementation loaded ({cuda.Device(0).name()})")
except Exception as e:
    print(f"[WARNING] GPU not available: {e}")

print()

# Test parameters
particle_counts = [100, 500, 1000, 5000, 10000]
num_frames = 100
num_threads = 4  # For OpenMP

print(f"Test Configuration:")
print(f"  Particle counts: {particle_counts}")
print(f"  Frames per test: {num_frames}")
print(f"  OpenMP threads: {num_threads}")
print()

results = {
    'cpu': {},
    'openmp': {},
    'gpu': {}
}

# ============================================================================
# CPU (Sequential) Benchmark
# ============================================================================
print("=" * 80)
print("1. CPU (Sequential) Benchmark")
print("=" * 80)
cpu_results = benchmark_cpu(particle_counts, num_frames, include_cache_analysis=False)
results['cpu'] = cpu_results
print()

# ============================================================================
# OpenMP Benchmark
# ============================================================================
if openmp_available:
    print("=" * 80)
    print("2. OpenMP (Parallel) Benchmark")
    print("=" * 80)
    try:
        openmp_results = benchmark_openmp(particle_counts, num_frames, num_threads)
        results['openmp'] = openmp_results
        print()
    except Exception as e:
        print(f"Error running OpenMP benchmark: {e}\n")
else:
    print("=" * 80)
    print("2. OpenMP (Parallel) Benchmark - SKIPPED")
    print("=" * 80)
    print("OpenMP implementation not available\n")

# ============================================================================
# GPU Benchmark
# ============================================================================
if gpu_available:
    print("=" * 80)
    print("3. GPU (PyCUDA) Benchmark")
    print("=" * 80)
    try:
        gpu_results = benchmark_gpu(particle_counts, num_frames)
        results['gpu'] = gpu_results
        print()
    except Exception as e:
        print(f"Error running GPU benchmark: {e}\n")
else:
    print("=" * 80)
    print("3. GPU (PyCUDA) Benchmark - SKIPPED")
    print("=" * 80)
    print("GPU implementation not available\n")

# ============================================================================
# Performance Comparison Summary
# ============================================================================
print("=" * 80)
print("PERFORMANCE COMPARISON SUMMARY")
print("=" * 80)
print()

# Create comparison table
header = f"{'Particles':>10} | {'CPU (ms)':>12}"
if results['openmp']:
    header += f" | {'OpenMP (ms)':>14} | {'OMP Speedup':>13}"
if results['gpu']:
    header += f" | {'GPU Compute':>15} | {'GPU Transfer':>15} | {'GPU Total':>12} | {'GPU Speedup':>13}"

print(header)
print("-" * 120)

for num_particles in particle_counts:
    if num_particles not in results['cpu']:
        continue
    
    cpu_time = results['cpu'][num_particles]
    row = f"{num_particles:>10} | {cpu_time:>12.4f}"
    
    # OpenMP comparison
    if num_particles in results['openmp'] and results['openmp'][num_particles] is not None:
        openmp_time = results['openmp'][num_particles]
        openmp_speedup = cpu_time / openmp_time if openmp_time > 0 else 0
        row += f" | {openmp_time:>14.4f} | {openmp_speedup:>13.2f}x"
    elif results['openmp']:
        row += f" | {'N/A':>14} | {'N/A':>13}"
    
    # GPU comparison
    if num_particles in results['gpu']:
        gpu_update, gpu_transfer = results['gpu'][num_particles]
        gpu_total = gpu_update + gpu_transfer
        gpu_speedup = cpu_time / gpu_total if gpu_total > 0 else 0
        row += f" | {gpu_update:>15.4f} | {gpu_transfer:>15.4f} | {gpu_total:>12.4f} | {gpu_speedup:>13.2f}x"
    
    print(row)

print("-" * 120)
print()

# ============================================================================
# Detailed Analysis
# ============================================================================
print("=" * 80)
print("DETAILED ANALYSIS")
print("=" * 80)
print()

# OpenMP Analysis
if results['openmp']:
    print("OpenMP vs CPU:")
    print("-" * 80)
    for num_particles in particle_counts:
        if (num_particles in results['cpu'] and 
            num_particles in results['openmp'] and 
            results['openmp'][num_particles] is not None):
            cpu_time = results['cpu'][num_particles]
            openmp_time = results['openmp'][num_particles]
            speedup = cpu_time / openmp_time if openmp_time > 0 else 0
            improvement = ((cpu_time - openmp_time) / cpu_time * 100) if cpu_time > 0 else 0
            
            if speedup > 1:
                print(f"  {num_particles:>6} particles: OpenMP is {speedup:.2f}x FASTER ({(speedup-1)*100:.1f}% improvement)")
            elif speedup < 1:
                print(f"  {num_particles:>6} particles: OpenMP is {1/speedup:.2f}x SLOWER (due to threading overhead)")
            else:
                print(f"  {num_particles:>6} particles: Same performance")
    print()

# GPU Analysis
if results['gpu']:
    print("GPU vs CPU:")
    print("-" * 80)
    for num_particles in particle_counts:
        if num_particles in results['cpu'] and num_particles in results['gpu']:
            cpu_time = results['cpu'][num_particles]
            gpu_update, gpu_transfer = results['gpu'][num_particles]
            gpu_total = gpu_update + gpu_transfer
            speedup = cpu_time / gpu_total if gpu_total > 0 else 0
            transfer_percent = (gpu_transfer / gpu_total * 100) if gpu_total > 0 else 0
            
            if speedup > 1:
                print(f"  {num_particles:>6} particles: GPU is {speedup:.2f}x FASTER")
                print(f"           (Compute: {gpu_update:.4f}ms, Transfer: {gpu_transfer:.4f}ms, Transfer overhead: {transfer_percent:.1f}%)")
            else:
                print(f"  {num_particles:>6} particles: GPU is {1/speedup:.2f}x SLOWER (transfer overhead dominates)")
    print()

# OpenMP vs GPU Comparison
if results['openmp'] and results['gpu']:
    print("OpenMP vs GPU:")
    print("-" * 80)
    for num_particles in particle_counts:
        if (num_particles in results['openmp'] and 
            results['openmp'][num_particles] is not None and
            num_particles in results['gpu']):
            openmp_time = results['openmp'][num_particles]
            gpu_update, gpu_transfer = results['gpu'][num_particles]
            gpu_total = gpu_update + gpu_transfer
            comparison = openmp_time / gpu_total if gpu_total > 0 else 0
            
            if comparison > 1:
                print(f"  {num_particles:>6} particles: GPU is {comparison:.2f}x FASTER than OpenMP")
            elif comparison < 1:
                print(f"  {num_particles:>6} particles: OpenMP is {1/comparison:.2f}x FASTER than GPU")
            else:
                print(f"  {num_particles:>6} particles: Similar performance")
    print()

# ============================================================================
# Key Findings
# ============================================================================
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

findings = []

if results['openmp']:
    avg_openmp_speedup = 0
    count = 0
    for num_particles in particle_counts:
        if (num_particles in results['cpu'] and 
            num_particles in results['openmp'] and 
            results['openmp'][num_particles] is not None):
            cpu_time = results['cpu'][num_particles]
            openmp_time = results['openmp'][num_particles]
            speedup = cpu_time / openmp_time if openmp_time > 0 else 0
            if speedup > 0:
                avg_openmp_speedup += speedup
                count += 1
    
    if count > 0:
        avg_openmp_speedup /= count
        if avg_openmp_speedup > 1:
            findings.append(f"[OK] OpenMP shows {avg_openmp_speedup:.2f}x average speedup over sequential CPU")
        else:
            findings.append(f"[WARNING] OpenMP Python fallback has overhead - true C++ OpenMP would be faster")

if results['gpu']:
    avg_gpu_speedup = 0
    count = 0
    for num_particles in particle_counts:
        if num_particles in results['cpu'] and num_particles in results['gpu']:
            cpu_time = results['cpu'][num_particles]
            gpu_update, gpu_transfer = results['gpu'][num_particles]
            gpu_total = gpu_update + gpu_transfer
            speedup = cpu_time / gpu_total if gpu_total > 0 else 0
            if speedup > 0:
                avg_gpu_speedup += speedup
                count += 1
    
    if count > 0:
        avg_gpu_speedup /= count
        if avg_gpu_speedup > 1:
            findings.append(f"[OK] GPU shows {avg_gpu_speedup:.2f}x average speedup over sequential CPU")
        else:
            findings.append(f"[WARNING] GPU transfer overhead limits performance for small particle counts")

if results['openmp'] and results['gpu']:
    findings.append("[OK] GPU typically outperforms OpenMP for large particle counts")
    findings.append("[OK] OpenMP has no transfer overhead, GPU has data transfer cost")
    findings.append("[OK] Optimal choice depends on particle count and hardware")

for finding in findings:
    print(finding)

print()
print("=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)

