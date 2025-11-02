"""
Quick Comparison Script
=======================
Runs both CPU and GPU versions side-by-side for quick performance comparison.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from baseline_cpu_simulation import ParticleSystemCPU
from gpu_simulation_pycuda import ParticleSystemGPU
import pycuda.driver as cuda

def quick_comparison(num_particles=5000, num_frames=100):
    """
    Quick performance comparison between CPU and GPU versions.
    
    Args:
        num_particles: Number of particles to test
        num_frames: Number of frames to run
    """
    print("=" * 70)
    print(f"Quick Performance Comparison: {num_particles} particles, {num_frames} frames")
    print("=" * 70)
    
    # CPU Test
    print("\n[CPU] Running sequential implementation...")
    cpu_system = ParticleSystemCPU(num_particles)
    
    # Warm-up
    for _ in range(10):
        cpu_system.update()
    
    # Benchmark
    cpu_start = time.perf_counter()
    for _ in range(num_frames):
        cpu_system.update()
    cpu_end = time.perf_counter()
    
    cpu_time = (cpu_end - cpu_start) * 1000  # Convert to ms
    cpu_time_per_frame = cpu_time / num_frames
    
    print(f"CPU Total Time: {cpu_time:.2f} ms")
    print(f"CPU Time per Frame: {cpu_time_per_frame:.4f} ms")
    
    # GPU Test
    print("\n[GPU] Running parallel PyCUDA implementation...")
    print(f"GPU: {cuda.Device(0).name()}")
    
    gpu_system = ParticleSystemGPU(num_particles)
    
    # Warm-up
    for _ in range(10):
        gpu_system.update()
    cuda.Context.synchronize()
    
    # Benchmark
    gpu_start = time.perf_counter()
    for _ in range(num_frames):
        gpu_system.update()
        cuda.Context.synchronize()
    gpu_end = time.perf_counter()
    
    # Transfer time
    transfer_start = time.perf_counter()
    for _ in range(num_frames):
        gpu_system.get_positions()
    transfer_end = time.perf_counter()
    
    gpu_compute_time = (gpu_end - gpu_start) * 1000
    gpu_transfer_time = (transfer_end - transfer_start) * 1000
    gpu_total_time = gpu_compute_time + gpu_transfer_time
    gpu_time_per_frame = gpu_total_time / num_frames
    
    print(f"GPU Compute Time: {gpu_compute_time:.2f} ms")
    print(f"GPU Transfer Time: {gpu_transfer_time:.2f} ms")
    print(f"GPU Total Time: {gpu_total_time:.2f} ms")
    print(f"GPU Time per Frame: {gpu_time_per_frame:.4f} ms")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    speedup = cpu_time / gpu_total_time if gpu_total_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    print(f"Transfer Overhead: {gpu_transfer_time / gpu_total_time * 100:.1f}%")
    print("\nCPU is", "SLOWER" if speedup > 1 else "FASTER", "than GPU" + 
          (f" by {abs(1 - speedup) * 100:.1f}%" if speedup != 1 else ""))
    print("=" * 70)


if __name__ == "__main__":
    num_particles = 5000
    num_frames = 100
    
    if len(sys.argv) > 1:
        try:
            num_particles = int(sys.argv[1])
        except ValueError:
            print(f"Invalid particle count: {sys.argv[1]}. Using default: {num_particles}")
    
    if len(sys.argv) > 2:
        try:
            num_frames = int(sys.argv[2])
        except ValueError:
            print(f"Invalid frame count: {sys.argv[2]}. Using default: {num_frames}")
    
    quick_comparison(num_particles, num_frames)
