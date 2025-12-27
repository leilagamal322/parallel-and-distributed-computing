"""
Quick Comparison Script
=======================
Runs CPU, OpenMP, and GPU versions side-by-side for quick performance comparison.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from baseline_cpu_simulation import ParticleSystemCPU

# Try to import GPU version
gpu_available = False
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from gpu_simulation_pycuda import ParticleSystemGPU
    gpu_available = True
except ImportError:
    print("Warning: PyCUDA not available, skipping GPU comparison")

# Try to import OpenMP version
openmp_available = False
try:
    from openmp_simulation import ParticleSystemOpenMP
    openmp_available = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: OpenMP not available ({e}), skipping OpenMP comparison")

def quick_comparison(num_particles=5000, num_frames=100):
    """
    Quick performance comparison between CPU, OpenMP, and GPU versions.
    
    Args:
        num_particles: Number of particles to test
        num_frames: Number of frames to run
    """
    print("=" * 70)
    print(f"Quick Performance Comparison: {num_particles} particles, {num_frames} frames")
    print("=" * 70)
    
    results = {}
    
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
    results['CPU'] = cpu_time_per_frame
    
    print(f"CPU Total Time: {cpu_time:.2f} ms")
    print(f"CPU Time per Frame: {cpu_time_per_frame:.4f} ms")
    
    # OpenMP Test
    if openmp_available:
        print("\n[OpenMP] Running parallel OpenMP implementation...")
        try:
            openmp_system = ParticleSystemOpenMP(num_particles)
            
            # Warm-up
            for _ in range(10):
                openmp_system.update()
            
            # Benchmark
            openmp_start = time.perf_counter()
            for _ in range(num_frames):
                openmp_system.update()
            openmp_end = time.perf_counter()
            
            openmp_time = (openmp_end - openmp_start) * 1000
            openmp_time_per_frame = openmp_time / num_frames
            results['OpenMP'] = openmp_time_per_frame
            
            print(f"OpenMP Total Time: {openmp_time:.2f} ms")
            print(f"OpenMP Time per Frame: {openmp_time_per_frame:.4f} ms")
            print(f"Using {openmp_system.num_threads} threads")
            
            openmp_system.cleanup()
        except Exception as e:
            print(f"OpenMP test failed: {e}")
    else:
        print("\n[OpenMP] Skipped - not available")
    
    # GPU Test
    if gpu_available:
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
        results['GPU'] = gpu_time_per_frame
        
        print(f"GPU Compute Time: {gpu_compute_time:.2f} ms")
        print(f"GPU Transfer Time: {gpu_transfer_time:.2f} ms")
        print(f"GPU Total Time: {gpu_total_time:.2f} ms")
        print(f"GPU Time per Frame: {gpu_time_per_frame:.4f} ms")
    else:
        print("\n[GPU] Skipped - not available")
    
    # Results Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if 'CPU' in results:
        print(f"CPU:      {results['CPU']:8.4f} ms/frame (baseline)")
    
    if 'OpenMP' in results:
        openmp_speedup = results['CPU'] / results['OpenMP'] if results['OpenMP'] > 0 else 0
        print(f"OpenMP:   {results['OpenMP']:8.4f} ms/frame ({openmp_speedup:.2f}x speedup)")
    
    if 'GPU' in results:
        gpu_speedup = results['CPU'] / results['GPU'] if results['GPU'] > 0 else 0
        print(f"GPU:      {results['GPU']:8.4f} ms/frame ({gpu_speedup:.2f}x speedup)")
    
    if 'OpenMP' in results and 'GPU' in results:
        openmp_vs_gpu = results['GPU'] / results['OpenMP'] if results['OpenMP'] > 0 else 0
        print(f"\nOpenMP vs GPU: {openmp_vs_gpu:.2f}x ({'OpenMP' if openmp_vs_gpu > 1 else 'GPU'} is faster)")
    
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

    

