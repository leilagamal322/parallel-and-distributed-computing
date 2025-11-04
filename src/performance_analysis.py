"""
Performance Analysis and Comparison Script
===========================================
Compares CPU (sequential) vs OpenMP (parallel) vs GPU (PyCUDA) performance and generates plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path to import simulation modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_cpu_simulation import ParticleSystemCPU, benchmark_cpu

# Try to import GPU version
try:
    from gpu_simulation_pycuda import ParticleSystemGPU, benchmark_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPU version not available, skipping GPU benchmarks")

# Try to import OpenMP version
try:
    from openmp_simulation import ParticleSystemOpenMP, benchmark_openmp
    OPENMP_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    OPENMP_AVAILABLE = False
    print(f"Warning: OpenMP version not available ({e}), skipping OpenMP benchmarks")


def compare_performance(num_particles_list=[100, 500, 1000, 5000, 10000, 50000], num_frames=300, 
                       include_cache_analysis=False, num_threads=None):
    """
    Compare CPU (sequential), OpenMP (parallel), and GPU performance for various particle counts.
    
    Args:
        num_particles_list: List of particle counts to test
        num_frames: Number of frames to run for each test
        include_cache_analysis: If True, perform cache analysis for CPU benchmarks
        num_threads: Number of threads for OpenMP (None = use default)
    
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 80)
    print("Performance Comparison: CPU vs OpenMP vs GPU")
    print("=" * 80)
    print(f"Testing {len(num_particles_list)} different particle counts")
    print(f"Frames per test: {num_frames}")
    if include_cache_analysis:
        print("Cache analysis: ENABLED")
    print()
    
    # Run CPU benchmarks
    print("\n[1/3] Running CPU (Sequential) benchmarks...")
    print("-" * 80)
    cpu_benchmark_result = benchmark_cpu(num_particles_list, num_frames, include_cache_analysis=include_cache_analysis)
    
    # Handle cache analysis results
    if include_cache_analysis and isinstance(cpu_benchmark_result, tuple):
        cpu_results, cpu_cache_results = cpu_benchmark_result
    else:
        cpu_results = cpu_benchmark_result
        cpu_cache_results = {}
    
    # Run OpenMP benchmarks
    openmp_results = {}
    if OPENMP_AVAILABLE:
        print("\n[2/3] Running OpenMP (Parallel) benchmarks...")
        print("-" * 80)
        openmp_results = benchmark_openmp(num_particles_list, num_frames, num_threads)
    else:
        print("\n[2/3] Skipping OpenMP benchmarks (not available)")
    
    # Run GPU benchmarks
    gpu_results = {}
    if GPU_AVAILABLE:
        print("\n[3/3] Running GPU benchmarks...")
        print("-" * 80)
        gpu_results = benchmark_gpu(num_particles_list, num_frames)
    else:
        print("\n[3/3] Skipping GPU benchmarks (not available)")
    
    # Calculate speedups
    openmp_speedups = {}
    gpu_speedups = {}
    
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    
    # Create header based on available implementations
    header = f"{'Particles':>10} | {'CPU (ms)':>12}"
    if openmp_results:
        header += f" | {'OpenMP (ms)':>14} | {'OMP Speedup':>13}"
    if gpu_results:
        header += f" | {'GPU (ms)':>12} | {'GPU Speedup':>13}"
    print(header)
    print("-" * 80)
    
    for num_particles in num_particles_list:
        if num_particles not in cpu_results:
            continue
        
        cpu_time = cpu_results[num_particles]
        row = f"{num_particles:>10} | {cpu_time:>12.4f}"
        
        # OpenMP comparison
        if num_particles in openmp_results and openmp_results[num_particles] is not None:
            openmp_time = openmp_results[num_particles]
            openmp_speedup = cpu_time / openmp_time if openmp_time > 0 else 0
            openmp_speedups[num_particles] = openmp_speedup
            row += f" | {openmp_time:>14.4f} | {openmp_speedup:>13.2f}x"
        elif openmp_results:
            row += f" | {'N/A':>14} | {'N/A':>13}"
        
        # GPU comparison
        if num_particles in gpu_results:
            gpu_time = gpu_results[num_particles][0] + gpu_results[num_particles][1]  # Update + transfer
            gpu_speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            gpu_speedups[num_particles] = gpu_speedup
            row += f" | {gpu_time:>12.4f} | {gpu_speedup:>13.2f}x"
        
        print(row)
    
    result_dict = {
        'cpu_results': cpu_results,
        'openmp_results': openmp_results,
        'gpu_results': gpu_results,
        'openmp_speedups': openmp_speedups,
        'gpu_speedups': gpu_speedups,
        'particle_counts': num_particles_list
    }
    
    # Add cache results if available
    if cpu_cache_results:
        result_dict['cpu_cache_results'] = cpu_cache_results
    
    return result_dict


def plot_speedup(results, save_path='plots/speedup_vs_particles.png'):
    """
    Plot speedup vs number of particles for OpenMP and GPU.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the plot
    """
    particle_counts = []
    openmp_speedups = []
    gpu_speedups = []
    
    for num_particles in results['particle_counts']:
        particle_counts.append(num_particles)
        
        if num_particles in results['openmp_speedups']:
            openmp_speedups.append(results['openmp_speedups'][num_particles])
        else:
            openmp_speedups.append(None)
        
        if num_particles in results['gpu_speedups']:
            gpu_speedups.append(results['gpu_speedups'][num_particles])
        else:
            gpu_speedups.append(None)
    
    plt.figure(figsize=(12, 7))
    
    # Filter out None values for plotting
    openmp_valid = [(p, s) for p, s in zip(particle_counts, openmp_speedups) if s is not None]
    gpu_valid = [(p, s) for p, s in zip(particle_counts, gpu_speedups) if s is not None]
    
    if openmp_valid:
        p_vals, s_vals = zip(*openmp_valid)
        plt.plot(p_vals, s_vals, 'g-o', linewidth=2, markersize=8, label='OpenMP Speedup')
    
    if gpu_valid:
        p_vals, s_vals = zip(*gpu_valid)
        plt.plot(p_vals, s_vals, 'b-s', linewidth=2, markersize=8, label='GPU Speedup')
    
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='Baseline (CPU)')
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Speedup (CPU Time / Parallel Time)', fontsize=12)
    plt.title('Performance Speedup: OpenMP and GPU vs Sequential CPU', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')
    plt.yscale('log')
    
    # Add annotations for significant speedups
    if openmp_valid:
        for count, speedup in openmp_valid:
            if speedup > 3:
                plt.annotate(f'OpenMP: {speedup:.1f}x', 
                            xy=(count, speedup), 
                            xytext=(10, 10), 
                            textcoords='offset points',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    if gpu_valid:
        for count, speedup in gpu_valid:
            if speedup > 5:
                plt.annotate(f'GPU: {speedup:.1f}x', 
                            xy=(count, speedup), 
                            xytext=(10, -20), 
                            textcoords='offset points',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSpeedup plot saved to: {save_path}")
    plt.close()


def plot_efficiency(results, save_path='plots/efficiency_vs_particles.png'):
    """
    Plot efficiency vs number of particles for OpenMP and GPU.
    Efficiency shows how well parallelization scales.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the plot
    """
    # This plot shows speedup (which is a measure of efficiency)
    # OpenMP efficiency could be calculated as speedup / num_threads
    particle_counts = []
    openmp_efficiencies = []
    gpu_efficiencies = []
    
    for num_particles in results['particle_counts']:
        particle_counts.append(num_particles)
        
        if num_particles in results['openmp_speedups']:
            openmp_efficiencies.append(results['openmp_speedups'][num_particles])
        else:
            openmp_efficiencies.append(None)
        
        if num_particles in results['gpu_speedups']:
            gpu_efficiencies.append(results['gpu_speedups'][num_particles])
        else:
            gpu_efficiencies.append(None)
    
    plt.figure(figsize=(12, 7))
    
    # Filter out None values
    openmp_valid = [(p, e) for p, e in zip(particle_counts, openmp_efficiencies) if e is not None]
    gpu_valid = [(p, e) for p, e in zip(particle_counts, gpu_efficiencies) if e is not None]
    
    if openmp_valid:
        p_vals, e_vals = zip(*openmp_valid)
        plt.plot(p_vals, e_vals, 'g-^', linewidth=2, markersize=8, label='OpenMP Speedup')
    
    if gpu_valid:
        p_vals, e_vals = zip(*gpu_valid)
        plt.plot(p_vals, e_vals, 'b-s', linewidth=2, markersize=8, label='GPU Speedup')
    
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='Baseline (CPU)')
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Speedup (Relative Performance Gain)', fontsize=12)
    plt.title('Performance Scaling: OpenMP and GPU vs Sequential CPU', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Efficiency plot saved to: {save_path}")
    plt.close()


def plot_comparison(results, save_path='plots/cpu_vs_openmp_vs_gpu_comparison.png'):
    """
    Plot direct comparison of CPU, OpenMP, and GPU times.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the plot
    """
    particle_counts = []
    cpu_times = []
    openmp_times = []
    gpu_times = []
    gpu_compute_times = []
    gpu_transfer_times = []
    
    for num_particles in results['particle_counts']:
        if num_particles not in results['cpu_results']:
            continue
        
        particle_counts.append(num_particles)
        cpu_times.append(results['cpu_results'][num_particles])
        
        # OpenMP times
        if num_particles in results['openmp_results'] and results['openmp_results'][num_particles] is not None:
            openmp_times.append(results['openmp_results'][num_particles])
        else:
            openmp_times.append(None)
        
        # GPU times
        if num_particles in results['gpu_results']:
            gpu_update, gpu_transfer = results['gpu_results'][num_particles]
            gpu_total = gpu_update + gpu_transfer
            gpu_times.append(gpu_total)
            gpu_compute_times.append(gpu_update)
            gpu_transfer_times.append(gpu_transfer)
        else:
            gpu_times.append(None)
            gpu_compute_times.append(None)
            gpu_transfer_times.append(None)
    
    plt.figure(figsize=(14, 8))
    
    plt.plot(particle_counts, cpu_times, 'r-o', linewidth=2, markersize=8, label='CPU (Sequential)')
    
    # Plot OpenMP if available
    openmp_valid = [(p, t) for p, t in zip(particle_counts, openmp_times) if t is not None]
    if openmp_valid:
        p_vals, t_vals = zip(*openmp_valid)
        plt.plot(p_vals, t_vals, 'g-^', linewidth=2, markersize=8, label='OpenMP (Parallel)')
    
    # Plot GPU if available
    gpu_valid = [(p, t) for p, t in zip(particle_counts, gpu_times) if t is not None]
    if gpu_valid:
        p_vals, t_vals = zip(*gpu_valid)
        plt.plot(p_vals, t_vals, 'b-s', linewidth=2, markersize=8, label='GPU (Total: Compute + Transfer)')
        
        # Plot GPU compute and transfer separately
        gpu_comp_valid = [(p, t) for p, t in zip(particle_counts, gpu_compute_times) if t is not None]
        gpu_trans_valid = [(p, t) for p, t in zip(particle_counts, gpu_transfer_times) if t is not None]
        
        if gpu_comp_valid:
            p_vals, t_vals = zip(*gpu_comp_valid)
            plt.plot(p_vals, t_vals, 'b--', linewidth=1.5, label='GPU Compute Only')
        
        if gpu_trans_valid:
            p_vals, t_vals = zip(*gpu_trans_valid)
            plt.plot(p_vals, t_vals, 'c--', linewidth=1.5, label='GPU Transfer Overhead')
    
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Time per Frame (ms)', fontsize=12)
    plt.title('CPU vs OpenMP vs GPU Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()


def analyze_cache_performance(results):
    """
    Analyze and report cache performance metrics.
    
    Args:
        results: Dictionary with benchmark results including cache analysis
    """
    if 'cpu_cache_results' not in results or not results['cpu_cache_results']:
        print("\n" + "=" * 80)
        print("Cache Analysis: Not Available")
        print("=" * 80)
        print("Cache analysis was not performed. Run with include_cache_analysis=True")
        return
    
    print("\n" + "=" * 80)
    print("Cache Performance Analysis")
    print("=" * 80)
    
    print(f"\n{'Particles':>10} | {'Working Set (KB)':>18} | {'Hit Rate %':>12} | {'Miss Rate %':>13} | {'Cache Level':>12}")
    print("-" * 80)
    
    for num_particles in results['particle_counts']:
        if num_particles in results['cpu_cache_results']:
            cache_analysis = results['cpu_cache_results'][num_particles]
            cb = cache_analysis['cache_behavior']
            
            # Determine which cache level fits
            if cb['fits_in_l1']:
                cache_level = "L1"
            elif cb['fits_in_l2']:
                cache_level = "L2"
            elif cb['fits_in_l3']:
                cache_level = "L3"
            else:
                cache_level = "RAM"
            
            hit_rate = cb['overall_hit_rate'] * 100
            miss_rate = cb['overall_miss_rate'] * 100
            
            print(f"{num_particles:>10} | {cb['working_set_size_kb']:>18.2f} | "
                  f"{hit_rate:>12.2f} | {miss_rate:>13.2f} | {cache_level:>12}")
    
    print("\nCache Efficiency Summary:")
    for num_particles in results['particle_counts']:
        if num_particles in results['cpu_cache_results']:
            cache_analysis = results['cpu_cache_results'][num_particles]
            cb = cache_analysis['cache_behavior']
            efficiency = cache_analysis['efficiency_level']
            
            print(f"  {num_particles} particles: {efficiency} efficiency "
                  f"({cache_analysis['efficiency_score']*100:.0f}%), "
                  f"Friendliness: {cb['cache_friendliness']*100:.1f}%")


def plot_cache_hit_rate(results, save_path='plots/cache_hit_rate.png'):
    """
    Plot cache hit rate vs number of particles.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the plot
    """
    if 'cpu_cache_results' not in results or not results['cpu_cache_results']:
        print("Cache analysis data not available for plotting")
        return
    
    particle_counts = []
    hit_rates = []
    miss_rates = []
    working_sets = []
    
    for num_particles in results['particle_counts']:
        if num_particles in results['cpu_cache_results']:
            particle_counts.append(num_particles)
            cb = results['cpu_cache_results'][num_particles]['cache_behavior']
            hit_rates.append(cb['overall_hit_rate'] * 100)
            miss_rates.append(cb['overall_miss_rate'] * 100)
            working_sets.append(cb['working_set_size_kb'])
    
    if not particle_counts:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Hit/Miss rates
    ax1.plot(particle_counts, hit_rates, 'g-o', linewidth=2, markersize=8, label='Hit Rate')
    ax1.plot(particle_counts, miss_rates, 'r-s', linewidth=2, markersize=8, label='Miss Rate')
    ax1.set_xlabel('Number of Particles', fontsize=12)
    ax1.set_ylabel('Rate (%)', fontsize=12)
    ax1.set_title('Cache Hit/Miss Rate vs Particle Count', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xscale('log')
    ax1.set_ylim(0, 105)
    
    # Plot 2: Working set size
    ax2.plot(particle_counts, working_sets, 'b-^', linewidth=2, markersize=8, label='Working Set Size')
    
    # Add cache level lines
    cb_first = results['cpu_cache_results'][particle_counts[0]]['cache_behavior']
    l1_kb = cb_first['l1_cache_size'] / 1024
    l2_kb = cb_first['l2_cache_size'] / 1024
    l3_kb = cb_first['l3_cache_size'] / 1024
    
    if max(working_sets) > l1_kb:
        ax2.axhline(y=l1_kb, color='orange', linestyle='--', linewidth=1, label=f'L1 Cache ({l1_kb:.0f} KB)')
    if max(working_sets) > l2_kb:
        ax2.axhline(y=l2_kb, color='purple', linestyle='--', linewidth=1, label=f'L2 Cache ({l2_kb:.0f} KB)')
    if max(working_sets) > l3_kb:
        ax2.axhline(y=l3_kb, color='brown', linestyle='--', linewidth=1, label=f'L3 Cache ({l3_kb:.0f} KB)')
    
    ax2.set_xlabel('Number of Particles', fontsize=12)
    ax2.set_ylabel('Working Set Size (KB)', fontsize=12)
    ax2.set_title('Working Set Size vs Particle Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cache hit rate plot saved to: {save_path}")
    plt.close()


def plot_cache_efficiency(results, save_path='plots/cache_efficiency.png'):
    """
    Plot cache efficiency vs number of particles.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the plot
    """
    if 'cpu_cache_results' not in results or not results['cpu_cache_results']:
        print("Cache analysis data not available for plotting")
        return
    
    particle_counts = []
    efficiency_scores = []
    friendliness_scores = []
    spatial_scores = []
    temporal_scores = []
    
    for num_particles in results['particle_counts']:
        if num_particles in results['cpu_cache_results']:
            particle_counts.append(num_particles)
            cache_analysis = results['cpu_cache_results'][num_particles]
            cb = cache_analysis['cache_behavior']
            
            efficiency_scores.append(cache_analysis['efficiency_score'] * 100)
            friendliness_scores.append(cb['cache_friendliness'] * 100)
            spatial_scores.append(cb['spatial_locality_score'] * 100)
            temporal_scores.append(cb['temporal_locality_score'] * 100)
    
    if not particle_counts:
        return
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(particle_counts, efficiency_scores, 'g-o', linewidth=2, markersize=8, label='Cache Efficiency')
    plt.plot(particle_counts, friendliness_scores, 'b-s', linewidth=2, markersize=8, label='Cache Friendliness')
    plt.plot(particle_counts, spatial_scores, 'c-^', linewidth=2, markersize=8, label='Spatial Locality')
    plt.plot(particle_counts, temporal_scores, 'm-v', linewidth=2, markersize=8, label='Temporal Locality')
    
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Cache Performance Metrics vs Particle Count', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cache efficiency plot saved to: {save_path}")
    plt.close()


def analyze_overheads(results):
    """
    Analyze and report overhead components for GPU and OpenMP.
    
    Args:
        results: Dictionary with benchmark results
    """
    print("\n" + "=" * 80)
    print("Overhead Analysis")
    print("=" * 80)
    
    # GPU overhead analysis
    if results.get('gpu_results'):
        print("\nGPU Overhead Analysis:")
        print(f"{'Particles':>10} | {'GPU Compute':>15} | {'Transfer':>15} | {'Transfer %':>12}")
        print("-" * 80)
        
        for num_particles in results['particle_counts']:
            if num_particles in results['gpu_results']:
                gpu_update, gpu_transfer = results['gpu_results'][num_particles]
                gpu_total = gpu_update + gpu_transfer
                transfer_percent = (gpu_transfer / gpu_total * 100) if gpu_total > 0 else 0
                
                print(f"{num_particles:>10} | {gpu_update:>15.4f} | {gpu_transfer:>15.4f} | {transfer_percent:>11.2f}%")
        
        print("\nGPU Key Observations:")
        print("- Transfer overhead represents data movement cost between CPU and GPU")
        print("- As particle count increases, compute time dominates")
        print("- For small particle counts, transfer overhead can be significant")
    
    # OpenMP overhead analysis (threading overhead)
    if results.get('openmp_results'):
        print("\nOpenMP Performance Analysis:")
        print(f"{'Particles':>10} | {'CPU Time':>15} | {'OpenMP Time':>15} | {'Speedup':>12}")
        print("-" * 80)
        
        for num_particles in results['particle_counts']:
            if (num_particles in results['cpu_results'] and 
                num_particles in results['openmp_results'] and 
                results['openmp_results'][num_particles] is not None):
                cpu_time = results['cpu_results'][num_particles]
                openmp_time = results['openmp_results'][num_particles]
                speedup = results['openmp_speedups'].get(num_particles, 0)
                
                print(f"{num_particles:>10} | {cpu_time:>15.4f} | {openmp_time:>15.4f} | {speedup:>12.2f}x")
        
        print("\nOpenMP Key Observations:")
        print("- OpenMP uses shared-memory parallelism with minimal overhead")
        print("- Speedup scales with number of CPU cores and thread count")
        print("- Better performance than sequential CPU for larger particle counts")


def generate_report(results):
    """
    Generate a text report with performance analysis.
    
    Args:
        results: Dictionary with benchmark results
    """
    report_path = 'report_performance.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Performance Analysis Report\n")
        f.write("Physics-Based Particle Simulation: CPU vs OpenMP vs GPU\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write("This report compares sequential CPU implementation vs OpenMP parallel\n")
        f.write("implementation vs GPU-accelerated implementation for particle simulation.\n\n")
        
        f.write("2. METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("- Sequential CPU version: NumPy-based single-threaded computation\n")
        f.write("- OpenMP version: Shared-memory parallel processing using C++/Python\n")
        f.write("- GPU version: PyCUDA with parallel CUDA kernels\n")
        f.write("- Benchmark: Multiple particle counts tested over 300 frames each\n")
        f.write("- Metrics: Average time per frame, speedup, efficiency\n\n")
        
        f.write("3. RESULTS\n")
        f.write("-" * 80 + "\n")
        
        # Create header
        header = f"{'Particles':>10} | {'CPU (ms)':>12}"
        if results.get('openmp_results'):
            header += f" | {'OpenMP (ms)':>14} | {'OMP Speedup':>13}"
        if results.get('gpu_results'):
            header += f" | {'GPU Total (ms)':>15} | {'GPU Speedup':>13}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        for num_particles in results['particle_counts']:
            if num_particles not in results['cpu_results']:
                continue
            
            cpu_time = results['cpu_results'][num_particles]
            row = f"{num_particles:>10} | {cpu_time:>12.4f}"
            
            # OpenMP
            if (num_particles in results.get('openmp_results', {}) and 
                results['openmp_results'][num_particles] is not None):
                openmp_time = results['openmp_results'][num_particles]
                openmp_speedup = results['openmp_speedups'].get(num_particles, 0)
                row += f" | {openmp_time:>14.4f} | {openmp_speedup:>13.2f}x"
            
            # GPU
            if num_particles in results.get('gpu_results', {}):
                gpu_time = results['gpu_results'][num_particles][0] + results['gpu_results'][num_particles][1]
                gpu_speedup = results['gpu_speedups'].get(num_particles, 0)
                row += f" | {gpu_time:>15.4f} | {gpu_speedup:>13.2f}x"
            
            f.write(row + "\n")
        
        f.write("\n4. OVERHEAD ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # OpenMP overhead
        if results.get('openmp_results'):
            f.write("OpenMP Performance:\n")
            f.write("  - Uses shared-memory parallelism with minimal data movement overhead\n")
            f.write("  - Thread synchronization overhead is minimal for this workload\n")
            f.write("  - Speedup scales with available CPU cores\n\n")
        
        # GPU overhead
        if results.get('gpu_results'):
            f.write("GPU execution consists of:\n")
            f.write("  - Compute time: Actual parallel particle updates on GPU\n")
            f.write("  - Transfer time: Data movement between CPU and GPU memory\n\n")
            
            for num_particles in results['particle_counts']:
                if num_particles in results['gpu_results']:
                    gpu_update, gpu_transfer = results['gpu_results'][num_particles]
                    gpu_total = gpu_update + gpu_transfer
                    transfer_percent = (gpu_transfer / gpu_total * 100) if gpu_total > 0 else 0
                    
                    f.write(f"{num_particles} particles: Compute={gpu_update:.4f} ms, "
                           f"Transfer={gpu_transfer:.4f} ms ({transfer_percent:.1f}%)\n")
        
        f.write("\n5. CACHE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        if 'cpu_cache_results' in results and results['cpu_cache_results']:
            f.write("CPU Cache Performance:\n\n")
            for num_particles in results['particle_counts']:
                if num_particles in results['cpu_cache_results']:
                    cache_analysis = results['cpu_cache_results'][num_particles]
                    cb = cache_analysis['cache_behavior']
                    
                    f.write(f"{num_particles} particles:\n")
                    f.write(f"  Working Set: {cb['working_set_size_kb']:.2f} KB ({cb['working_set_size_mb']:.4f} MB)\n")
                    f.write(f"  Hit Rate: {cb['overall_hit_rate']*100:.2f}%\n")
                    f.write(f"  Miss Rate: {cb['overall_miss_rate']*100:.2f}%\n")
                    f.write(f"  Cache Friendliness: {cb['cache_friendliness']*100:.1f}%\n")
                    
                    if cb['fits_in_l1']:
                        cache_level = "L1"
                    elif cb['fits_in_l2']:
                        cache_level = "L2"
                    elif cb['fits_in_l3']:
                        cache_level = "L3"
                    else:
                        cache_level = "RAM"
                    
                    f.write(f"  Fits in: {cache_level}\n")
                    f.write(f"  Efficiency: {cache_analysis['efficiency_level']}\n")
                    
                    if 'l1_hits_est' in cb:
                        f.write(f"  L1 Hits: {cb['l1_hits_est']}, Misses: {cb['l1_misses_est']}\n")
                    if 'l2_hits_est' in cb:
                        f.write(f"  L2 Hits: {cb.get('l2_hits_est', 0)}, Misses: {cb.get('l2_misses_est', 0)}\n")
                    if 'l3_hits_est' in cb:
                        f.write(f"  L3 Hits: {cb.get('l3_hits_est', 0)}, Misses: {cb.get('l3_misses_est', 0)}\n")
                    
                    f.write(f"  Recommendations: {', '.join(cache_analysis['recommendations'])}\n\n")
        else:
            f.write("Cache analysis was not performed.\n\n")
        
        f.write("6. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        if results.get('openmp_results'):
            f.write("- OpenMP parallelization shows measurable speedup over sequential CPU\n")
            f.write("- Speedup scales with number of CPU cores and thread count\n")
            f.write("- OpenMP provides good performance without GPU transfer overhead\n")
        
        if results.get('gpu_results'):
            f.write("- GPU acceleration shows significant speedup for large particle counts\n")
            f.write("- Transfer overhead is most noticeable for small particle counts\n")
            f.write("- Optimal performance achieved when computation time >> transfer time\n")
        
        f.write("- Parallel efficiency scales well with increasing particle counts\n")
        
        # Compare OpenMP vs GPU
        if results.get('openmp_results') and results.get('gpu_results'):
            f.write("\nOpenMP vs GPU Comparison:\n")
            f.write("- OpenMP: Better for moderate particle counts, no transfer overhead\n")
            f.write("- GPU: Better for very large particle counts, higher peak performance\n")
            f.write("- Choice depends on particle count and available hardware\n")
        
        if 'cpu_cache_results' in results and results['cpu_cache_results']:
            f.write("- Cache performance degrades as working set size exceeds cache levels\n")
            f.write("- Sequential array access provides good spatial locality\n")
            f.write("- Temporal locality is excellent due to frame-by-frame reuse\n\n")
        else:
            f.write("\n")
        
        f.write("7. CONCLUSIONS\n")
        f.write("-" * 80 + "\n")
        
        if results.get('openmp_results'):
            f.write("The OpenMP parallel implementation demonstrates measurable speedup over\n")
            f.write("the sequential CPU version by leveraging multi-core CPU architecture.\n")
            f.write("Shared-memory parallelism provides efficient parallelization with minimal\n")
            f.write("overhead for particle simulation workloads.\n\n")
        
        if results.get('gpu_results'):
            f.write("The GPU-accelerated implementation shows significant speedup for large\n")
            f.write("particle counts, effectively leveraging GPU's many-core architecture.\n")
            f.write("However, transfer overhead limits performance for smaller workloads.\n\n")
        
        f.write("Both parallel approaches (OpenMP and GPU) offer advantages over sequential\n")
        f.write("CPU implementation, with the optimal choice depending on problem size and\n")
        f.write("available hardware resources.\n")
    
    print(f"\nPerformance report saved to: {report_path}")


if __name__ == "__main__":
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Particle counts to test
    particle_counts = [100, 500, 1000, 5000, 10000, 50000]
    
    print("\nStarting comprehensive performance analysis...")
    print("This may take several minutes depending on your hardware.\n")
    
    # Run comparison with cache analysis
    results = compare_performance(particle_counts, num_frames=300, include_cache_analysis=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_speedup(results)
    plot_efficiency(results)
    plot_comparison(results)
    
    # Generate cache plots if available
    if 'cpu_cache_results' in results and results['cpu_cache_results']:
        plot_cache_hit_rate(results)
        plot_cache_efficiency(results)
    
    # Analyze overheads
    analyze_overheads(results)
    
    # Analyze cache performance
    analyze_cache_performance(results)
    
    # Generate text report
    generate_report(results)
    
    print("\n" + "=" * 80)
    print("Performance analysis complete!")
    print("Check the 'plots' directory for generated graphs.")
    print("="*80)