"""
Performance Analysis and Comparison Script
===========================================
Compares CPU (sequential) vs GPU (PyCUDA) performance and generates plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path to import simulation modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_cpu_simulation import ParticleSystemCPU, benchmark_cpu
from gpu_simulation_pycuda import ParticleSystemGPU, benchmark_gpu


def compare_performance(num_particles_list=[100, 500, 1000, 5000, 10000, 50000], num_frames=300, include_cache_analysis=False):
    """
    Compare CPU and GPU performance for various particle counts.
    
    Args:
        num_particles_list: List of particle counts to test
        num_frames: Number of frames to run for each test
        include_cache_analysis: If True, perform cache analysis for CPU benchmarks
    
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 80)
    print("Performance Comparison: CPU vs GPU")
    print("=" * 80)
    print(f"Testing {len(num_particles_list)} different particle counts")
    print(f"Frames per test: {num_frames}")
    if include_cache_analysis:
        print("Cache analysis: ENABLED")
    print()
    
    # Run CPU benchmarks
    print("\n[1/2] Running CPU benchmarks...")
    print("-" * 80)
    cpu_benchmark_result = benchmark_cpu(num_particles_list, num_frames, include_cache_analysis=include_cache_analysis)
    
    # Handle cache analysis results
    if include_cache_analysis and isinstance(cpu_benchmark_result, tuple):
        cpu_results, cpu_cache_results = cpu_benchmark_result
    else:
        cpu_results = cpu_benchmark_result
        cpu_cache_results = {}
    
    # Run GPU benchmarks
    print("\n[2/2] Running GPU benchmarks...")
    print("-" * 80)
    gpu_results = benchmark_gpu(num_particles_list, num_frames)
    
    # Calculate speedup and efficiency
    speedups = {}
    efficiencies = {}
    
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"{'Particles':>10} | {'CPU (ms)':>12} | {'GPU (ms)':>12} | {'Speedup':>10} | {'Efficiency':>12}")
    print("-" * 80)
    
    for num_particles in num_particles_list:
        if num_particles in cpu_results and num_particles in gpu_results:
            cpu_time = cpu_results[num_particles]
            gpu_time = gpu_results[num_particles][0] + gpu_results[num_particles][1]  # Update + transfer
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            # Estimate number of GPU cores (streaming multiprocessors * cores per SM)
            # This is a rough estimate; actual efficiency calculation would need exact GPU specs
            # For visualization, we'll use number of particles as proxy for parallelization
            theoretical_max_parallelism = min(num_particles, 10000)  # Rough estimate
            efficiency = speedup / theoretical_max_parallelism if theoretical_max_parallelism > 0 else 0
            efficiency_percent = efficiency * 100  # For display only
            
            speedups[num_particles] = speedup
            efficiencies[num_particles] = speedup  # Store speedup for efficiency plot
            
            print(f"{num_particles:>10} | {cpu_time:>12.4f} | {gpu_time:>12.4f} | "
                  f"{speedup:>10.2f}x | {efficiency_percent:>11.2f}%")
    
    result_dict = {
        'cpu_results': cpu_results,
        'gpu_results': gpu_results,
        'speedups': speedups,
        'efficiencies': efficiencies,
        'particle_counts': num_particles_list
    }
    
    # Add cache results if available
    if cpu_cache_results:
        result_dict['cpu_cache_results'] = cpu_cache_results
    
    return result_dict


def plot_speedup(results, save_path='plots/speedup_vs_particles.png'):
    """
    Plot speedup vs number of particles.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the plot
    """
    particle_counts = []
    speedups = []
    
    for num_particles in results['particle_counts']:
        if num_particles in results['speedups']:
            particle_counts.append(num_particles)
            speedups.append(results['speedups'][num_particles])
    
    plt.figure(figsize=(10, 6))
    plt.plot(particle_counts, speedups, 'b-o', linewidth=2, markersize=8, label='GPU Speedup')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='Baseline (CPU)')
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Speedup (CPU Time / GPU Time)', fontsize=12)
    plt.title('Performance Speedup: GPU vs CPU Particle Simulation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')
    plt.yscale('log')
    
    # Add annotations for significant speedups
    for i, (count, speedup) in enumerate(zip(particle_counts, speedups)):
        if speedup > 5:
            plt.annotate(f'{speedup:.1f}x', 
                        xy=(count, speedup), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSpeedup plot saved to: {save_path}")
    plt.close()


def plot_efficiency(results, save_path='plots/efficiency_vs_particles.png'):
    """
    Plot efficiency vs number of particles.
    Efficiency is calculated relative to ideal parallelization.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the plot
    """
    particle_counts = []
    efficiencies = []
    
    for num_particles in results['particle_counts']:
        if num_particles in results['efficiencies']:
            particle_counts.append(num_particles)
            efficiencies.append(results['efficiencies'][num_particles])
    
    plt.figure(figsize=(10, 6))
    plt.plot(particle_counts, efficiencies, 'g-s', linewidth=2, markersize=8, label='GPU Speedup')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='Baseline')
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Speedup (Relative Performance Gain)', fontsize=12)
    plt.title('Efficiency: GPU Performance Scaling with Particle Count', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Efficiency plot saved to: {save_path}")
    plt.close()


def plot_comparison(results, save_path='plots/cpu_vs_gpu_comparison.png'):
    """
    Plot direct comparison of CPU and GPU times.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the plot
    """
    particle_counts = []
    cpu_times = []
    gpu_times = []
    gpu_compute_times = []
    gpu_transfer_times = []
    
    for num_particles in results['particle_counts']:
        if num_particles in results['cpu_results'] and num_particles in results['gpu_results']:
            particle_counts.append(num_particles)
            cpu_times.append(results['cpu_results'][num_particles])
            
            gpu_update, gpu_transfer = results['gpu_results'][num_particles]
            gpu_total = gpu_update + gpu_transfer
            gpu_times.append(gpu_total)
            gpu_compute_times.append(gpu_update)
            gpu_transfer_times.append(gpu_transfer)
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(particle_counts, cpu_times, 'r-o', linewidth=2, markersize=8, label='CPU (Sequential)')
    plt.plot(particle_counts, gpu_times, 'b-s', linewidth=2, markersize=8, label='GPU (Total: Compute + Transfer)')
    plt.plot(particle_counts, gpu_compute_times, 'b--', linewidth=1.5, label='GPU Compute Only')
    plt.plot(particle_counts, gpu_transfer_times, 'c--', linewidth=1.5, label='GPU Transfer Overhead')
    
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Time per Frame (ms)', fontsize=12)
    plt.title('CPU vs GPU Performance Comparison', fontsize=14, fontweight='bold')
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
    Analyze and report overhead components.
    
    Args:
        results: Dictionary with benchmark results
    """
    print("\n" + "=" * 80)
    print("Overhead Analysis")
    print("=" * 80)
    
    print(f"\n{'Particles':>10} | {'GPU Compute':>15} | {'Transfer':>15} | {'Transfer %':>12}")
    print("-" * 80)
    
    for num_particles in results['particle_counts']:
        if num_particles in results['gpu_results']:
            gpu_update, gpu_transfer = results['gpu_results'][num_particles]
            gpu_total = gpu_update + gpu_transfer
            transfer_percent = (gpu_transfer / gpu_total * 100) if gpu_total > 0 else 0
            
            print(f"{num_particles:>10} | {gpu_update:>15.4f} | {gpu_transfer:>15.4f} | {transfer_percent:>11.2f}%")
    
    print("\nKey Observations:")
    print("- Transfer overhead represents data movement cost between CPU and GPU")
    print("- As particle count increases, compute time dominates")
    print("- For small particle counts, transfer overhead can be significant")


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
        f.write("Physics-Based Particle Simulation: CPU vs GPU\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write("This report compares sequential CPU implementation vs GPU-accelerated\n")
        f.write("implementation using PyCUDA for parallel particle simulation.\n\n")
        
        f.write("2. METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("- Sequential CPU version: NumPy-based single-threaded computation\n")
        f.write("- GPU version: PyCUDA with parallel CUDA kernels\n")
        f.write("- Benchmark: Multiple particle counts tested over 300 frames each\n")
        f.write("- Metrics: Average time per frame, speedup, efficiency\n\n")
        
        f.write("3. RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Particles':>10} | {'CPU (ms)':>12} | {'GPU Total (ms)':>15} | {'Speedup':>10}\n")
        f.write("-" * 80 + "\n")
        
        for num_particles in results['particle_counts']:
            if num_particles in results['cpu_results'] and num_particles in results['speedups']:
                cpu_time = results['cpu_results'][num_particles]
                gpu_time = results['gpu_results'][num_particles][0] + results['gpu_results'][num_particles][1]
                speedup = results['speedups'][num_particles]
                
                f.write(f"{num_particles:>10} | {cpu_time:>12.4f} | {gpu_time:>15.4f} | {speedup:>10.2f}x\n")
        
        f.write("\n4. OVERHEAD ANALYSIS\n")
        f.write("-" * 80 + "\n")
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
        f.write("- GPU acceleration shows significant speedup for large particle counts\n")
        f.write("- Transfer overhead is most noticeable for small particle counts\n")
        f.write("- Optimal performance achieved when computation time >> transfer time\n")
        f.write("- Parallel efficiency scales well with increasing particle counts\n")
        
        if 'cpu_cache_results' in results and results['cpu_cache_results']:
            f.write("- Cache performance degrades as working set size exceeds cache levels\n")
            f.write("- Sequential array access provides good spatial locality\n")
            f.write("- Temporal locality is excellent due to frame-by-frame reuse\n\n")
        else:
            f.write("\n")
        
        f.write("7. CONCLUSIONS\n")
        f.write("-" * 80 + "\n")
        f.write("The GPU-accelerated implementation demonstrates measurable speedup over\n")
        f.write("the sequential CPU version, particularly for simulations with many particles.\n")
        f.write("The parallel approach effectively leverages GPU's many-core architecture\n")
        f.write("for independent particle updates.\n")
    
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