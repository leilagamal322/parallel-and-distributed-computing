"""
Script to run MPI scaling experiments across multiple process counts.

This script automates running strong and weak scaling experiments
with different numbers of MPI processes and generates comprehensive reports.
"""

import subprocess
import sys
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def run_experiment(script_path: str, num_processes: int, experiment_type: str, 
                   problem_size: int = None, elements_per_process: int = None) -> Dict:
    """
    Run an MPI experiment with specified number of processes.
    
    Returns:
        Dictionary with results
    """
    cmd = ['mpirun', '-n', str(num_processes), 'python', script_path]
    
    if experiment_type == 'strong':
        cmd.extend(['--strong-scaling', '--problem-size', str(problem_size)])
    elif experiment_type == 'weak':
        cmd.extend(['--weak-scaling', '--elements-per-process', str(elements_per_process)])
    elif experiment_type == 'latency':
        cmd.extend(['--latency-bandwidth'])
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Try to load results from JSON file
        if experiment_type == 'strong':
            result_file = f'results/strong_scaling_p{num_processes}.json'
        elif experiment_type == 'weak':
            result_file = f'results/weak_scaling_p{num_processes}.json'
        else:
            result_file = 'results/latency_bandwidth.json'
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"stderr: {e.stderr}")
        return {}
    except FileNotFoundError:
        print("Error: mpirun not found. Make sure MPI is installed and in PATH.")
        return {}
    
    return {}


def run_strong_scaling_sweep(script_path: str, problem_size: int = 10_000_000,
                             process_counts: List[int] = None) -> Dict:
    """
    Run strong scaling experiment across multiple process counts.
    
    Returns:
        Dictionary with results for all process counts
    """
    if process_counts is None:
        process_counts = [1, 2, 4, 8, 16]
    
    print("="*70)
    print("STRONG SCALING EXPERIMENT SWEEP")
    print("="*70)
    print(f"Fixed problem size: {problem_size:,} elements")
    print(f"Process counts: {process_counts}")
    print()
    
    results = {}
    times = []
    process_counts_actual = []
    
    for num_procs in process_counts:
        result = run_experiment(script_path, num_procs, 'strong', 
                               problem_size=problem_size)
        if result:
            results[num_procs] = result
            times.append(result['total_time_seconds'])
            process_counts_actual.append(num_procs)
    
    # Calculate speedup and efficiency
    if len(times) > 0 and 1 in results:
        baseline_time = results[1]['total_time_seconds']
        speedups = []
        efficiencies = []
        
        for num_procs in process_counts_actual:
            if num_procs in results:
                time_p = results[num_procs]['total_time_seconds']
                speedup = baseline_time / time_p
                efficiency = speedup / num_procs
                speedups.append(speedup)
                efficiencies.append(efficiency)
                results[num_procs]['speedup'] = speedup
                results[num_procs]['efficiency'] = efficiency
    
    return results


def run_weak_scaling_sweep(script_path: str, elements_per_process: int = 1_000_000,
                          process_counts: List[int] = None) -> Dict:
    """
    Run weak scaling experiment across multiple process counts.
    
    Returns:
        Dictionary with results for all process counts
    """
    if process_counts is None:
        process_counts = [1, 2, 4, 8, 16]
    
    print("="*70)
    print("WEAK SCALING EXPERIMENT SWEEP")
    print("="*70)
    print(f"Elements per process: {elements_per_process:,}")
    print(f"Process counts: {process_counts}")
    print()
    
    results = {}
    times = []
    process_counts_actual = []
    
    for num_procs in process_counts:
        result = run_experiment(script_path, num_procs, 'weak',
                               elements_per_process=elements_per_process)
        if result:
            results[num_procs] = result
            times.append(result['total_time_seconds'])
            process_counts_actual.append(num_procs)
    
    return results


def plot_strong_scaling(results: Dict, output_file: str = 'plots/strong_scaling.png'):
    """Plot strong scaling results."""
    os.makedirs('plots', exist_ok=True)
    
    process_counts = sorted([p for p in results.keys() if isinstance(p, int)])
    times = [results[p]['total_time_seconds'] for p in process_counts]
    speedups = [results[p].get('speedup', 0) for p in process_counts]
    efficiencies = [results[p].get('efficiency', 0) for p in process_counts]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Runtime vs processes
    axes[0].plot(process_counts, times, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Processes')
    axes[0].set_ylabel('Runtime (seconds)')
    axes[0].set_title('Strong Scaling: Runtime vs Processes')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log', base=2)
    
    # Speedup vs processes
    ideal_speedup = process_counts
    axes[1].plot(process_counts, speedups, 'o-', linewidth=2, markersize=8, 
                label='Actual Speedup')
    axes[1].plot(process_counts, ideal_speedup, '--', linewidth=2, 
                label='Ideal Speedup', alpha=0.7)
    axes[1].set_xlabel('Number of Processes')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Strong Scaling: Speedup vs Processes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log', base=2)
    
    # Efficiency vs processes
    axes[2].plot(process_counts, efficiencies, 'o-', linewidth=2, markersize=8)
    axes[2].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Ideal Efficiency')
    axes[2].set_xlabel('Number of Processes')
    axes[2].set_ylabel('Efficiency')
    axes[2].set_title('Strong Scaling: Efficiency vs Processes')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log', base=2)
    axes[2].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nStrong scaling plot saved to {output_file}")


def plot_weak_scaling(results: Dict, output_file: str = 'plots/weak_scaling.png'):
    """Plot weak scaling results."""
    os.makedirs('plots', exist_ok=True)
    
    process_counts = sorted([p for p in results.keys() if isinstance(p, int)])
    times = [results[p]['total_time_seconds'] for p in process_counts]
    problem_sizes = [results[p]['total_problem_size'] for p in process_counts]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Runtime vs processes
    axes[0].plot(process_counts, times, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Processes')
    axes[0].set_ylabel('Runtime (seconds)')
    axes[0].set_title('Weak Scaling: Runtime vs Processes')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log', base=2)
    
    # Runtime vs problem size
    axes[1].plot(problem_sizes, times, 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Total Problem Size (elements)')
    axes[1].set_ylabel('Runtime (seconds)')
    axes[1].set_title('Weak Scaling: Runtime vs Problem Size')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nWeak scaling plot saved to {output_file}")


def generate_report(results_strong: Dict, results_weak: Dict, 
                   latency_results: Dict = None):
    """Generate a comprehensive text report."""
    os.makedirs('results', exist_ok=True)
    report_file = 'results/scaling_experiments_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MPI SCALING EXPERIMENTS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Latency & Bandwidth
        if latency_results:
            f.write("A. LATENCY & BANDWIDTH MEASUREMENTS\n")
            f.write("-"*70 + "\n")
            f.write(f"Latency: {latency_results.get('latency_us', 0):.3f} microseconds\n\n")
            f.write("Bandwidth:\n")
            for size, bw in latency_results.get('bandwidth_mb_s', {}).items():
                f.write(f"  {size/1024:.1f} KB: {bw:.2f} MB/s\n")
            f.write("\n")
        
        # Strong Scaling
        f.write("B. STRONG SCALING EXPERIMENT\n")
        f.write("-"*70 + "\n")
        if results_strong:
            process_counts = sorted([p for p in results_strong.keys() if isinstance(p, int)])
            f.write(f"Fixed problem size: {results_strong[process_counts[0]]['problem_size']:,} elements\n\n")
            f.write(f"{'Processes':<12} {'Time (s)':<15} {'Speedup':<12} {'Efficiency':<12}\n")
            f.write("-"*70 + "\n")
            for p in process_counts:
                r = results_strong[p]
                speedup = r.get('speedup', 0)
                efficiency = r.get('efficiency', 0)
                f.write(f"{p:<12} {r['total_time_seconds']:<15.6f} {speedup:<12.3f} {efficiency:<12.3f}\n")
        f.write("\n")
        
        # Weak Scaling
        f.write("C. WEAK SCALING EXPERIMENT\n")
        f.write("-"*70 + "\n")
        if results_weak:
            process_counts = sorted([p for p in results_weak.keys() if isinstance(p, int)])
            elements_per_proc = results_weak[process_counts[0]]['elements_per_process']
            f.write(f"Elements per process: {elements_per_proc:,}\n\n")
            f.write(f"{'Processes':<12} {'Total Size':<15} {'Time (s)':<15}\n")
            f.write("-"*70 + "\n")
            for p in process_counts:
                r = results_weak[p]
                f.write(f"{p:<12} {r['total_problem_size']:<15,} {r['total_time_seconds']:<15.6f}\n")
        f.write("\n")
        
        # Analysis
        f.write("ANALYSIS\n")
        f.write("-"*70 + "\n")
        if results_strong and len(results_strong) > 1:
            f.write("Strong Scaling:\n")
            f.write("- Runtime should decrease as processes increase\n")
            f.write("- Speedup shows how much faster we get\n")
            f.write("- Efficiency shows how well we utilize parallel resources\n")
            f.write("- Efficiency typically decreases due to communication overhead\n")
            f.write("- This demonstrates Amdahl's Law effects\n\n")
        
        if results_weak and len(results_weak) > 1:
            f.write("Weak Scaling:\n")
            f.write("- Ideal: Runtime stays constant as problem size scales\n")
            f.write("- Real: Runtime increases due to communication overhead\n")
            f.write("- More processes = more halo exchanges\n")
            f.write("- Longer communication distances\n")
            f.write("- Larger communication volume\n\n")
    
    print(f"\nReport saved to {report_file}")


def main():
    """Main function."""
    script_path = 'src/mpi_performance_experiments.py'
    
    import argparse
    parser = argparse.ArgumentParser(description='Run MPI scaling experiments')
    parser.add_argument('--strong', action='store_true',
                       help='Run strong scaling sweep')
    parser.add_argument('--weak', action='store_true',
                       help='Run weak scaling sweep')
    parser.add_argument('--latency', action='store_true',
                       help='Run latency/bandwidth measurement')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--problem-size', type=int, default=10_000_000,
                       help='Fixed problem size for strong scaling')
    parser.add_argument('--elements-per-process', type=int, default=1_000_000,
                       help='Elements per process for weak scaling')
    parser.add_argument('--processes', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                       help='Process counts to test')
    
    args = parser.parse_args()
    
    results_strong = {}
    results_weak = {}
    latency_results = {}
    
    if args.all or args.latency:
        print("\nRunning latency/bandwidth measurement...")
        latency_results = run_experiment(script_path, 2, 'latency')
    
    if args.all or args.strong:
        results_strong = run_strong_scaling_sweep(
            script_path, 
            problem_size=args.problem_size,
            process_counts=args.processes
        )
        if results_strong:
            plot_strong_scaling(results_strong)
            # Save aggregated results
            os.makedirs('results', exist_ok=True)
            with open('results/strong_scaling_all.json', 'w') as f:
                json.dump(results_strong, f, indent=2)
    
    if args.all or args.weak:
        results_weak = run_weak_scaling_sweep(
            script_path,
            elements_per_process=args.elements_per_process,
            process_counts=args.processes
        )
        if results_weak:
            plot_weak_scaling(results_weak)
            # Save aggregated results
            os.makedirs('results', exist_ok=True)
            with open('results/weak_scaling_all.json', 'w') as f:
                json.dump(results_weak, f, indent=2)
    
    if (args.all or args.strong or args.weak) and (results_strong or results_weak):
        generate_report(results_strong, results_weak, latency_results)
    
    if not (args.all or args.strong or args.weak or args.latency):
        print("No experiment specified. Use --help for options.")
        print("\nExample usage:")
        print("  python run_mpi_scaling_experiments.py --strong")
        print("  python run_mpi_scaling_experiments.py --weak")
        print("  python run_mpi_scaling_experiments.py --all")


if __name__ == "__main__":
    main()

