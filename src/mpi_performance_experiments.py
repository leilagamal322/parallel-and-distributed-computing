
import numpy as np
import sys
import time
import json
import os
from typing import Dict, List, Tuple

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py not found. Install with: pip install mpi4py")
    sys.exit(1)


# ============================================================================
# A. LATENCY & BANDWIDTH MEASUREMENTS
# ============================================================================

def measure_latency(comm: MPI.Comm, num_iterations: int = 1000, warmup: int = 100) -> float:
    """
    Measure message latency using ping-pong test.
    
    Latency = (Round-trip time) / 2
    
    Args:
        comm: MPI communicator
        num_iterations: Number of ping-pong iterations
        warmup: Number of warmup iterations
    
    Returns:
        Average latency in seconds
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        if rank == 0:
            print("Warning: Need at least 2 processes for latency measurement")
        return 0.0
    
    # Use rank 0 and rank 1 for ping-pong
    if rank >= 2:
        return 0.0
    
    # Small message (1 byte) for latency measurement
    data = np.array([42], dtype=np.int8)
    
    # Warmup
    for _ in range(warmup):
        if rank == 0:
            comm.Send([data, MPI.INT8], dest=1, tag=1)
            comm.Recv([data, MPI.INT8], source=1, tag=2)
        elif rank == 1:
            comm.Recv([data, MPI.INT8], source=0, tag=1)
            comm.Send([data, MPI.INT8], dest=0, tag=2)
    
    comm.Barrier()
    
    # Actual measurement
    times = []
    for _ in range(num_iterations):
        if rank == 0:
            t0 = time.perf_counter()
            comm.Send([data, MPI.INT8], dest=1, tag=1)
            comm.Recv([data, MPI.INT8], source=1, tag=2)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        elif rank == 1:
            comm.Recv([data, MPI.INT8], source=0, tag=1)
            comm.Send([data, MPI.INT8], dest=0, tag=2)
    
    if rank == 0:
        avg_rtt = np.mean(times)
        latency = avg_rtt / 2.0  # One-way latency
        return latency
    
    return 0.0


def measure_bandwidth(comm: MPI.Comm, message_size_bytes: int, 
                     num_iterations: int = 100, warmup: int = 10) -> float:
    """
    Measure bandwidth using large message transfers.
    
    Bandwidth = message_size_bytes / (one-way time)
    
    Args:
        comm: MPI communicator
        message_size_bytes: Size of message in bytes
        num_iterations: Number of transfer iterations
        warmup: Number of warmup iterations
    
    Returns:
        Bandwidth in bytes per second
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        if rank == 0:
            print("Warning: Need at least 2 processes for bandwidth measurement")
        return 0.0
    
    # Use rank 0 and rank 1
    if rank >= 2:
        return 0.0
    
    # Create message buffer
    num_elements = message_size_bytes // 4  # Assuming float32 (4 bytes)
    if num_elements == 0:
        num_elements = 1
    data = np.random.rand(num_elements).astype(np.float32)
    actual_size = data.nbytes
    
    # Warmup
    for _ in range(warmup):
        if rank == 0:
            comm.Send([data, MPI.FLOAT], dest=1, tag=10)
        elif rank == 1:
            comm.Recv([data, MPI.FLOAT], source=0, tag=10)
    
    comm.Barrier()
    
    # Actual measurement
    times = []
    for _ in range(num_iterations):
        if rank == 0:
            t0 = time.perf_counter()
            comm.Send([data, MPI.FLOAT], dest=1, tag=10)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        elif rank == 1:
            comm.Recv([data, MPI.FLOAT], source=0, tag=10)
    
    if rank == 0:
        avg_time = np.mean(times)
        bandwidth = actual_size / avg_time  # bytes per second
        return bandwidth
    
    return 0.0


def run_latency_bandwidth_experiment(comm: MPI.Comm) -> Dict:
    """
    Run comprehensive latency and bandwidth measurements.
    
    Returns:
        Dictionary with latency and bandwidth results
    """
    rank = comm.Get_rank()
    
    results = {
        'latency_us': 0.0,
        'bandwidth_mb_s': {},
        'message_sizes_bytes': []
    }
    
    if rank == 0:
        print("\n" + "="*70)
        print("A. LATENCY & BANDWIDTH MEASUREMENTS")
        print("="*70)
    
    # Measure latency
    if rank == 0:
        print("\nMeasuring latency (small messages)...")
    latency = measure_latency(comm, num_iterations=1000, warmup=100)
    
    if rank == 0:
        latency_us = latency * 1e6  # Convert to microseconds
        results['latency_us'] = latency_us
        print(f"  Latency: {latency_us:.3f} microseconds")
    
    # Measure bandwidth for various message sizes
    if rank == 0:
        print("\nMeasuring bandwidth (large messages)...")
    
    # Message sizes: 1KB, 10KB, 100KB, 1MB, 10MB
    message_sizes = [1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024]
    
    for msg_size in message_sizes:
        if rank == 0:
            print(f"  Testing {msg_size / 1024:.1f} KB...", end=" ", flush=True)
        
        bandwidth = measure_bandwidth(comm, msg_size, num_iterations=50, warmup=5)
        
        if rank == 0:
            bandwidth_mb_s = bandwidth / (1024 * 1024)  # Convert to MB/s
            results['bandwidth_mb_s'][msg_size] = bandwidth_mb_s
            results['message_sizes_bytes'].append(msg_size)
            print(f"Bandwidth: {bandwidth_mb_s:.2f} MB/s")
    
    comm.Barrier()
    return results


# ============================================================================
# B. STRONG SCALING EXPERIMENT
# ============================================================================

def run_strong_scaling_experiment(comm: MPI.Comm, fixed_problem_size: int, 
                                  num_iterations: int = 100) -> Dict:
    """
    Strong scaling: Fixed problem size, varying number of processes.
    
    Measures how a fixed problem size performs as we increase processes.
    Expected: Runtime decreases, but communication overhead increases.
    
    Args:
        comm: MPI communicator
        fixed_problem_size: Fixed number of elements to process
        num_iterations: Number of computation iterations
    
    Returns:
        Dictionary with timing results and scaling metrics
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*70)
        print("B. STRONG SCALING EXPERIMENT")
        print("="*70)
        print(f"Fixed problem size: {fixed_problem_size:,} elements")
        print(f"Number of processes: {size}")
        print(f"Iterations per run: {num_iterations}")
    
    # Distribute work evenly
    elements_per_rank = fixed_problem_size // size
    remainder = fixed_problem_size % size
    
    # Rank 0 gets any remainder
    local_size = elements_per_rank + (remainder if rank == 0 else 0)
    
    # Create local data
    local_data = np.random.rand(local_size).astype(np.float64)
    
    comm.Barrier()
    
    # Warmup
    for _ in range(10):
        result = np.sum(local_data ** 2)
        comm.Allreduce([np.array([result], dtype=np.float64), MPI.DOUBLE],
                      [np.array([0.0], dtype=np.float64), MPI.DOUBLE],
                      op=MPI.SUM)
    
    comm.Barrier()
    
    # Actual measurement
    t0 = time.perf_counter()
    
    for iteration in range(num_iterations):
        # Local computation
        local_result = np.sum(local_data ** 2)
        
        # Communication: Allreduce to get global sum
        global_result = np.array([0.0], dtype=np.float64)
        comm.Allreduce([np.array([local_result], dtype=np.float64), MPI.DOUBLE],
                      [global_result, MPI.DOUBLE],
                      op=MPI.SUM)
        
        # Simulate some more computation
        local_data = local_data * 1.0001  # Small update
    
    comm.Barrier()
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    times = np.array([0.0], dtype=np.float64)
    comm.Gather([np.array([total_time], dtype=np.float64), MPI.DOUBLE],
               [times, MPI.DOUBLE], root=0)
    
    if rank == 0:
        # Get the maximum time (slowest process)
        max_time = np.max(times)
        avg_time = np.mean(times)
        
        results = {
            'problem_size': fixed_problem_size,
            'num_processes': size,
            'total_time_seconds': float(max_time),
            'avg_time_seconds': float(avg_time),
            'elements_per_process': elements_per_rank,
            'iterations': num_iterations
        }
        
        print(f"\nResults:")
        print(f"  Total time: {max_time:.6f} seconds")
        print(f"  Average time per process: {avg_time:.6f} seconds")
        print(f"  Elements per process: {elements_per_rank}")
        
        return results
    
    return {}


def run_strong_scaling_sweep(comm: MPI.Comm, fixed_problem_size: int = 10_000_000) -> Dict:
    """
    Run strong scaling experiment across multiple process counts.
    Note: This requires running the script multiple times with different -n values.
    
    For a single run, this just reports the current configuration.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    results = run_strong_scaling_experiment(comm, fixed_problem_size, num_iterations=100)
    
    if rank == 0 and results:
        # Calculate metrics if we have baseline (single process) time
        # In practice, you'd run this multiple times with mpirun -n 1, -n 2, etc.
        print(f"\nStrong Scaling Analysis:")
        print(f"  To get full scaling data, run with:")
        print(f"    mpirun -n 1 python src/mpi_performance_experiments.py --strong-scaling")
        print(f"    mpirun -n 2 python src/mpi_performance_experiments.py --strong-scaling")
        print(f"    mpirun -n 4 python src/mpi_performance_experiments.py --strong-scaling")
        print(f"    mpirun -n 8 python src/mpi_performance_experiments.py --strong-scaling")
        print(f"    mpirun -n 16 python src/mpi_performance_experiments.py --strong-scaling")
    
    return results


# ============================================================================
# C. WEAK SCALING EXPERIMENT
# ============================================================================

def run_weak_scaling_experiment(comm: MPI.Comm, elements_per_process: int,
                                num_iterations: int = 100) -> Dict:
    """
    Weak scaling: Problem size increases proportionally with processes.
    
    Each process gets the same amount of work.
    Expected: Runtime stays roughly constant (ideal), but increases due to
    communication overhead.
    
    Args:
        comm: MPI communicator
        elements_per_process: Number of elements each process handles
        num_iterations: Number of computation iterations
    
    Returns:
        Dictionary with timing results
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*70)
        print("C. WEAK SCALING EXPERIMENT")
        print("="*70)
        print(f"Elements per process: {elements_per_process:,}")
        print(f"Total problem size: {elements_per_process * size:,} elements")
        print(f"Number of processes: {size}")
        print(f"Iterations per run: {num_iterations}")
    
    # Each process gets the same amount of work
    local_data = np.random.rand(elements_per_process).astype(np.float64)
    
    comm.Barrier()
    
    # Warmup
    for _ in range(10):
        result = np.sum(local_data ** 2)
        comm.Allreduce([np.array([result], dtype=np.float64), MPI.DOUBLE],
                      [np.array([0.0], dtype=np.float64), MPI.DOUBLE],
                      op=MPI.SUM)
    
    comm.Barrier()
    
    # Actual measurement
    t0 = time.perf_counter()
    
    for iteration in range(num_iterations):
        # Local computation
        local_result = np.sum(local_data ** 2)
        
        # Communication: Allreduce to get global sum
        global_result = np.array([0.0], dtype=np.float64)
        comm.Allreduce([np.array([local_result], dtype=np.float64), MPI.DOUBLE],
                      [global_result, MPI.DOUBLE],
                      op=MPI.SUM)
        
        # Simulate some more computation
        local_data = local_data * 1.0001  # Small update
    
    comm.Barrier()
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    times = np.array([0.0], dtype=np.float64)
    comm.Gather([np.array([total_time], dtype=np.float64), MPI.DOUBLE],
               [times, MPI.DOUBLE], root=0)
    
    if rank == 0:
        # Get the maximum time (slowest process)
        max_time = np.max(times)
        avg_time = np.mean(times)
        
        results = {
            'elements_per_process': elements_per_process,
            'total_problem_size': elements_per_process * size,
            'num_processes': size,
            'total_time_seconds': float(max_time),
            'avg_time_seconds': float(avg_time),
            'iterations': num_iterations
        }
        
        print(f"\nResults:")
        print(f"  Total time: {max_time:.6f} seconds")
        print(f"  Average time per process: {avg_time:.6f} seconds")
        print(f"  Time per element: {max_time / elements_per_process * 1e6:.3f} microseconds")
        
        return results
    
    return {}


def run_weak_scaling_sweep(comm: MPI.Comm, elements_per_process: int = 1_000_000) -> Dict:
    """
    Run weak scaling experiment across multiple process counts.
    Note: This requires running the script multiple times with different -n values.
    """
    rank = comm.Get_rank()
    
    results = run_weak_scaling_experiment(comm, elements_per_process, num_iterations=100)
    
    if rank == 0 and results:
        print(f"\nWeak Scaling Analysis:")
        print(f"  To get full scaling data, run with:")
        print(f"    mpirun -n 1 python src/mpi_performance_experiments.py --weak-scaling")
        print(f"    mpirun -n 2 python src/mpi_performance_experiments.py --weak-scaling")
        print(f"    mpirun -n 4 python src/mpi_performance_experiments.py --weak-scaling")
        print(f"    mpirun -n 8 python src/mpi_performance_experiments.py --weak-scaling")
        print(f"    mpirun -n 16 python src/mpi_performance_experiments.py --weak-scaling")
    
    return results


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def save_results(results: Dict, filename: str):
    """Save results to JSON file."""
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filepath}")


def main():
    """Main function to run performance experiments."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    import argparse
    parser = argparse.ArgumentParser(description='MPI Performance Experiments')
    parser.add_argument('--latency-bandwidth', action='store_true',
                       help='Run latency and bandwidth measurements')
    parser.add_argument('--strong-scaling', action='store_true',
                       help='Run strong scaling experiment')
    parser.add_argument('--weak-scaling', action='store_true',
                       help='Run weak scaling experiment')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--problem-size', type=int, default=10_000_000,
                       help='Fixed problem size for strong scaling (default: 10M)')
    parser.add_argument('--elements-per-process', type=int, default=1_000_000,
                       help='Elements per process for weak scaling (default: 1M)')
    
    args = parser.parse_args()
    
    if rank == 0:
        print("="*70)
        print("MPI PERFORMANCE EXPERIMENTS")
        print("="*70)
        print(f"MPI Size: {size} processes")
        print(f"MPI Rank: {rank}")
        print()
    
    all_results = {}
    
    # Run experiments based on arguments
    if args.all or args.latency_bandwidth:
        results = run_latency_bandwidth_experiment(comm)
        if rank == 0:
            all_results['latency_bandwidth'] = results
            save_results(results, 'latency_bandwidth.json')
    
    if args.all or args.strong_scaling:
        results = run_strong_scaling_sweep(comm, fixed_problem_size=args.problem_size)
        if rank == 0 and results:
            all_results['strong_scaling'] = results
            save_results(results, f'strong_scaling_p{size}.json')
    
    if args.all or args.weak_scaling:
        results = run_weak_scaling_sweep(comm, elements_per_process=args.elements_per_process)
        if rank == 0 and results:
            all_results['weak_scaling'] = results
            save_results(results, f'weak_scaling_p{size}.json')
    
    if rank == 0:
        if not (args.all or args.latency_bandwidth or args.strong_scaling or args.weak_scaling):
            print("\nNo experiment specified. Use --help for options.")
            print("\nExample usage:")
            print("  mpirun -n 2 python src/mpi_performance_experiments.py --latency-bandwidth")
            print("  mpirun -n 4 python src/mpi_performance_experiments.py --strong-scaling")
            print("  mpirun -n 8 python src/mpi_performance_experiments.py --weak-scaling")
            print("  mpirun -n 2 python src/mpi_performance_experiments.py --all")
        else:
            print("\n" + "="*70)
            print("All experiments completed!")
            print("="*70)


if __name__ == "__main__":
    main()

