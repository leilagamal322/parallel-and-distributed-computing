"""
MPI Performance Experiments — Phase 2
Supports:
  --strong-scaling
"""

import argparse
import time
import numpy as np
from mpi4py import MPI
import os
import json

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def strong_scaling_experiment(problem_size, iterations=100):
    local_size = problem_size // size
    data = np.ones(local_size, dtype=np.float64)

    for _ in range(5):
        data *= 1.0000001

    comm.Barrier()
    t0 = time.perf_counter()

    for _ in range(iterations):
        data *= 1.0000001
        comm.Barrier()

    t1 = time.perf_counter()
    local_time = t1 - t0

    if rank == 0:
        all_times = np.empty(size, dtype=np.float64)
    else:
        all_times = None

    comm.Gather(np.array(local_time, dtype=np.float64), all_times, root=0)

    if rank == 0:
        avg_time = np.mean(all_times)
        total_time = np.max(all_times)

        results = {
            "processes": size,
            "problem_size": problem_size,
            "iterations": iterations,
            "total_time": total_time,
            "avg_time": avg_time
        }

        os.makedirs("results", exist_ok=True)
        out = f"results/strong_scaling_p{size}.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {out}")
        print(f"Avg time: {avg_time * 1000:.6f} ms/frame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong-scaling", action="store_true")
    parser.add_argument("--problem-size", type=int, default=10_000_000)
    args = parser.parse_args()

    if not args.strong_scaling:
        if rank == 0:
            print("Error: No experiment selected")
        return

    if rank == 0:
        print("=" * 70)
        print("MPI PERFORMANCE EXPERIMENTS")
        print("=" * 70)
        print(f"MPI Size: {size}")

    strong_scaling_experiment(args.problem_size)


if __name__ == "__main__":
    main()
