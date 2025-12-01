# MPI Performance Experiments

This document describes the MPI performance experiments implemented for measuring latency, bandwidth, strong scaling, and weak scaling.

## Overview

The performance experiments consist of three main components:

### A. Latency & Bandwidth Measurements
- **Latency**: Measures one-way message latency using ping-pong test
  - Formula: `Latency = (Round-trip time) / 2`
  - Uses small messages (1 byte) to measure pure communication overhead
- **Bandwidth**: Measures data transfer rate for large messages
  - Formula: `Bandwidth = message_size_bytes / (one-way time)`
  - Tests multiple message sizes: 1KB, 10KB, 100KB, 1MB, 10MB

### B. Strong Scaling Experiment
- **Definition**: Measures how a fixed problem size performs as we increase the number of processes
- **Expected Behavior**:
  - Runtime decreases as more processes are added
  - Communication overhead increases
  - Speedup eventually saturates
  - Efficiency drops as number of processes increases
- **Concepts Demonstrated**:
  - Amdahl's Law
  - Communication overhead
  - Parallel efficiency

### C. Weak Scaling Experiment
- **Definition**: Measures performance when problem size increases proportionally with number of processes
- **Expected Behavior**:
  - Ideal: Runtime stays roughly constant
  - Real: Runtime slowly increases due to:
    - More halo exchanges
    - Longer communication distances
    - Larger communication volume
- **Concepts Demonstrated**:
  - Communication-computation ratio
  - How algorithms scale to bigger clusters
  - Importance of minimizing communication

## Installation

1. Install MPI (OpenMPI or MPICH):
   - **Linux (Ubuntu/Debian)**: `sudo apt-get install openmpi-bin libopenmpi-dev`
   - **macOS**: `brew install openmpi`
   - **Windows**: Install MS-MPI from Microsoft

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify MPI installation:
   ```bash
   mpirun --version
   ```

4. Verify mpi4py:
   ```bash
   python -c "from mpi4py import MPI; print(MPI.Get_version())"
   ```

## Usage

### Running Individual Experiments

#### Latency & Bandwidth
```bash
mpirun -n 2 python src/mpi_performance_experiments.py --latency-bandwidth
```

#### Strong Scaling (Single Run)
```bash
# Run with 4 processes
mpirun -n 4 python src/mpi_performance_experiments.py --strong-scaling --problem-size 10000000
```

#### Weak Scaling (Single Run)
```bash
# Run with 8 processes
mpirun -n 8 python src/mpi_performance_experiments.py --weak-scaling --elements-per-process 1000000
```

#### All Experiments
```bash
mpirun -n 2 python src/mpi_performance_experiments.py --all
```

### Running Scaling Sweeps

The `run_mpi_scaling_experiments.py` script automates running experiments across multiple process counts:

#### Strong Scaling Sweep
```bash
python run_mpi_scaling_experiments.py --strong --problem-size 10000000
```

This will automatically run with 1, 2, 4, 8, and 16 processes and generate:
- Performance plots in `plots/strong_scaling.png`
- JSON results in `results/strong_scaling_all.json`
- Text report in `results/scaling_experiments_report.txt`

#### Weak Scaling Sweep
```bash
python run_mpi_scaling_experiments.py --weak --elements-per-process 1000000
```

#### All Experiments
```bash
python run_mpi_scaling_experiments.py --all
```

#### Custom Process Counts
```bash
python run_mpi_scaling_experiments.py --strong --processes 1 2 4 8 16 32
```

## Output Files

### Results Directory
- `results/latency_bandwidth.json` - Latency and bandwidth measurements
- `results/strong_scaling_p{N}.json` - Strong scaling results for N processes
- `results/weak_scaling_p{N}.json` - Weak scaling results for N processes
- `results/strong_scaling_all.json` - Aggregated strong scaling results
- `results/weak_scaling_all.json` - Aggregated weak scaling results
- `results/scaling_experiments_report.txt` - Comprehensive text report

### Plots Directory
- `plots/strong_scaling.png` - Strong scaling plots (runtime, speedup, efficiency)
- `plots/weak_scaling.png` - Weak scaling plots (runtime vs processes, runtime vs problem size)

## Understanding the Results

### Strong Scaling Analysis

**Speedup Calculation**:
```
Speedup(p) = T(1) / T(p)
```
where:
- `T(1)` = runtime with 1 process
- `T(p)` = runtime with p processes

**Efficiency Calculation**:
```
Efficiency(p) = Speedup(p) / p
```

**Expected Patterns**:
- Speedup increases but less than linearly (due to Amdahl's Law)
- Efficiency decreases as processes increase
- Communication overhead becomes dominant at high process counts

### Weak Scaling Analysis

**Ideal Behavior**:
- Runtime remains constant as problem size scales
- Each process does the same amount of work

**Real Behavior**:
- Runtime increases slightly due to:
  - More communication operations
  - Longer communication paths
  - Increased synchronization overhead

**Quality Metric**:
- Compare actual runtime to baseline (1 process)
- Good weak scaling: runtime increases < 20% when doubling processes

## Example Results Interpretation

### Strong Scaling Example
```
Processes | Time (s) | Speedup | Efficiency
----------|----------|---------|-----------
1         | 10.0     | 1.00    | 1.00
2         | 5.5      | 1.82    | 0.91
4         | 3.2      | 3.13    | 0.78
8         | 2.1      | 4.76    | 0.60
16        | 1.8      | 5.56    | 0.35
```

**Analysis**:
- Good speedup up to 4 processes (efficiency > 0.75)
- Efficiency drops significantly at 16 processes
- Communication overhead dominates at high process counts

### Weak Scaling Example
```
Processes | Total Size | Time (s) | Time/Element (μs)
----------|------------|----------|------------------
1         | 1,000,000  | 2.0      | 2.00
2         | 2,000,000  | 2.1      | 1.05
4         | 4,000,000  | 2.3      | 0.58
8         | 8,000,000  | 2.6      | 0.33
16        | 16,000,000 | 3.2      | 0.20
```

**Analysis**:
- Runtime increases slowly (good weak scaling)
- Time per element decreases (benefit of parallelization)
- Communication overhead visible but manageable

## Troubleshooting

### MPI Not Found
```bash
# Check if MPI is installed
which mpirun
mpirun --version

# If not found, install MPI (see Installation section)
```

### mpi4py Import Error
```bash
# Reinstall mpi4py
pip uninstall mpi4py
pip install mpi4py

# On some systems, you may need to specify MPI path
MPICC=/usr/bin/mpicc pip install mpi4py
```

### Permission Errors
- Ensure MPI is properly configured
- Check that you have permissions to run MPI processes
- On clusters, ensure you're using the correct MPI module

### Performance Issues
- Use appropriate number of processes (don't exceed CPU cores)
- Ensure problem size is large enough to see scaling effects
- Run multiple iterations for statistical significance
- Use warmup iterations to avoid cold start effects

## Advanced Usage

### Custom Problem Sizes
```bash
# Strong scaling with 100 million elements
python run_mpi_scaling_experiments.py --strong --problem-size 100000000

# Weak scaling with 10 million elements per process
python run_mpi_scaling_experiments.py --weak --elements-per-process 10000000
```

### Running on Clusters
```bash
# Using SLURM
sbatch --ntasks=16 --wrap="python run_mpi_scaling_experiments.py --all"

# Using PBS
qsub -l select=4:ncpus=4:mpiprocs=4 script.sh
```

## Theory Background

### Amdahl's Law
Strong scaling is limited by the serial fraction of the program:
```
Speedup ≤ 1 / (s + (1-s)/p)
```
where:
- `s` = serial fraction
- `p` = number of processes

### Communication Overhead
As processes increase:
- More communication operations needed
- Longer communication paths (logarithmic in process count)
- Synchronization overhead increases

### Weak Scaling Ideal
If work per process is constant:
- Computation time stays constant
- Communication overhead grows (but ideally slowly)
- Overall runtime should remain roughly constant

## References

- MPI Standard: https://www.mpi-forum.org/
- mpi4py Documentation: https://mpi4py.readthedocs.io/
- Parallel Computing Concepts: Amdahl's Law, Gustafson's Law

