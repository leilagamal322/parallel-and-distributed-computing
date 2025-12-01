# Running MPI Performance Experiments

## Current Status

**MPI Runtime**: Not installed (required to run experiments)
**mpi4py**: Installed ✓

## Quick Start (After Installing MPI)

Once MS-MPI is installed on Windows, you can run:

### 1. Latency & Bandwidth Test
```bash
mpirun -n 2 python src/mpi_performance_experiments.py --latency-bandwidth
```

**Expected Output:**
- Latency measurement (microseconds)
- Bandwidth measurements for 1KB, 10KB, 100KB, 1MB, 10MB messages
- Results saved to `results/latency_bandwidth.json`

### 2. Strong Scaling Experiment (Single Run)
```bash
mpirun -n 4 python src/mpi_performance_experiments.py --strong-scaling --problem-size 10000000
```

**Expected Output:**
- Runtime for fixed problem size (10M elements) with 4 processes
- Results saved to `results/strong_scaling_p4.json`

### 3. Weak Scaling Experiment (Single Run)
```bash
mpirun -n 8 python src/mpi_performance_experiments.py --weak-scaling --elements-per-process 1000000
```

**Expected Output:**
- Runtime for 8 processes, each handling 1M elements (8M total)
- Results saved to `results/weak_scaling_p8.json`

### 4. Automated Scaling Sweeps

Run experiments across multiple process counts automatically:

```bash
# Strong scaling sweep (1, 2, 4, 8, 16 processes)
python run_mpi_scaling_experiments.py --strong --problem-size 10000000

# Weak scaling sweep (1, 2, 4, 8, 16 processes)
python run_mpi_scaling_experiments.py --weak --elements-per-process 1000000

# All experiments
python run_mpi_scaling_experiments.py --all
```

**This will:**
- Automatically run with different process counts
- Generate plots in `plots/` directory
- Create comprehensive report in `results/scaling_experiments_report.txt`

## Installation Steps for Windows

1. **Download MS-MPI:**
   - Visit: https://www.microsoft.com/en-us/download/details.aspx?id=57467
   - Download and install:
     - `msmpisdk.msi` (SDK - required)
     - `msmpisetup.exe` (Runtime - required)

2. **Add to PATH** (usually automatic, but verify):
   - `C:\Program Files\Microsoft MPI\Bin`
   - `C:\Program Files (x86)\Microsoft SDKs\MPI\Bin`

3. **Verify Installation:**
   ```bash
   mpiexec --version
   ```

4. **Test mpi4py:**
   ```bash
   python -c "from mpi4py import MPI; print(MPI.Get_version())"
   ```

5. **Run Experiments:**
   ```bash
   mpirun -n 2 python src/mpi_performance_experiments.py --latency-bandwidth
   ```

## What Each Experiment Measures

### A. Latency & Bandwidth
- **Latency**: One-way message latency using ping-pong test
  - Formula: `Latency = (Round-trip time) / 2`
  - Uses 1-byte messages to measure pure communication overhead
  
- **Bandwidth**: Data transfer rate for large messages
  - Formula: `Bandwidth = message_size_bytes / (one-way time)`
  - Tests: 1KB, 10KB, 100KB, 1MB, 10MB

### B. Strong Scaling
- **Fixed problem size** (e.g., 10 million elements)
- **Varying process counts** (1, 2, 4, 8, 16)
- **Measures:**
  - Runtime as processes increase
  - Speedup = T(1) / T(p)
  - Efficiency = Speedup / p
- **Expected**: Runtime decreases, but efficiency drops due to communication overhead

### C. Weak Scaling
- **Proportional problem size** (work per process stays constant)
- **Varying process counts** (1, 2, 4, 8, 16)
- **Measures:**
  - Runtime as problem size scales
  - Time per element
- **Expected**: Runtime stays roughly constant (ideal), but increases slightly due to communication

## Output Files

After running experiments, you'll find:

```
results/
├── latency_bandwidth.json          # Latency & bandwidth results
├── strong_scaling_p1.json          # Strong scaling with 1 process
├── strong_scaling_p2.json          # Strong scaling with 2 processes
├── strong_scaling_p4.json          # etc.
├── strong_scaling_all.json         # Aggregated strong scaling
├── weak_scaling_p1.json            # Weak scaling with 1 process
├── weak_scaling_all.json            # Aggregated weak scaling
└── scaling_experiments_report.txt  # Comprehensive text report

plots/
├── strong_scaling.png              # Strong scaling plots
└── weak_scaling.png                # Weak scaling plots
```

## Example Results

### Strong Scaling (Expected)
```
Processes | Time (s) | Speedup | Efficiency
----------|----------|---------|-----------
1         | 10.0     | 1.00    | 1.00
2         | 5.5      | 1.82    | 0.91
4         | 3.2      | 3.13    | 0.78
8         | 2.1      | 4.76    | 0.60
16        | 1.8      | 5.56    | 0.35
```

### Weak Scaling (Expected)
```
Processes | Total Size | Time (s) | Time/Element (μs)
----------|------------|----------|------------------
1         | 1,000,000  | 2.0      | 2.00
2         | 2,000,000  | 2.1      | 1.05
4         | 4,000,000  | 2.3      | 0.58
8         | 8,000,000  | 2.6      | 0.33
16        | 16,000,000 | 3.2      | 0.20
```

## Troubleshooting

### "mpirun not found"
- Install MS-MPI (see Installation Steps)
- Restart terminal after installation
- Verify PATH includes MPI bin directory

### "cannot load MPI library"
- Ensure MS-MPI SDK is installed (not just runtime)
- Reinstall mpi4py: `pip uninstall mpi4py && pip install mpi4py`
- Check that `msmpi.dll` exists in system directories

### Experiments run but show errors
- Ensure at least 2 processes for latency/bandwidth tests
- Check that problem sizes are reasonable for your system
- Verify all processes can communicate (check firewall/network settings)

## Next Steps

1. Install MS-MPI following the steps above
2. Verify installation with `mpiexec --version`
3. Run a simple test: `mpirun -n 2 python src/mpi_performance_experiments.py --latency-bandwidth`
4. Run full scaling experiments: `python run_mpi_scaling_experiments.py --all`

