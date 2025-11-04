# Physics-Based Particle Simulation with Parallel Computing

A parallel computing project demonstrating multiple parallelization approaches: OpenMP (shared-memory), GPU acceleration (PyCUDA), and sequential CPU baseline for physics-based particle simulation with real-time Pygame visualization.

## 🎯 Project Overview

This project implements a particle simulation system that:
- Simulates thousands of particles with gravity and collision physics
- Compares sequential CPU (NumPy) vs OpenMP (shared-memory) vs GPU (PyCUDA) implementations
- Provides real-time visualization using Pygame
- Measures and analyzes performance speedup and efficiency

## 📁 Project Structure

```
parallel/
├── src/
│   ├── baseline_cpu_simulation.py      # Sequential CPU implementation
│   ├── openmp_simulation.py            # OpenMP parallel implementation (Python wrapper)
│   ├── particle_simulation_openmp.cpp  # OpenMP C++ core
│   ├── gpu_simulation_pycuda.py        # GPU-accelerated implementation
│   └── performance_analysis.py         # Benchmarking and plotting
├── data/                                # Input data (optional)
├── plots/                               # Generated performance plots
├── setup_openmp.py                      # Script to compile OpenMP extension
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🔧 Requirements

### Hardware
- Multi-core CPU (for OpenMP)
- NVIDIA GPU with CUDA support (Compute Capability 2.0 or higher) - Optional, for GPU version
- CUDA Toolkit installed - Optional, for GPU version

### Software
- Python 3.7+
- C++ compiler with OpenMP support (g++, MinGW-w64, or MSVC)
- CUDA Toolkit (for PyCUDA) - Optional
- Required Python packages (see `requirements.txt`)

## 📦 Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Compile OpenMP extension**:
   ```bash
   python setup_openmp.py
   ```
   
   Or manually:
   - **Windows (MinGW)**:
     ```bash
     g++ -fopenmp -O3 -shared -o src/particle_simulation_openmp.dll src/particle_simulation_openmp.cpp
     ```
   - **Linux/macOS**:
     ```bash
     g++ -fopenmp -O3 -shared -fPIC -o src/libparticle_simulation_openmp.so src/particle_simulation_openmp.cpp
     ```
   
   **Note**: If you don't have a C++ compiler with OpenMP:
   - **Windows**: Install [MinGW-w64](https://www.mingw-w64.org/) or Visual Studio
   - **Linux**: `sudo apt-get install g++ libomp-dev`
   - **macOS**: `brew install libomp`

3. **Install CUDA Toolkit** (optional, for GPU version):
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your OS
   - Verify PyCUDA installation:
     ```python
     import pycuda.autoinit
     import pycuda.driver as cuda
     print(cuda.Device(0).name())
     ```

## 🚀 Usage

### Interactive Visualization

**CPU Version (Sequential):**
```bash
python src/baseline_cpu_simulation.py [num_particles]
```

**OpenMP Version (Shared-Memory Parallel):**
```bash
python src/openmp_simulation.py [num_particles] [num_threads]
```

**GPU Version (CUDA):**
```bash
python src/gpu_simulation_pycuda.py [num_particles]
```

Examples:
```bash
# Run CPU with default 1000 particles
python src/baseline_cpu_simulation.py

# Run OpenMP with 10000 particles and 8 threads
python src/openmp_simulation.py 10000 8

# Run GPU with 10000 particles
python src/gpu_simulation_pycuda.py 10000
```

### Performance Benchmarking

**CPU Benchmark:**
```bash
python src/baseline_cpu_simulation.py benchmark
```

**OpenMP Benchmark:**
```bash
python src/openmp_simulation.py benchmark
```

**GPU Benchmark:**
```bash
python src/gpu_simulation_pycuda.py benchmark
```

**Quick Comparison (All Implementations):**
```bash
python run_comparison.py [num_particles] [num_frames]
```

**Full Performance Analysis:**
```bash
python src/performance_analysis.py
```

This will:
- Run benchmarks on CPU, OpenMP, and GPU (if available)
- Generate performance plots in `plots/`
- Create a text report with detailed analysis

## 🎮 Controls

- **ESC** or **Close Window**: Exit simulation
- Particle count can be adjusted via command-line argument

## 📊 Performance Analysis

The `performance_analysis.py` script generates:

1. **Speedup Plot** (`plots/speedup_vs_particles.png`): Shows GPU speedup over CPU
2. **Efficiency Plot** (`plots/efficiency_vs_particles.png`): Shows performance scaling
3. **Comparison Plot** (`plots/cpu_vs_gpu_comparison.png`): Direct CPU vs GPU time comparison

### Expected Results

- **Small particle counts (< 1000)**: 
  - CPU/OpenMP may be faster than GPU due to transfer overhead
  - OpenMP typically 2-4x faster than sequential CPU (depending on core count)
- **Medium particle counts (1000-10000)**: 
  - OpenMP shows 2-8x speedup over CPU (scales with core count)
  - GPU shows moderate speedup (2-5x)
- **Large particle counts (> 10000)**: 
  - OpenMP scales well with core count (2-16x speedup)
  - GPU shows significant speedup (5-20x+)

## 🔬 Implementation Details

### CPU Version (`baseline_cpu_simulation.py`)
- Sequential NumPy array operations
- Single-threaded particle updates
- Direct memory access

### OpenMP Version (`openmp_simulation.py` + `particle_simulation_openmp.cpp`)
- C++ core with OpenMP parallelization
- Shared-memory parallel processing
- Each particle updated by separate CPU thread
- Configurable thread count (default: system cores)
- Python wrapper using ctypes

### GPU Version (`gpu_simulation_pycuda.py`)
- CUDA kernel with parallel thread execution
- Each particle updated by separate GPU thread
- Block size: 256 threads
- Data transfer: Host ↔ Device memory

**OpenMP Parallel Region:**
```cpp
#pragma omp parallel for
for (int i = 0; i < num_particles; i++) {
    // Update particle physics
}
```

**CUDA Kernel:**
```cuda
__global__ void update_particles(
    float *x, float *y, float *vx, float *vy,
    float gravity, float dt, float width, float height, int n
)
```

### Physics Model
- **Gravity**: Constant downward acceleration
- **Velocity**: Updated via `v += a * dt`
- **Position**: Updated via `p += v * dt`
- **Collisions**: Elastic bouncing with damping (coefficient: 0.8)

## 📈 Performance Metrics

The analysis includes:
- **Speedup**: `T_CPU / T_GPU`
- **Efficiency**: Speedup relative to ideal parallelization
- **Overhead Analysis**: GPU compute time vs data transfer time

## 🎨 Visualization Features

- Real-time particle rendering
- Smooth 60 FPS animation
- Performance metrics overlay (FPS, update time)
- White particles on black background

## 🔍 Troubleshooting

### OpenMP Compilation Issues

**g++ not found:**
- Windows: Install MinGW-w64 or use Visual Studio
- Linux: `sudo apt-get install g++`
- macOS: `xcode-select --install`

**OpenMP not supported:**
- Linux: `sudo apt-get install libomp-dev`
- macOS: `brew install libomp`
- Windows: Ensure MinGW-w64 has OpenMP support

**Library not loading:**
- Check that the compiled library exists in `src/` directory
- Verify library name matches platform (`.dll`, `.so`, or `.dylib`)
- Try recompiling: `python setup_openmp.py`

### PyCUDA Installation Issues
```bash
# Install with specific CUDA version
pip install pycuda --no-cache-dir

# Or build from source if needed
```

### CUDA Not Found
- Ensure CUDA Toolkit is installed
- Check CUDA paths are in system PATH
- Verify GPU is CUDA-compatible: `nvidia-smi`

### Performance Issues
- Reduce particle count for testing
- Check OpenMP thread count: Set `OMP_NUM_THREADS` environment variable
- Check GPU utilization with `nvidia-smi`
- Ensure sufficient GPU memory

## 📝 Report Requirements

For the course project, this implementation provides:
1. ✅ Baseline sequential implementation
2. ✅ OpenMP shared-memory parallelization
3. ✅ GPU parallelization with PyCUDA
4. ✅ Real-time visualization with Pygame
5. ✅ Performance benchmarking and comparison
6. ✅ Speedup and efficiency analysis
7. ✅ Overhead discussion (data transfer, synchronization, thread management)

## 🎓 Academic Context

This project demonstrates:
- **Parallel Computing Fundamentals**: Sequential vs OpenMP vs GPU comparison
- **Shared-Memory Parallelism**: OpenMP thread-based parallelization
- **Distributed-Memory Parallelism**: GPU offloading with CUDA
- **Amdahl's Law**: Serial portion limitations (data transfer, synchronization)
- **Gustafson's Law**: Scaling with problem size
- **Thread Scaling**: OpenMP performance vs core count
- **Overhead Analysis**: Memory transfer costs, thread synchronization
- **Performance Measurement**: Precise timing and profiling

## 🔮 Future Enhancements

Potential extensions:
- Collision detection between particles
- Variable gravity or wind forces
- Color gradients based on particle speed
- Interactive controls for particle count
- Shared memory optimization (OpenMP)
- Texture memory usage (CUDA)
- Multi-GPU support
- Hybrid OpenMP+CUDA implementation
- MPI for distributed-memory systems

## 📄 License

Educational project for Parallel Computing course.

## 👤 Author

Parallelization Lead - Phase 1: Parallel Foundations

---

**Note**: 
- OpenMP version requires compilation of the C++ extension (run `python setup_openmp.py`)
- GPU version requires a CUDA-compatible GPU and proper drivers installed
- All implementations can be compared using `python run_comparison.py`

