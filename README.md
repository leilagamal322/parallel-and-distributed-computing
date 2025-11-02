# Physics-Based Particle Simulation with PyCUDA

A parallel computing project demonstrating GPU acceleration using PyCUDA for physics-based particle simulation with real-time Pygame visualization.

## 🎯 Project Overview

This project implements a particle simulation system that:
- Simulates thousands of particles with gravity and collision physics
- Compares sequential CPU (NumPy) vs parallel GPU (PyCUDA) implementations
- Provides real-time visualization using Pygame
- Measures and analyzes performance speedup and efficiency

## 📁 Project Structure

```
parallel/
├── src/
│   ├── baseline_cpu_simulation.py    # Sequential CPU implementation
│   ├── gpu_simulation_pycuda.py      # GPU-accelerated implementation
│   └── performance_analysis.py       # Benchmarking and plotting
├── data/                              # Input data (optional)
├── plots/                             # Generated performance plots
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🔧 Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 2.0 or higher)
- CUDA Toolkit installed

### Software
- Python 3.7+
- CUDA Toolkit (for PyCUDA)
- Required Python packages (see `requirements.txt`)

## 📦 Installation

1. **Install CUDA Toolkit** (if not already installed):
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your OS

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify PyCUDA installation**:
   ```python
   import pycuda.autoinit
   import pycuda.driver as cuda
   print(cuda.Device(0).name())
   ```

## 🚀 Usage

### Interactive Visualization

**CPU Version:**
```bash
python src/baseline_cpu_simulation.py [num_particles]
```

**GPU Version:**
```bash
python src/gpu_simulation_pycuda.py [num_particles]
```

Examples:
```bash
# Run with default 1000 particles
python src/gpu_simulation_pycuda.py

# Run with 10000 particles
python src/gpu_simulation_pycuda.py 10000
```

### Performance Benchmarking

**CPU Benchmark:**
```bash
python src/baseline_cpu_simulation.py benchmark
```

**GPU Benchmark:**
```bash
python src/gpu_simulation_pycuda.py benchmark
```

**Full Performance Analysis:**
```bash
python src/performance_analysis.py
```

This will:
- Run benchmarks on both CPU and GPU
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

- **Small particle counts (< 1000)**: CPU may be faster due to GPU transfer overhead
- **Medium particle counts (1000-10000)**: GPU shows moderate speedup (2-5x)
- **Large particle counts (> 10000)**: GPU shows significant speedup (5-20x+)

## 🔬 Implementation Details

### CPU Version (`baseline_cpu_simulation.py`)
- Sequential NumPy array operations
- Single-threaded particle updates
- Direct memory access

### GPU Version (`gpu_simulation_pycuda.py`)
- CUDA kernel with parallel thread execution
- Each particle updated by separate GPU thread
- Block size: 256 threads
- Data transfer: Host ↔ Device memory

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
- Check GPU utilization with `nvidia-smi`
- Ensure sufficient GPU memory

## 📝 Report Requirements

For the course project, this implementation provides:
1. ✅ Baseline sequential implementation
2. ✅ GPU parallelization with PyCUDA
3. ✅ Real-time visualization with Pygame
4. ✅ Performance benchmarking and comparison
5. ✅ Speedup and efficiency analysis
6. ✅ Overhead discussion (data transfer, synchronization)

## 🎓 Academic Context

This project demonstrates:
- **Parallel Computing Fundamentals**: GPU vs CPU comparison
- **Amdahl's Law**: Serial portion limitations (data transfer)
- **Gustafson's Law**: Scaling with problem size
- **Overhead Analysis**: Memory transfer costs
- **Performance Measurement**: Precise timing and profiling

## 🔮 Future Enhancements

Potential extensions:
- Collision detection between particles
- Variable gravity or wind forces
- Color gradients based on particle speed
- Interactive controls for particle count
- Shared memory optimization
- Texture memory usage
- Multi-GPU support

## 📄 License

Educational project for Parallel Computing course.

## 👤 Author

Parallelization Lead - Phase 1: Parallel Foundations

---

**Note**: Ensure you have a CUDA-compatible GPU and proper drivers installed before running the GPU version.

