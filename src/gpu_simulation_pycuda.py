"""
GPU-Accelerated Particle Simulation using PyCUDA
=================================================
This version offloads particle physics computation to the GPU for parallel processing.
"""

import numpy as np
import pygame
import time
import os
import sys

# Add CUDA bin directory to DLL search path for Windows
if sys.platform == 'win32':
    # Try multiple possible CUDA paths
    possible_paths = [
        os.environ.get('CUDA_PATH'),
        os.environ.get('CUDA_PATH_V13_0'),
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0',
    ]
    for cuda_path in possible_paths:
        if cuda_path and os.path.exists(cuda_path):
            cuda_bin = os.path.join(cuda_path, 'bin')
            if os.path.exists(cuda_bin):
                os.add_dll_directory(cuda_bin)
                # Also add to PATH
                os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
except ImportError as e:
    print("=" * 70)
    print("ERROR: Cannot load PyCUDA/CUDA")
    print("=" * 70)
    print(f"Error: {e}")
    print("\nPyCUDA requires CUDA Toolkit runtime libraries to be installed.")
    print("Your system has:")
    print("  - NVIDIA GPU: Detected (GTX 1650)")
    print("  - CUDA Drivers: Installed (CUDA 12.5)")
    print("  - CUDA Toolkit: May not be fully installed")
    print("\nTo fix this:")
    print("1. Download and install CUDA Toolkit from:")
    print("   https://developer.nvidia.com/cuda-downloads")
    print("2. Make sure to install the full toolkit (not just drivers)")
    print("3. Restart your terminal after installation")
    print("\nFor now, you can use the CPU version:")
    print("  py src/baseline_cpu_simulation.py")
    print("=" * 70)
    sys.exit(1)

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
GRAVITY = 0.5  # pixels per frame squared
DT = 1.0  # time step per frame
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (100, 150, 255)

# CUDA kernel code for particle update
CUDA_KERNEL = """
__global__ void update_particles(
    float *x, float *y, 
    float *vx, float *vy,
    float gravity, float dt, 
    float width, float height,
    int n
) {
    // Calculate thread index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < n) {
        // Update velocity with gravity
        vy[i] += gravity * dt;
        
        // Update position
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        
        // Boundary collision with damping
        // Left wall
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
            vx[i] *= -0.8f;
        }
        // Right wall
        if (x[i] > width) {
            x[i] = width;
            vx[i] *= -0.8f;
        }
        // Top wall
        if (y[i] < 0.0f) {
            y[i] = 0.0f;
            vy[i] *= -0.8f;
        }
        // Bottom wall
        if (y[i] > height) {
            y[i] = height;
            vy[i] *= -0.8f;
        }
    }
}
"""


class ParticleSystemGPU:
    """GPU-accelerated particle system using PyCUDA."""
    
    def __init__(self, num_particles=1000):
        """
        Initialize GPU particle system.
        
        Args:
            num_particles: Number of particles to simulate
        """
        self.num_particles = num_particles
        
        # Initialize particle properties on CPU (host)
        x_host = np.random.uniform(0, WIDTH, num_particles).astype(np.float32)
        y_host = np.random.uniform(0, HEIGHT // 2, num_particles).astype(np.float32)
        vx_host = np.random.uniform(-2, 2, num_particles).astype(np.float32)
        vy_host = np.random.uniform(0, 2, num_particles).astype(np.float32)
        
        # Allocate GPU memory
        self.x_gpu = cuda.mem_alloc(x_host.nbytes)
        self.y_gpu = cuda.mem_alloc(y_host.nbytes)
        self.vx_gpu = cuda.mem_alloc(vx_host.nbytes)
        self.vy_gpu = cuda.mem_alloc(vy_host.nbytes)
        
        # Copy initial data to GPU
        cuda.memcpy_htod(self.x_gpu, x_host)
        cuda.memcpy_htod(self.y_gpu, y_host)
        cuda.memcpy_htod(self.vx_gpu, vx_host)
        cuda.memcpy_htod(self.vy_gpu, vy_host)
        
        # Host arrays for reading back results
        self.x_host = np.empty_like(x_host)
        self.y_host = np.empty_like(y_host)
        
        # Compile CUDA kernel
        self.mod = SourceModule(CUDA_KERNEL)
        self.update_particles = self.mod.get_function("update_particles")
        
        # Calculate grid and block dimensions
        self.block_size = 256  # Threads per block
        self.grid_size = (num_particles + self.block_size - 1) // self.block_size
    
    def update(self):
        """Update particle positions and velocities on GPU."""
        # Launch CUDA kernel
        self.update_particles(
            self.x_gpu, self.y_gpu, self.vx_gpu, self.vy_gpu,
            np.float32(GRAVITY), np.float32(DT),
            np.float32(WIDTH), np.float32(HEIGHT),
            np.int32(self.num_particles),
            block=(self.block_size, 1, 1),
            grid=(self.grid_size, 1)
        )
    
    def get_positions(self):
        """Copy particle positions from GPU to CPU for rendering."""
        cuda.memcpy_dtoh(self.x_host, self.x_gpu)
        cuda.memcpy_dtoh(self.y_host, self.y_gpu)
        return self.x_host, self.y_host
    
    def cleanup(self):
        """Free GPU memory (optional, will be freed automatically on deletion)."""
        pass


def run_simulation(num_particles=1000, max_frames=None, profile=False):
    """
    Run the GPU-based particle simulation.
    
    Args:
        num_particles: Number of particles to simulate
        max_frames: Maximum number of frames to run (None for infinite)
        profile: If True, measure and print performance metrics
    
    Returns:
        Average time per frame if profiling
    """
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"GPU Particle Simulation (PyCUDA) - {num_particles} particles")
    clock = pygame.time.Clock()
    
    particle_system = ParticleSystemGPU(num_particles)
    
    running = True
    frame_count = 0
    total_update_time = 0.0
    total_transfer_time = 0.0
    
    font = pygame.font.Font(None, 36)
    
    print(f"GPU Info: {cuda.Device(0).name()}")
    print(f"Starting simulation with {num_particles} particles...")
    print("Press ESC or close window to exit")
    
    while running:
        frame_start = time.perf_counter()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update particles on GPU
        update_start = time.perf_counter()
        particle_system.update()
        cuda.Context.synchronize()  # Wait for GPU to finish
        update_end = time.perf_counter()
        
        if profile:
            total_update_time += (update_end - update_start) * 1000  # Convert to ms
        
        # Copy results back to CPU for rendering
        transfer_start = time.perf_counter()
        x_pos, y_pos = particle_system.get_positions()
        transfer_end = time.perf_counter()
        
        if profile:
            total_transfer_time += (transfer_end - transfer_start) * 1000
        
        # Render
        screen.fill(BLACK)
        
        # Draw particles
        for i in range(num_particles):
            px = int(np.clip(x_pos[i], 0, WIDTH - 1))
            py = int(np.clip(y_pos[i], 0, HEIGHT - 1))
            pygame.draw.circle(screen, WHITE, (px, py), 2)
        
        # Display FPS and particle count
        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
        count_text = font.render(f"Particles: {num_particles}", True, WHITE)
        screen.blit(fps_text, (10, 10))
        screen.blit(count_text, (10, 50))
        
        if profile:
            avg_update = total_update_time / (frame_count + 1)
            avg_transfer = total_transfer_time / (frame_count + 1)
            update_text = font.render(f"GPU Update: {avg_update:.2f} ms", True, WHITE)
            transfer_text = font.render(f"Transfer: {avg_transfer:.2f} ms", True, WHITE)
            screen.blit(update_text, (10, 90))
            screen.blit(transfer_text, (10, 130))
        
        pygame.display.flip()
        clock.tick(FPS)
        
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            running = False
        
        frame_end = time.perf_counter()
    
    particle_system.cleanup()
    pygame.quit()
    
    if profile:
        avg_update_time = total_update_time / frame_count if frame_count > 0 else 0
        avg_transfer_time = total_transfer_time / frame_count if frame_count > 0 else 0
        return avg_update_time, avg_transfer_time
    
    return None, None


def benchmark_gpu(num_particles_list=[100, 500, 1000, 5000, 10000, 50000], num_frames=300):
    """
    Benchmark GPU performance for different particle counts.
    
    Args:
        num_particles_list: List of particle counts to test
        num_frames: Number of frames to run for each test
    
    Returns:
        Dictionary mapping particle counts to (update_time, transfer_time) tuples (ms)
    """
    results = {}
    
    print("=" * 60)
    print("GPU Performance Benchmark (PyCUDA)")
    print("=" * 60)
    print(f"GPU: {cuda.Device(0).name()}")
    print(f"Compute Capability: {cuda.Device(0).compute_capability()}")
    print()
    
    for num_particles in num_particles_list:
        print(f"Testing with {num_particles} particles...")
        
        # Initialize system
        particle_system = ParticleSystemGPU(num_particles)
        
        # Warm-up run
        for _ in range(10):
            particle_system.update()
        cuda.Context.synchronize()
        
        # Actual benchmark
        total_update_time = 0.0
        total_transfer_time = 0.0
        
        for frame in range(num_frames):
            # GPU update
            update_start = time.perf_counter()
            particle_system.update()
            cuda.Context.synchronize()
            update_end = time.perf_counter()
            total_update_time += (update_end - update_start) * 1000
            
            # CPU transfer
            transfer_start = time.perf_counter()
            particle_system.get_positions()
            transfer_end = time.perf_counter()
            total_transfer_time += (transfer_end - transfer_start) * 1000
        
        avg_update = total_update_time / num_frames
        avg_transfer = total_transfer_time / num_frames
        results[num_particles] = (avg_update, avg_transfer)
        
        print(f"  Average GPU update time: {avg_update:.4f} ms per frame")
        print(f"  Average transfer time: {avg_transfer:.4f} ms per frame")
        print(f"  Total GPU time: {avg_update + avg_transfer:.4f} ms per frame")
        print(f"  Total time for {num_frames} frames: {total_update_time + total_transfer_time:.2f} ms")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Default: run interactive simulation
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run benchmark mode
        results = benchmark_gpu()
        print("\n" + "=" * 60)
        print("Benchmark Results Summary")
        print("=" * 60)
        for num_particles, (update_time, transfer_time) in results.items():
            total_time = update_time + transfer_time
            print(f"{num_particles:6d} particles: GPU={update_time:8.4f} ms, "
                  f"Transfer={transfer_time:8.4f} ms, Total={total_time:8.4f} ms")
    else:
        # Run interactive visualization
        num_particles = 1000
        if len(sys.argv) > 1:
            try:
                num_particles = int(sys.argv[1])
            except ValueError:
                print(f"Invalid particle count: {sys.argv[1]}. Using default: {num_particles}")
        
        run_simulation(num_particles=num_particles, profile=True)

