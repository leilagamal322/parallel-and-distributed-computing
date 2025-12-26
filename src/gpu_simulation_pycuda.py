# gpu_simulation_pycuda.py
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Simulation constants
WIDTH, HEIGHT = 1200, 800
GRAVITY = 0.5
DT = 1.0

CUDA_KERNEL = """
__global__ void update_particles(
    float *x, float *y, 
    float *vx, float *vy,
    float gravity, float dt, 
    float width, float height,
    int n
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    vy[i] += gravity * dt;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;

    // Boundary collisions with damping
    if (x[i] < 0.0f) {
        x[i] = 0.0f;
        vx[i] *= -0.8f;
    }
    if (x[i] > width) {
        x[i] = width;
        vx[i] *= -0.8f;
    }
    if (y[i] < 0.0f) {
        y[i] = 0.0f;
        vy[i] *= -0.8f;
    }
    if (y[i] > height) {
        y[i] = height;
        vy[i] *= -0.8f;
    }
}
"""

class ParticleSystemGPU:
    def __init__(self, num_particles=1000):
        self.num_particles = num_particles

        # Host initialization
        x_host = np.random.uniform(0, WIDTH, num_particles).astype(np.float32)
        y_host = np.random.uniform(0, HEIGHT / 2, num_particles).astype(np.float32)
        vx_host = np.random.uniform(-2, 2, num_particles).astype(np.float32)
        vy_host = np.random.uniform(0, 2, num_particles).astype(np.float32)

        # Allocate GPU memory
        self.x_gpu = cuda.mem_alloc(x_host.nbytes)
        self.y_gpu = cuda.mem_alloc(y_host.nbytes)
        self.vx_gpu = cuda.mem_alloc(vx_host.nbytes)
        self.vy_gpu = cuda.mem_alloc(vy_host.nbytes)

        cuda.memcpy_htod(self.x_gpu, x_host)
        cuda.memcpy_htod(self.y_gpu, y_host)
        cuda.memcpy_htod(self.vx_gpu, vx_host)
        cuda.memcpy_htod(self.vy_gpu, vy_host)

        self.x_host = np.empty_like(x_host)
        self.y_host = np.empty_like(y_host)

        self.mod = SourceModule(CUDA_KERNEL)
        self.kernel = self.mod.get_function("update_particles")

        self.block_size = 256
        self.grid_size = (num_particles + self.block_size - 1) // self.block_size

    def update(self):
        """Run ONE physics step on the GPU"""
        self.kernel(
            self.x_gpu, self.y_gpu,
            self.vx_gpu, self.vy_gpu,
            np.float32(GRAVITY), np.float32(DT),
            np.float32(WIDTH), np.float32(HEIGHT),
            np.int32(self.num_particles),
            block=(self.block_size, 1, 1),
            grid=(self.grid_size, 1)
        )

    def step(self, frames=1):
        """Run multiple steps (used by gRPC later)"""
        for _ in range(frames):
            self.update()

    def get_positions(self):
        """Copy positions back to CPU"""
        cuda.memcpy_dtoh(self.x_host, self.x_gpu)
        cuda.memcpy_dtoh(self.y_host, self.y_gpu)
        return self.x_host, self.y_host
