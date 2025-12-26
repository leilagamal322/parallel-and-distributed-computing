# mock_gpu_simulation.py
# Mock GPU simulation for testing resilience without PyCUDA
# This allows testing the resilience features without requiring CUDA/GPU

import numpy as np
import time

WIDTH, HEIGHT = 1200, 800
GRAVITY = 0.5
DT = 1.0

class ParticleSystemGPU:
    """
    Mock GPU simulation that mimics the interface of the real GPU simulation
    but runs on CPU. Used for testing resilience features without requiring CUDA.
    """
    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        
        # Initialize particle positions and velocities
        self.x = np.random.rand(num_particles).astype(np.float32) * WIDTH
        self.y = np.random.rand(num_particles).astype(np.float32) * HEIGHT
        self.vx = (np.random.rand(num_particles).astype(np.float32) - 0.5) * 10
        self.vy = (np.random.rand(num_particles).astype(np.float32) - 0.5) * 10
        
        print(f"[MOCK GPU] Initialized {num_particles} particles (CPU simulation)")
    
    def step(self, steps=1):
        """
        Simulate particle physics for the specified number of steps
        """
        for _ in range(steps):
            # Apply gravity
            self.vy += GRAVITY * DT
            
            # Update positions
            self.x += self.vx * DT
            self.y += self.vy * DT
            
            # Boundary collisions with damping
            # Left and right walls
            mask = self.x < 0
            self.x[mask] = 0
            self.vx[mask] *= -0.8
            
            mask = self.x > WIDTH
            self.x[mask] = WIDTH
            self.vx[mask] *= -0.8
            
            # Top and bottom walls
            mask = self.y < 0
            self.y[mask] = 0
            self.vy[mask] *= -0.8
            
            mask = self.y > HEIGHT
            self.y[mask] = HEIGHT
            self.vy[mask] *= -0.8
        
        # Simulate some computation time (like GPU would take)
        time.sleep(0.001 * steps)  # 1ms per step
    
    def get_positions(self):
        """Get current particle positions"""
        return self.x, self.y
    
    def get_velocities(self):
        """Get current particle velocities"""
        return self.vx, self.vy


if __name__ == "__main__":
    # Test the mock simulation
    print("Testing Mock GPU Simulation...")
    sim = ParticleSystemGPU(num_particles=1000)
    
    print("Running 10 simulation steps...")
    start = time.time()
    sim.step(10)
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed*1000:.2f}ms")
    print("Mock GPU simulation working correctly!")

