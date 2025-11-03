"""
Baseline CPU Sequential Particle Simulation
===========================================
This is the sequential (single-threaded) version using NumPy and Pygame.
Used as a baseline for performance comparison with the GPU version.
"""

import numpy as np
import pygame
import time

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


class ParticleSystemCPU:
    """Sequential CPU-based particle system."""
    
    def __init__(self, num_particles=1000):
        """
        Initialize particle system.
        
        Args:
            num_particles: Number of particles to simulate
        """
        self.num_particles = num_particles
        
        # Initialize particle properties
        # Position: random initial positions
        self.x = np.random.uniform(0, WIDTH, num_particles).astype(np.float32)
        self.y = np.random.uniform(0, HEIGHT // 2, num_particles).astype(np.float32)
        
        # Velocity: random initial velocities
        self.vx = np.random.uniform(-2, 2, num_particles).astype(np.float32)
        self.vy = np.random.uniform(0, 2, num_particles).astype(np.float32)
        
        # Acceleration (gravity)
        self.ax = np.zeros(num_particles, dtype=np.float32)
        self.ay = np.full(num_particles, GRAVITY, dtype=np.float32)
        
    def update(self):
        """Update particle positions and velocities (sequential CPU computation)."""
        # Update velocities: v = v + a * dt
        self.vx += self.ax * DT
        self.vy += self.ay * DT
        
        # Update positions: p = p + v * dt
        self.x += self.vx * DT
        self.y += self.vy * DT
        
        # Boundary collision (bounce off walls)
        # Left and right walls
        mask_left = self.x < 0
        mask_right = self.x > WIDTH
        self.x[mask_left] = 0
        self.x[mask_right] = WIDTH
        self.vx[mask_left] *= -0.8  # Damping on bounce
        self.vx[mask_right] *= -0.8
        
        # Top and bottom walls
        mask_top = self.y < 0
        mask_bottom = self.y > HEIGHT
        self.y[mask_top] = 0
        self.y[mask_bottom] = HEIGHT
        self.vy[mask_top] *= -0.8
        self.vy[mask_bottom] *= -0.8
    
    def get_positions(self):
        """Get current particle positions for rendering."""
        return self.x, self.y


def run_simulation(num_particles=1000, max_frames=None, profile=False):
    """
    Run the CPU-based particle simulation.
    
    Args:
        num_particles: Number of particles to simulate
        max_frames: Maximum number of frames to run (None for infinite)
        profile: If True, measure and print performance metrics
    
    Returns:
        Average time per frame if profiling
    """
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"CPU Particle Simulation - {num_particles} particles")
    clock = pygame.time.Clock()
    
    particle_system = ParticleSystemCPU(num_particles)
    
    running = True
    frame_count = 0
    total_update_time = 0.0
    
    font = pygame.font.Font(None, 36)
    
    while running:
        frame_start = time.perf_counter()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update particles (CPU computation)
        update_start = time.perf_counter()
        particle_system.update()
        update_end = time.perf_counter()
        
        if profile:
            total_update_time += (update_end - update_start) * 1000  # Convert to ms
        
        # Render
        screen.fill(BLACK)
        
        # Draw particles
        x_pos, y_pos = particle_system.get_positions()
        for i in range(num_particles):
            # Clamp positions to screen bounds for rendering
            px = int(np.clip(x_pos[i], 0, WIDTH - 1))
            py = int(np.clip(y_pos[i], 0, HEIGHT - 1))
            pygame.draw.circle(screen, WHITE, (px, py), 2)
        
        # Display FPS and particle count
        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
        count_text = font.render(f"Particles: {num_particles}", True, WHITE)
        screen.blit(fps_text, (10, 10))
        screen.blit(count_text, (10, 50))
        
        if profile:
            avg_time = total_update_time / (frame_count + 1)
            time_text = font.render(f"Avg Update: {avg_time:.2f} ms", True, WHITE)
            screen.blit(time_text, (10, 90))
        
        pygame.display.flip()
        clock.tick(FPS)
        
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            running = False
        
        frame_end = time.perf_counter()
    
    pygame.quit()
    
    if profile:
        avg_update_time = total_update_time / frame_count if frame_count > 0 else 0
        return avg_update_time
    
    return None


def benchmark_cpu(num_particles_list=[100, 500, 1000, 5000, 10000], num_frames=300):
    """
    Benchmark CPU performance for different particle counts.
    
    Args:
        num_particles_list: List of particle counts to test
        num_frames: Number of frames to run for each test
    
    Returns:
        Dictionary mapping particle counts to average update times (ms)
    """
    results = {}
    
    print("=" * 60)
    print("CPU Performance Benchmark")
    print("=" * 60)
    
    for num_particles in num_particles_list:
        print(f"\nTesting with {num_particles} particles...")
        
        # Initialize system
        particle_system = ParticleSystemCPU(num_particles)
        
        # Warm-up run
        for _ in range(10):
            particle_system.update()
        
        # Actual benchmark
        total_time = 0.0
        for frame in range(num_frames):
            start = time.perf_counter()
            particle_system.update()
            end = time.perf_counter()
            total_time += (end - start) * 1000  # Convert to ms
        
        avg_time = total_time / num_frames
        results[num_particles] = avg_time
        
        print(f"  Average update time: {avg_time:.4f} ms per frame")
        print(f"  Total time for {num_frames} frames: {total_time:.2f} ms")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Default: run interactive simulation
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run benchmark mode
        results = benchmark_cpu()
        print("\n" + "=" * 60)
        print("Benchmark Results Summary")
        print("=" * 60)
        for num_particles, avg_time in results.items():
            print(f"{num_particles:6d} particles: {avg_time:8.4f} ms/frame")
    else:
        # Run interactive visualization
        num_particles = 1000
        if len(sys.argv) > 1:
            try:
                num_particles = int(sys.argv[1])
            except ValueError:
                print(f"Invalid particle count: {sys.argv[1]}. Using default: {num_particles}")
        
        print(f"Starting CPU simulation with {num_particles} particles...")
        print("Press ESC or close window to exit")
        run_simulation(num_particles=num_particles, profile=True)