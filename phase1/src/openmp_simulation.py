"""
OpenMP-Accelerated Particle Simulation
=======================================
This version uses OpenMP (via C++ extension) for shared-memory parallel processing.
Requires compilation of the C++ extension with OpenMP support.
Falls back to Python multiprocessing if C++ extension is not available.
"""

import numpy as np
import pygame
import time
import os
import sys
import ctypes
from ctypes import c_int, c_float, POINTER, Structure, c_void_p
from concurrent.futures import ThreadPoolExecutor
import threading

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

# Try to load the OpenMP shared library
_openmp_lib = None
_openmp_available = False

def _load_openmp_library():
    """Load the compiled OpenMP shared library."""
    global _openmp_lib, _openmp_available
    
    if _openmp_lib is not None:
        return _openmp_available
    
    # Try different library names and paths
    lib_names = []
    
    if sys.platform == 'win32':
        lib_names = ['particle_simulation_openmp.dll', 'libparticle_simulation_openmp.dll']
    elif sys.platform == 'darwin':
        lib_names = ['libparticle_simulation_openmp.dylib', 'particle_simulation_openmp.dylib']
    else:
        lib_names = ['libparticle_simulation_openmp.so', 'particle_simulation_openmp.so']
    
    # Search in current directory and src directory
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd(),
    ]
    
    for path in search_paths:
        for lib_name in lib_names:
            lib_path = os.path.join(path, lib_name)
            if os.path.exists(lib_path):
                try:
                    _openmp_lib = ctypes.CDLL(lib_path)
                    _setup_openmp_functions(_openmp_lib)
                    _openmp_available = True
                    print(f"Successfully loaded OpenMP library: {lib_path}")
                    return True
                except OSError as e:
                    print(f"Failed to load {lib_path}: {e}")
                    continue
    
    # Library not found, will use Python fallback
    # Don't print warning - fallback will be used automatically
    return False

def _setup_openmp_functions(lib):
    """Set up function signatures for the OpenMP library."""
    # Create particle system
    lib.create_particle_system.argtypes = [c_int, c_float, c_float]
    lib.create_particle_system.restype = c_void_p
    
    # Update particles
    lib.update_particles.argtypes = [c_void_p]
    lib.update_particles.restype = None
    
    # Get positions
    lib.get_positions.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]
    lib.get_positions.restype = None
    
    # Free particle system
    lib.free_particle_system.argtypes = [c_void_p]
    lib.free_particle_system.restype = None
    
    # Thread management
    lib.get_num_threads.argtypes = []
    lib.get_num_threads.restype = c_int
    
    lib.set_num_threads.argtypes = [c_int]
    lib.set_num_threads.restype = None


# Python fallback implementation using threading and NumPy
def _update_particle_chunk(start_idx, end_idx, x, y, vx, vy):
    """Update a chunk of particles (for threading fallback)."""
    # Use NumPy vectorized operations for better performance
    # Update velocities
    vy[start_idx:end_idx] += GRAVITY * DT
    
    # Update positions
    x[start_idx:end_idx] += vx[start_idx:end_idx] * DT
    y[start_idx:end_idx] += vy[start_idx:end_idx] * DT
    
    # Boundary collisions with damping
    mask_left = x[start_idx:end_idx] < 0.0
    mask_right = x[start_idx:end_idx] > WIDTH
    mask_top = y[start_idx:end_idx] < 0.0
    mask_bottom = y[start_idx:end_idx] > HEIGHT
    
    x_slice = x[start_idx:end_idx]
    y_slice = y[start_idx:end_idx]
    vx_slice = vx[start_idx:end_idx]
    vy_slice = vy[start_idx:end_idx]
    
    x_slice[mask_left] = 0.0
    x_slice[mask_right] = WIDTH
    vx_slice[mask_left] *= -0.8
    vx_slice[mask_right] *= -0.8
    
    y_slice[mask_top] = 0.0
    y_slice[mask_bottom] = HEIGHT
    vy_slice[mask_top] *= -0.8
    vy_slice[mask_bottom] *= -0.8


class ParticleSystemOpenMPPython:
    """Python fallback implementation using threading."""
    
    def __init__(self, num_particles=1000, num_threads=None):
        """Initialize Python-based parallel particle system."""
        import multiprocessing
        self.num_particles = num_particles
        
        # Determine number of threads
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        self.num_threads = num_threads
        
        # Initialize particle properties
        self.x = np.random.uniform(0, WIDTH, num_particles).astype(np.float32)
        self.y = np.random.uniform(0, HEIGHT // 2, num_particles).astype(np.float32)
        self.vx = np.random.uniform(-2, 2, num_particles).astype(np.float32)
        self.vy = np.random.uniform(0, 2, num_particles).astype(np.float32)
        
        # Create thread pool
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        
        # Calculate chunk sizes
        chunk_size = num_particles // num_threads
        self.chunks = []
        for i in range(num_threads):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_threads - 1 else num_particles
            self.chunks.append((start, end))
        
        print(f"Using Python threading fallback with {self.num_threads} threads")
    
    def update(self):
        """Update particles using threading."""
        # Submit all chunks to thread pool
        futures = []
        for start_idx, end_idx in self.chunks:
            future = self.executor.submit(
                _update_particle_chunk,
                start_idx, end_idx,
                self.x, self.y, self.vx, self.vy
            )
            futures.append(future)
        
        # Wait for all threads to complete
        for future in futures:
            future.result()
    
    def get_positions(self):
        """Get current particle positions for rendering."""
        return self.x.copy(), self.y.copy()
    
    def cleanup(self):
        """Clean up thread pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class ParticleSystemOpenMP:
    """OpenMP-accelerated particle system using C++ extension."""
    
    def __init__(self, num_particles=1000, num_threads=None):
        """
        Initialize OpenMP particle system.
        
        Args:
            num_particles: Number of particles to simulate
            num_threads: Number of OpenMP threads (None = use default)
        """
        # Try to load C++ OpenMP library, fall back to Python if not available
        use_cpp = False
        if _openmp_available or _load_openmp_library():
            use_cpp = True
        
        if not use_cpp:
            # Use Python fallback
            self._use_python = True
            self._python_impl = ParticleSystemOpenMPPython(num_particles, num_threads)
            self.num_particles = num_particles
            self.num_threads = self._python_impl.num_threads
            return
        
        self._use_python = False
        self.num_particles = num_particles
        
        # Set number of threads if specified
        if num_threads is not None:
            _openmp_lib.set_num_threads(num_threads)
        
        # Create particle system in C++
        self._ps_ptr = _openmp_lib.create_particle_system(
            c_int(num_particles),
            c_float(WIDTH),
            c_float(HEIGHT)
        )
        
        if not self._ps_ptr:
            raise RuntimeError("Failed to create particle system")
        
        # Allocate arrays for position retrieval
        self._x_array = (c_float * num_particles)()
        self._y_array = (c_float * num_particles)()
        
        # Get number of threads being used
        self.num_threads = _openmp_lib.get_num_threads()
        print(f"OpenMP using {self.num_threads} threads")
    
    def update(self):
        """Update particle positions and velocities using OpenMP."""
        if self._use_python:
            self._python_impl.update()
            return
        
        if not self._ps_ptr:
            raise RuntimeError("Particle system not initialized")
        _openmp_lib.update_particles(self._ps_ptr)
    
    def get_positions(self):
        """Get current particle positions for rendering."""
        if self._use_python:
            return self._python_impl.get_positions()
        
        if not self._ps_ptr:
            raise RuntimeError("Particle system not initialized")
        
        _openmp_lib.get_positions(
            self._ps_ptr,
            self._x_array,
            self._y_array
        )
        
        # Convert to numpy arrays
        x_pos = np.array(self._x_array, dtype=np.float32)
        y_pos = np.array(self._y_array, dtype=np.float32)
        
        return x_pos, y_pos
    
    def cleanup(self):
        """Free C++ particle system memory or Python resources."""
        if self._use_python:
            if hasattr(self, '_python_impl'):
                self._python_impl.cleanup()
            return
        
        if self._ps_ptr:
            _openmp_lib.free_particle_system(self._ps_ptr)
            self._ps_ptr = None


def run_simulation(num_particles=1000, max_frames=None, profile=False, num_threads=None):
    """
    Run the OpenMP-based particle simulation.
    
    Args:
        num_particles: Number of particles to simulate
        max_frames: Maximum number of frames to run (None for infinite)
        profile: If True, measure and print performance metrics
        num_threads: Number of OpenMP threads (None = use default)
    
    Returns:
        Average time per frame if profiling
    """
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"OpenMP Particle Simulation - {num_particles} particles")
    clock = pygame.time.Clock()
    
    try:
        particle_system = ParticleSystemOpenMP(num_particles, num_threads)
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
        return None
    
    running = True
    frame_count = 0
    total_update_time = 0.0
    
    font = pygame.font.Font(None, 36)
    
    print(f"Starting OpenMP simulation with {num_particles} particles...")
    print(f"Using {particle_system.num_threads} OpenMP threads")
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
        
        # Update particles (OpenMP computation)
        update_start = time.perf_counter()
        particle_system.update()
        update_end = time.perf_counter()
        
        if profile:
            total_update_time += (update_end - update_start) * 1000  # Convert to ms
        
        # Get positions for rendering
        x_pos, y_pos = particle_system.get_positions()
        
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
        threads_text = font.render(f"Threads: {particle_system.num_threads}", True, WHITE)
        screen.blit(fps_text, (10, 10))
        screen.blit(count_text, (10, 50))
        screen.blit(threads_text, (10, 90))
        
        if profile:
            avg_time = total_update_time / (frame_count + 1)
            time_text = font.render(f"Avg Update: {avg_time:.2f} ms", True, WHITE)
            screen.blit(time_text, (10, 130))
        
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
        return avg_update_time
    
    return None


def benchmark_openmp(num_particles_list=[100, 500, 1000, 5000, 10000], num_frames=300, num_threads=None):
    """
    Benchmark OpenMP performance for different particle counts.
    
    Args:
        num_particles_list: List of particle counts to test
        num_frames: Number of frames to run for each test
        num_threads: Number of OpenMP threads (None = use default)
    
    Returns:
        Dictionary mapping particle counts to average update times (ms)
    """
    results = {}
    
    print("=" * 60)
    print("OpenMP Performance Benchmark")
    print("=" * 60)
    
    # Try to load library, but allow Python fallback
    use_cpp = False
    if _openmp_available or _load_openmp_library():
        use_cpp = True
        if num_threads is not None:
            _openmp_lib.set_num_threads(num_threads)
            print(f"Using {num_threads} threads (C++ OpenMP)")
        else:
            num_threads = _openmp_lib.get_num_threads()
            print(f"Using {num_threads} threads (C++ OpenMP, default)")
    else:
        # Will use Python fallback
        import multiprocessing
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        print(f"Using {num_threads} threads (Python threading fallback)")
    print()
    
    for num_particles in num_particles_list:
        print(f"Testing with {num_particles} particles...")
        
        try:
            # Initialize system
            particle_system = ParticleSystemOpenMP(num_particles, num_threads)
            
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
            
            particle_system.cleanup()
            
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            results[num_particles] = None
    
    return results


if __name__ == "__main__":
    import sys
    
    # Default: run interactive simulation
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run benchmark mode
        results = benchmark_openmp()
        if results:
            print("\n" + "=" * 60)
            print("Benchmark Results Summary")
            print("=" * 60)
            for num_particles, avg_time in results.items():
                if avg_time is not None:
                    print(f"{num_particles:6d} particles: {avg_time:8.4f} ms/frame")
    else:
        # Run interactive visualization
        num_particles = 1000
        num_threads = None
        
        if len(sys.argv) > 1:
            try:
                num_particles = int(sys.argv[1])
            except ValueError:
                print(f"Invalid particle count: {sys.argv[1]}. Using default: {num_particles}")
        
        if len(sys.argv) > 2:
            try:
                num_threads = int(sys.argv[2])
            except ValueError:
                print(f"Invalid thread count: {sys.argv[2]}. Using default")
        
        print(f"Starting OpenMP simulation with {num_particles} particles...")
        print("Press ESC or close window to exit")
        run_simulation(num_particles=num_particles, profile=True, num_threads=num_threads)

