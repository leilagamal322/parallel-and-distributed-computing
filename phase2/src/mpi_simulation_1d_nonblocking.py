import numpy as np
import sys
import pygame
import time

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py not found. Install with: pip install mpi4py")
    sys.exit(1)

WIDTH = 1200
HEIGHT = 800
GRAVITY = 0.5
DT = 1.0
DAMPING = 0.8
FPS = 60

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (100, 150, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
PINK = (255, 192, 203)


class Domain1D:
    def __init__(self, comm, width, height):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.width = width
        self.height = height
        self.halo_width = 50.0  # Larger halo for better overlap demonstration
        self._setup()
        self._find_neighbors()

    def _setup(self):
        block_width = self.width / self.size
        self.x_min = self.rank * block_width
        if self.rank < self.size - 1:
            self.x_max = (self.rank + 1) * block_width
        else:
            self.x_max = self.width
        
        self.y_min = 0.0
        self.y_max = self.height
        
        # Extended domain includes halo regions
        self.x_min_ext = max(0.0, self.x_min - self.halo_width)
        self.x_max_ext = min(self.width, self.x_max + self.halo_width)
        self.y_min_ext = 0.0
        self.y_max_ext = self.height
        
        self.block_width = block_width

    def _find_neighbors(self):
        self.left_neighbor = self.rank - 1 if self.rank > 0 else None
        self.right_neighbor = self.rank + 1 if self.rank < self.size - 1 else None

    def in_domain(self, x, y):
        """Check if position is in the actual domain (not halo)"""
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max

    def in_extended(self, x, y):
        """Check if position is in extended domain (includes halo)"""
        return self.x_min_ext <= x < self.x_max_ext and self.y_min_ext <= y < self.y_max_ext

    def in_interior(self, x, y):
        """Check if particle is in interior (not near boundaries where halo exchange is needed)"""
        interior_x_min = self.x_min + self.halo_width
        interior_x_max = self.x_max - self.halo_width
        return interior_x_min <= x < interior_x_max and self.y_min <= y < self.y_max

    def in_left_boundary(self, x, y):
        """Check if particle is in left boundary region (needs left neighbor data)"""
        return self.x_min <= x < self.x_min + self.halo_width and self.y_min <= y < self.y_max

    def in_right_boundary(self, x, y):
        """Check if particle is in right boundary region (needs right neighbor data)"""
        return self.x_max - self.halo_width <= x < self.x_max and self.y_min <= y < self.y_max

    def find_rank(self, x, y):
        dest = int(x / self.block_width)
        return min(max(dest, 0), self.size - 1)


class ParticleSystemMPINonBlocking:
    def __init__(self, num_particles, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.num_total = num_particles
        self.domain = Domain1D(comm, WIDTH, HEIGHT)
        
        # Fixed-size buffers for halo exchange to avoid two-phase communication
        # Estimate max particles in boundary: use generous buffer
        self.max_halo_particles = max(200, num_particles // self.size)
        self.halo_buffer_size = self.max_halo_particles * 4  # 4 floats per particle
        
        # Allocate persistent buffers
        self.send_left_buffer = np.zeros(self.halo_buffer_size + 1, dtype=np.float32)  # +1 for count
        self.send_right_buffer = np.zeros(self.halo_buffer_size + 1, dtype=np.float32)
        self.recv_left_buffer = np.zeros(self.halo_buffer_size + 1, dtype=np.float32)
        self.recv_right_buffer = np.zeros(self.halo_buffer_size + 1, dtype=np.float32)
        
        self._init_particles()

    def _init_particles(self):
        if self.rank == 0:
            x_all = np.random.uniform(0, WIDTH, self.num_total).astype(np.float32)
            y_all = np.random.uniform(0, HEIGHT // 2, self.num_total).astype(np.float32)
            vx_all = np.random.uniform(-2, 2, self.num_total).astype(np.float32)
            vy_all = np.random.uniform(0, 2, self.num_total).astype(np.float32)
        else:
            x_all = np.empty(self.num_total, dtype=np.float32)
            y_all = np.empty(self.num_total, dtype=np.float32)
            vx_all = np.empty(self.num_total, dtype=np.float32)
            vy_all = np.empty(self.num_total, dtype=np.float32)

        self.comm.Bcast([x_all, MPI.FLOAT], root=0)
        self.comm.Bcast([y_all, MPI.FLOAT], root=0)
        self.comm.Bcast([vx_all, MPI.FLOAT], root=0)
        self.comm.Bcast([vy_all, MPI.FLOAT], root=0)

        # Keep only particles in local domain
        mask = np.array([self.domain.in_domain(x_all[i], y_all[i]) for i in range(self.num_total)])
        local_idx = np.where(mask)[0]
        self.num_local = len(local_idx)

        if self.num_local > 0:
            self.x = x_all[local_idx].copy()
            self.y = y_all[local_idx].copy()
            self.vx = vx_all[local_idx].copy()
            self.vy = vy_all[local_idx].copy()
        else:
            self.x = np.array([], dtype=np.float32)
            self.y = np.array([], dtype=np.float32)
            self.vx = np.array([], dtype=np.float32)
            self.vy = np.array([], dtype=np.float32)

    def update(self):
        """
        Classical non-blocking MPI pattern with computation/communication overlap:
        1. Post MPI_Irecv for halo DATA
        2. Post MPI_Isend for halo DATA
        3. Compute interior (OVERLAPS with halo data transfer!)
        4. MPI_Wait
        5. Compute boundaries
        
        Uses fixed-size buffers with count in first element to avoid two-phase exchange.
        All ranks participate in communication, even with 0 particles.
        """
        # Categorize particles BEFORE computation (even if 0 particles)
        interior_indices = np.array([], dtype=np.int64)
        left_boundary_indices = np.array([], dtype=np.int64)
        right_boundary_indices = np.array([], dtype=np.int64)
        
        if self.num_local > 0:
            interior_mask = np.array([self.domain.in_interior(self.x[i], self.y[i]) 
                                      for i in range(self.num_local)])
            left_boundary_mask = np.array([self.domain.in_left_boundary(self.x[i], self.y[i]) 
                                           for i in range(self.num_local)])
            right_boundary_mask = np.array([self.domain.in_right_boundary(self.x[i], self.y[i]) 
                                            for i in range(self.num_local)])
            
            interior_indices = np.where(interior_mask)[0]
            left_boundary_indices = np.where(left_boundary_mask)[0]
            right_boundary_indices = np.where(right_boundary_mask)[0]

        # Prepare send buffers: [count, x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        # Left boundary data
        left_count = len(left_boundary_indices)
        self.send_left_buffer[0] = float(left_count)
        if left_count > 0 and left_count <= self.max_halo_particles:
            for j, idx in enumerate(left_boundary_indices):
                offset = 1 + j * 4
                self.send_left_buffer[offset] = self.x[idx]
                self.send_left_buffer[offset + 1] = self.y[idx]
                self.send_left_buffer[offset + 2] = self.vx[idx]
                self.send_left_buffer[offset + 3] = self.vy[idx]
        
        # Right boundary data
        right_count = len(right_boundary_indices)
        self.send_right_buffer[0] = float(right_count)
        if right_count > 0 and right_count <= self.max_halo_particles:
            for j, idx in enumerate(right_boundary_indices):
                offset = 1 + j * 4
                self.send_right_buffer[offset] = self.x[idx]
                self.send_right_buffer[offset + 1] = self.y[idx]
                self.send_right_buffer[offset + 2] = self.vx[idx]
                self.send_right_buffer[offset + 3] = self.vy[idx]

        # ============================================================
        # STEP 1: Post MPI_Irecv for halo DATA (not just counts!)
        # ============================================================
        recv_requests = []
        
        if self.domain.left_neighbor is not None:
            req = self.comm.Irecv(self.recv_left_buffer, source=self.domain.left_neighbor, tag=100)
            recv_requests.append(('left', req))
        
        if self.domain.right_neighbor is not None:
            req = self.comm.Irecv(self.recv_right_buffer, source=self.domain.right_neighbor, tag=101)
            recv_requests.append(('right', req))

        # ============================================================
        # STEP 2: Post MPI_Isend for halo DATA (not just counts!)
        # ============================================================
        send_requests = []
        
        if self.domain.left_neighbor is not None:
            req = self.comm.Isend(self.send_left_buffer, dest=self.domain.left_neighbor, tag=101)
            send_requests.append(('left', req))
        
        if self.domain.right_neighbor is not None:
            req = self.comm.Isend(self.send_right_buffer, dest=self.domain.right_neighbor, tag=100)
            send_requests.append(('right', req))

        # ============================================================
        # STEP 3: Compute interior (OVERLAPS with halo data transfer!)
        # ============================================================
        # This is the key: interior computation happens WHILE MPI is transferring
        # the actual halo particle data, not just counts
        if len(interior_indices) > 0:
            self._update_particles(interior_indices)

        # ============================================================
        # STEP 4: MPI_Wait (wait for halo DATA to arrive)
        # ============================================================
        # Wait for all receives to complete
        for name, req in recv_requests:
            req.Wait()
        
        # Wait for all sends to complete
        for name, req in send_requests:
            req.Wait()

        # ============================================================
        # STEP 5: Compute boundaries (now that halo data has arrived)
        # ============================================================
        # Update boundary particles
        if len(left_boundary_indices) > 0:
            self._update_particles(left_boundary_indices)
        
        if len(right_boundary_indices) > 0:
            self._update_particles(right_boundary_indices)
        
        # In a real application with particle-particle interactions,
        # we would use recv_left_buffer and recv_right_buffer here to compute
        # forces between boundary particles and ghost particles from neighbors
        
        # Handle particle migration after all updates
        self._migrate_particles()

    def _update_particles(self, indices):
        """Apply physics update to specified particle indices"""
        if len(indices) == 0:
            return
        
        # Apply gravity
        self.vy[indices] += GRAVITY * DT
        
        # Update positions
        self.x[indices] += self.vx[indices] * DT
        self.y[indices] += self.vy[indices] * DT
        
        # Apply boundary conditions
        self._apply_boundaries(indices)

    def _apply_boundaries(self, indices):
        """Apply boundary conditions to specific particle indices"""
        if len(indices) == 0:
            return

        # X boundaries
        mask = self.x[indices] < 0.0
        self.x[indices[mask]] = 0.0
        self.vx[indices[mask]] *= -DAMPING

        mask = self.x[indices] >= WIDTH
        self.x[indices[mask]] = WIDTH - 0.1
        self.vx[indices[mask]] *= -DAMPING

        # Y boundaries
        mask = self.y[indices] < 0.0
        self.y[indices[mask]] = 0.0
        self.vy[indices[mask]] *= -DAMPING

        mask = self.y[indices] >= HEIGHT
        self.y[indices[mask]] = HEIGHT - 0.1
        self.vy[indices[mask]] *= -DAMPING

    def _migrate_particles(self):
        """
        Handle particles that have moved to other domains.
        NOTE: All ranks must participate in communication, even with 0 particles!
        """
        # Identify particles that need to migrate
        particles_to_send = {}
        
        if self.num_local > 0:
            for i in range(self.num_local):
                if not self.domain.in_domain(self.x[i], self.y[i]):
                    dest = self.domain.find_rank(self.x[i], self.y[i])
                    if dest != self.rank:
                        if dest not in particles_to_send:
                            particles_to_send[dest] = []
                        particles_to_send[dest].append(i)
        
        # Exchange counts (using fixed-size buffers to avoid dynamic allocation)
        neighbors = []
        if self.domain.left_neighbor is not None:
            neighbors.append(self.domain.left_neighbor)
        if self.domain.right_neighbor is not None:
            neighbors.append(self.domain.right_neighbor)
        
        send_counts = {}
        for neighbor in neighbors:
            count = len(particles_to_send.get(neighbor, []))
            send_counts[neighbor] = np.array([count], dtype=np.int32)
        
        recv_counts = {}
        for neighbor in neighbors:
            recv_counts[neighbor] = np.zeros(1, dtype=np.int32)
        
        # Non-blocking exchange counts
        count_reqs = []
        for neighbor in neighbors:
            req = self.comm.Irecv(recv_counts[neighbor], source=neighbor, tag=300)
            count_reqs.append(req)
        
        for neighbor in neighbors:
            req = self.comm.Isend(send_counts[neighbor], dest=neighbor, tag=300)
            count_reqs.append(req)
        
        MPI.Request.Waitall(count_reqs)
        
        # Exchange particle data
        recv_buffers = {}
        data_reqs = []
        
        for neighbor in neighbors:
            count = int(recv_counts[neighbor][0])
            if count > 0:
                recv_buffers[neighbor] = np.zeros(count * 4, dtype=np.float32)
                req = self.comm.Irecv(recv_buffers[neighbor], source=neighbor, tag=400)
                data_reqs.append(req)
        
        send_buffers = {}
        for dest, indices in particles_to_send.items():
            if len(indices) > 0:
                send_buffers[dest] = np.zeros(len(indices) * 4, dtype=np.float32)
                for j, idx in enumerate(indices):
                    offset = j * 4
                    send_buffers[dest][offset] = self.x[idx]
                    send_buffers[dest][offset + 1] = self.y[idx]
                    send_buffers[dest][offset + 2] = self.vx[idx]
                    send_buffers[dest][offset + 3] = self.vy[idx]
                
                req = self.comm.Isend(send_buffers[dest], dest=dest, tag=400)
                data_reqs.append(req)
        
        MPI.Request.Waitall(data_reqs)
        
        # Remove migrated particles
        particles_to_remove = set()
        for indices in particles_to_send.values():
            particles_to_remove.update(indices)
        
        if particles_to_remove:
            keep_mask = np.ones(self.num_local, dtype=bool)
            for idx in particles_to_remove:
                keep_mask[idx] = False
            
            self.x = self.x[keep_mask]
            self.y = self.y[keep_mask]
            self.vx = self.vx[keep_mask]
            self.vy = self.vy[keep_mask]
            self.num_local = len(self.x)
        
        # Add received particles
        for neighbor, data in recv_buffers.items():
            count = len(data) // 4
            if count > 0:
                new_x = data[0::4]
                new_y = data[1::4]
                new_vx = data[2::4]
                new_vy = data[3::4]
                
                self.x = np.concatenate([self.x, new_x])
                self.y = np.concatenate([self.y, new_y])
                self.vx = np.concatenate([self.vx, new_vx])
                self.vy = np.concatenate([self.vy, new_vy])
                self.num_local = len(self.x)

    def get_positions(self):
        """Gather all particle positions to rank 0 for visualization"""
        local_count = np.array([self.num_local], dtype=np.int32)
        all_counts = np.empty(self.size, dtype=np.int32)
        self.comm.Allgather([local_count, MPI.INT], [all_counts, MPI.INT])

        if self.rank == 0:
            x_all = []
            y_all = []
            
            if self.num_local > 0:
                x_all.append(self.x.copy())
                y_all.append(self.y.copy())
            
            for r in range(1, self.size):
                if all_counts[r] > 0:
                    rx = np.empty(all_counts[r], dtype=np.float32)
                    ry = np.empty(all_counts[r], dtype=np.float32)
                    self.comm.Recv([rx, MPI.FLOAT], source=r, tag=2)
                    self.comm.Recv([ry, MPI.FLOAT], source=r, tag=3)
                    x_all.append(rx)
                    y_all.append(ry)
            
            if x_all:
                return np.concatenate(x_all), np.concatenate(y_all)
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        else:
            if self.num_local > 0:
                self.comm.Send([self.x, MPI.FLOAT], dest=0, tag=2)
                self.comm.Send([self.y, MPI.FLOAT], dest=0, tag=3)
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    def get_count(self):
        return self.num_local


def run_simulation(num_particles=1000, num_frames=None, visualize=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"1D Block Split (Non-Blocking MPI) - Particles: {num_particles}, Ranks: {size}")
        print("Classical non-blocking pattern with TRUE overlap:")
        print("  1. Post MPI_Irecv for halo DATA")
        print("  2. Post MPI_Isend for halo DATA")
        print("  3. Compute interior (overlaps with DATA transfer!)")
        print("  4. MPI_Wait")
        print("  5. Compute boundaries")
        print()

    if visualize and rank == 0:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"MPI 1D Non-Blocking - {num_particles} particles, {size} ranks")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)
        rank_colors = [WHITE, BLUE, GREEN, RED, YELLOW, CYAN, MAGENTA, ORANGE, PURPLE, PINK]

    ps = ParticleSystemMPINonBlocking(num_particles, comm)
    comm.Barrier()

    running = True
    frame_count = 0
    total_time = 0.0

    while running:
        if rank == 0 and visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        running_flag = np.array([1 if running else 0], dtype=np.int32)
        comm.Bcast([running_flag, MPI.INT], root=0)
        if running_flag[0] == 0:
            break

        start = time.perf_counter()
        ps.update()
        comm.Barrier()
        end = time.perf_counter()

        if rank == 0:
            frame_time = (end - start) * 1000
            total_time += frame_time

            if visualize:
                x_all, y_all = ps.get_positions()
                
                screen.fill(BLACK)
                
                # Draw domain boundaries
                if size <= len(rank_colors):
                    for r in range(size):
                        x_min = r * (WIDTH / size)
                        x_max = (r + 1) * (WIDTH / size) if r < size - 1 else WIDTH
                        color = rank_colors[r % len(rank_colors)]
                        pygame.draw.line(screen, color, (x_min, 0), (x_min, HEIGHT), 2)
                        if r == size - 1:
                            pygame.draw.line(screen, color, (x_max, 0), (x_max, HEIGHT), 2)
                
                # Draw particles
                if len(x_all) > 0:
                    for i in range(len(x_all)):
                        px = int(np.clip(x_all[i], 0, WIDTH - 1))
                        py = int(np.clip(y_all[i], 0, HEIGHT - 1))
                        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                            pygame.draw.circle(screen, WHITE, (px, py), 2)
                
                # Draw stats
                fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
                count_text = font.render(f"Particles: {len(x_all)}", True, WHITE)
                rank_text = font.render(f"Ranks: {size}", True, WHITE)
                time_text = font.render(f"Update: {frame_time:.2f} ms", True, WHITE)
                
                screen.blit(fps_text, (10, 10))
                screen.blit(count_text, (10, 50))
                screen.blit(rank_text, (10, 90))
                screen.blit(time_text, (10, 130))
                
                pygame.display.flip()
                clock.tick(FPS)
        else:
            if visualize:
                ps.get_positions()

        frame_count += 1
        if num_frames and frame_count >= num_frames:
            if rank == 0:
                running = False

    if rank == 0:
        if visualize:
            pygame.quit()
        avg = total_time / frame_count if frame_count > 0 else 0
        print(f"Avg time: {avg:.4f} ms/frame")

    return ps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MPI Particle Simulation with Non-Blocking Communication')
    parser.add_argument('--particles', type=int, default=1000, help='Number of particles')
    parser.add_argument('--frames', type=int, default=None, help='Limit number of frames')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    run_simulation(args.particles, args.frames, visualize=not args.no_viz)