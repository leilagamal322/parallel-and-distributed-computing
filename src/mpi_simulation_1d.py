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
        self.halo_width = 10.0
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
        self.x_min_ext = max(0.0, self.x_min - self.halo_width)
        self.x_max_ext = min(self.width, self.x_max + self.halo_width)
        self.y_min_ext = 0.0
        self.y_max_ext = self.height
        self.block_width = block_width

    def _find_neighbors(self):
        self.neighbors = []
        if self.rank > 0:
            self.neighbors.append(self.rank - 1)
        if self.rank < self.size - 1:
            self.neighbors.append(self.rank + 1)

    def in_domain(self, x, y):
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max

    def in_extended(self, x, y):
        return self.x_min_ext <= x < self.x_max_ext and self.y_min_ext <= y < self.y_max_ext

    def find_rank(self, x, y):
        dest = int(x / self.block_width)
        return min(dest, self.size - 1)


class ParticleSystemMPI:
    def __init__(self, num_particles, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.num_total = num_particles
        self.domain = Domain1D(comm, WIDTH, HEIGHT)
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

        mask = np.array([self.domain.in_domain(x_all[i], y_all[i]) 
                        for i in range(self.num_total)])
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
        if self.num_local == 0:
            return

        self.vy += GRAVITY * DT
        self.x += self.vx * DT
        self.y += self.vy * DT
        self._boundaries()
        self._exchange()

    def _boundaries(self):
        if self.num_local == 0:
            return

        mask = self.x < 0.0
        self.x[mask] = 0.0
        self.vx[mask] *= -DAMPING

        mask = self.x >= WIDTH
        self.x[mask] = WIDTH - 0.1
        self.vx[mask] *= -DAMPING

        mask = self.y < 0.0
        self.y[mask] = 0.0
        self.vy[mask] *= -DAMPING

        mask = self.y >= HEIGHT
        self.y[mask] = HEIGHT - 0.1
        self.vy[mask] *= -DAMPING

    def _exchange(self):
        if self.num_local == 0:
            return

        in_local = np.array([self.domain.in_domain(self.x[i], self.y[i]) 
                            for i in range(self.num_local)])
        to_send = np.where(~in_local)[0]

        if len(to_send) == 0:
            return

        by_rank = {}
        for idx in to_send:
            dest = self.domain.find_rank(self.x[idx], self.y[idx])
            if dest != self.rank:
                if dest not in by_rank:
                    by_rank[dest] = []
                by_rank[dest].append(idx)

        send_reqs = []
        for dest, indices in by_rank.items():
            data = {
                'x': self.x[indices].copy(),
                'y': self.y[indices].copy(),
                'vx': self.vx[indices].copy(),
                'vy': self.vy[indices].copy()
            }
            req = self.comm.isend(data, dest=dest, tag=1)
            send_reqs.append((req, indices))

        if by_rank:
            all_sent = []
            for _, indices in by_rank.items():
                all_sent.extend(indices)
            keep = np.ones(self.num_local, dtype=bool)
            keep[all_sent] = False
            self.x = self.x[keep]
            self.y = self.y[keep]
            self.vx = self.vx[keep]
            self.vy = self.vy[keep]
            self.num_local = len(self.x)

        received = []
        status = MPI.Status()
        for _ in range(self.size * 10):
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=1, status=status):
                src = status.Get_source()
                data = self.comm.recv(source=src, tag=1)
                received.append(data)
            else:
                break

        if received:
            for data in received:
                if len(data['x']) > 0:
                    mask = np.array([self.domain.in_domain(data['x'][i], data['y'][i])
                                   for i in range(len(data['x']))])
                    if np.any(mask):
                        self.x = np.concatenate([self.x, data['x'][mask]])
                        self.y = np.concatenate([self.y, data['y'][mask]])
                        self.vx = np.concatenate([self.vx, data['vx'][mask]])
                        self.vy = np.concatenate([self.vy, data['vy'][mask]])
                        self.num_local = len(self.x)

        for req, _ in send_reqs:
            req.Wait()

    def get_positions(self):
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
        print(f"1D Block Split - Particles: {num_particles}, Ranks: {size}")
        if visualize:
            pygame.init()
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(f"MPI 1D Block Split - {num_particles} particles, {size} ranks")
            clock = pygame.time.Clock()
            font = pygame.font.Font(None, 36)
            rank_colors = [WHITE, BLUE, GREEN, RED, YELLOW, CYAN, MAGENTA, ORANGE, PURPLE, PINK]

    ps = ParticleSystemMPI(num_particles, comm)
    comm.Barrier()

    if rank == 0 and visualize:
        print(f"Domain boundaries:")
        for r in range(size):
            x_min = r * (WIDTH / size)
            x_max = (r + 1) * (WIDTH / size) if r < size - 1 else WIDTH
            print(f"  Rank {r}: x in [{x_min:.1f}, {x_max:.1f})")

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

        if not running and rank == 0:
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

                if size <= len(rank_colors):
                    for r in range(size):
                        x_min = r * (WIDTH / size)
                        x_max = (r + 1) * (WIDTH / size) if r < size - 1 else WIDTH
                        color = rank_colors[r % len(rank_colors)]
                        pygame.draw.line(screen, color, (x_min, 0), (x_min, HEIGHT), 2)
                        if r == size - 1:
                            pygame.draw.line(screen, color, (x_max, 0), (x_max, HEIGHT), 2)

                if len(x_all) > 0:
                    for i in range(len(x_all)):
                        px = int(np.clip(x_all[i], 0, WIDTH - 1))
                        py = int(np.clip(y_all[i], 0, HEIGHT - 1))
                        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                            pygame.draw.circle(screen, WHITE, (px, py), 2)

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
            running = False

    local_count = ps.get_count()
    all_counts = np.empty(size, dtype=np.int32)
    comm.Allgather([np.array([local_count], dtype=np.int32), MPI.INT], [all_counts, MPI.INT])

    if rank == 0:
        if visualize:
            pygame.quit()
        avg = total_time / frame_count if frame_count > 0 else 0
        print(f"Avg time: {avg:.4f} ms/frame")
        print("Final particle distribution:")
        for r in range(size):
            print(f"  Rank {r}: {all_counts[r]}")
        print(f"  Total: {np.sum(all_counts)}")

    return ps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--particles', type=int, default=1000)
    parser.add_argument('--frames', type=int, default=None)
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    run_simulation(args.particles, args.frames, visualize=not args.no_viz)

