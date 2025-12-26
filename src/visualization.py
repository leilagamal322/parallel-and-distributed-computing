# visualization.py
import pygame
from gpu_simulation_pycuda import ParticleSystemGPU

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GPU Particle Simulation (Visualization)")
clock = pygame.time.Clock()

# Create GPU simulation
particle_system = ParticleSystemGPU(num_particles=1000)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # GPU update
    particle_system.update()
    x, y = particle_system.get_positions()

    # Render
    screen.fill((0, 0, 0))
    for i in range(len(x)):
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (int(x[i]), int(y[i])),
            2
        )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
