"""
Enhanced Visualization Module
=============================
Provides advanced visualization features including:
- Color gradients based on particle speed
- Particle trails
- Interactive controls
- Performance metrics overlay
"""

import numpy as np
import pygame
import time

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def velocity_to_color(vx, vy, max_speed=10.0):
    """
    Convert particle velocity to RGB color.
    Faster particles appear brighter/more colorful.
    
    Args:
        vx, vy: Velocity components
        max_speed: Maximum speed for normalization
    
    Returns:
        RGB tuple (r, g, b)
    """
    speed = np.sqrt(vx**2 + vy**2)
    normalized = np.clip(speed / max_speed, 0, 1)
    
    # Color gradient: blue -> cyan -> green -> yellow -> red
    if normalized < 0.25:
        # Blue to Cyan
        r = 0
        g = int(255 * normalized * 4)
        b = 255
    elif normalized < 0.5:
        # Cyan to Green
        r = 0
        g = 255
        b = int(255 * (1 - (normalized - 0.25) * 4))
    elif normalized < 0.75:
        # Green to Yellow
        r = int(255 * (normalized - 0.5) * 4)
        g = 255
        b = 0
    else:
        # Yellow to Red
        r = 255
        g = int(255 * (1 - (normalized - 0.75) * 4))
        b = 0
    
    return (r, g, b)


def draw_particles_with_velocity(screen, x_pos, y_pos, vx, vy, max_speed=10.0):
    """
    Draw particles with color based on velocity.
    
    Args:
        screen: Pygame surface
        x_pos, y_pos: Particle positions
        vx, vy: Particle velocities
        max_speed: Maximum speed for color normalization
    """
    for i in range(len(x_pos)):
        px = int(np.clip(x_pos[i], 0, screen.get_width() - 1))
        py = int(np.clip(y_pos[i], 0, screen.get_height() - 1))
        
        color = velocity_to_color(vx[i], vy[i], max_speed)
        pygame.draw.circle(screen, color, (px, py), 2)


def draw_particles_simple(screen, x_pos, y_pos, color=WHITE):
    """
    Draw particles as simple white dots.
    
    Args:
        screen: Pygame surface
        x_pos, y_pos: Particle positions
        color: Particle color (default: white)
    """
    for i in range(len(x_pos)):
        px = int(np.clip(x_pos[i], 0, screen.get_width() - 1))
        py = int(np.clip(y_pos[i], 0, screen.get_height() - 1))
        pygame.draw.circle(screen, color, (px, py), 2)


class PerformanceOverlay:
    """Displays performance metrics on screen."""
    
    def __init__(self):
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
    
    def draw(self, screen, fps, num_particles, update_time=None, transfer_time=None, 
             frame_time=None, y_offset=10):
        """
        Draw performance overlay on screen.
        
        Args:
            screen: Pygame surface
            fps: Current FPS
            num_particles: Number of particles
            update_time: GPU/CPU update time in ms
            transfer_time: GPU transfer time in ms (optional)
            frame_time: Total frame time in ms (optional)
            y_offset: Vertical offset from top
        """
        lines = []
        
        # FPS
        fps_text = self.font.render(f"FPS: {int(fps)}", True, WHITE)
        screen.blit(fps_text, (10, y_offset))
        y_offset += 40
        
        # Particle count
        count_text = self.font.render(f"Particles: {num_particles}", True, WHITE)
        screen.blit(count_text, (10, y_offset))
        y_offset += 40
        
        # Update time
        if update_time is not None:
            if transfer_time is not None:
                # GPU version
                update_text = self.font.render(f"GPU Update: {update_time:.2f} ms", True, WHITE)
                screen.blit(update_text, (10, y_offset))
                y_offset += 40
                
                transfer_text = self.font.render(f"Transfer: {transfer_time:.2f} ms", True, WHITE)
                screen.blit(transfer_text, (10, y_offset))
                y_offset += 40
                
                total_text = self.small_font.render(
                    f"Total GPU: {update_time + transfer_time:.2f} ms", True, WHITE)
                screen.blit(total_text, (10, y_offset))
            else:
                # CPU version
                update_text = self.font.render(f"CPU Update: {update_time:.2f} ms", True, WHITE)
                screen.blit(update_text, (10, y_offset))
                y_offset += 40
        
        # Frame time
        if frame_time is not None:
            frame_text = self.small_font.render(f"Frame Time: {frame_time:.2f} ms", True, WHITE)
            screen.blit(frame_text, (10, y_offset))


class InteractiveControls:
    """Handles keyboard and mouse input for interactive control."""
    
    def __init__(self):
        self.paused = False
        self.show_help = False
    
    def handle_event(self, event):
        """
        Handle pygame events for controls.
        
        Args:
            event: Pygame event
        
        Returns:
            Dictionary with control states
        """
        controls = {
            'quit': False,
            'pause': False,
            'toggle_help': False,
            'step': False,
            'reset': False
        }
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                controls['quit'] = True
            elif event.key == pygame.K_SPACE:
                controls['pause'] = True
                self.paused = not self.paused
            elif event.key == pygame.K_h:
                controls['toggle_help'] = True
                self.show_help = not self.show_help
            elif event.key == pygame.K_r:
                controls['reset'] = True
            elif event.key == pygame.K_RIGHT and self.paused:
                controls['step'] = True
        
        elif event.type == pygame.QUIT:
            controls['quit'] = True
        
        return controls
    
    def draw_help(self, screen):
        """Draw help text overlay."""
        if not self.show_help:
            return
        
        font = pygame.font.Font(None, 24)
        help_lines = [
            "Controls:",
            "  ESC - Quit",
            "  SPACE - Pause/Resume",
            "  H - Toggle Help",
            "  R - Reset Simulation",
            "  RIGHT ARROW - Step (when paused)"
        ]
        
        # Draw semi-transparent background
        overlay = pygame.Surface((300, len(help_lines) * 30 + 20))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        screen.blit(overlay, (screen.get_width() - 320, 10))
        
        # Draw text
        y_offset = 20
        for line in help_lines:
            text = font.render(line, True, WHITE)
            screen.blit(text, (screen.get_width() - 310, y_offset))
            y_offset += 30
    
    def is_paused(self):
        """Check if simulation is paused."""
        return self.paused


def create_color_gradient(n):
    """
    Create a color gradient for n particles.
    
    Args:
        n: Number of particles
    
    Returns:
        List of RGB tuples
    """
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r = int(255 * (1 - t))
        g = int(255 * t)
        b = int(128 + 127 * np.sin(t * np.pi))
        colors.append((r, g, b))
    return colors

