import numpy as np
import pygame
import sys
import time


class FluidSimulator:
    def __init__(self, N=100, diffusion=0.0001, viscosity=0.0001, dt=0.1):
        self.N = N  # Grid size
        self.size = (N + 2, N + 2)  # Include boundary cells
        self.diffusion = diffusion  # How fast density spreads
        self.viscosity = viscosity  # How thick the fluid is
        self.dt = dt  # Time step size

        # Velocity and density fields
        self.u = np.zeros(self.size)  # x-component of velocity
        self.v = np.zeros(self.size)  # y-component of velocity
        self.u_prev = np.zeros(self.size)
        self.v_prev = np.zeros(self.size)
        self.density = np.zeros(self.size)
        self.density_prev = np.zeros(self.size)

    def add_source(self, x, s):
        """Add source to field"""
        x += s * self.dt

    def set_boundary(self, b, x):
        """Set boundary conditions"""
        # Handle edges
        if b == 1:  # For u (x-velocity)
            x[0, 1:-1] = -x[1, 1:-1]  # Left wall
            x[-1, 1:-1] = -x[-2, 1:-1]  # Right wall
        else:
            x[0, 1:-1] = x[1, 1:-1]  # Left wall
            x[-1, 1:-1] = x[-2, 1:-1]  # Right wall

        if b == 2:  # For v (y-velocity)
            x[1:-1, 0] = -x[1:-1, 1]  # Bottom wall
            x[1:-1, -1] = -x[1:-1, -2]  # Top wall
        else:
            x[1:-1, 0] = x[1:-1, 1]  # Bottom wall
            x[1:-1, -1] = x[1:-1, -2]  # Top wall

        # Handle corners
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

    def diffuse(self, b, x, x0, diffusion):
        """Diffusion using Gauss-Seidel relaxation"""
        a = self.dt * diffusion * self.N * self.N

        for k in range(20):  # 20 iterations for relaxation
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (
                    x[0:-2, 1:-1] + x[2:, 1:-1] +
                    x[1:-1, 0:-2] + x[1:-1, 2:]
            )) / (1 + 4 * a)

            self.set_boundary(b, x)

    def advect(self, b, d, d0, u, v):
        """Advection using semi-Lagrangian method"""
        dt0 = self.dt * self.N

        # For each cell, trace back along velocity field
        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                # Backtrace position
                x = i - dt0 * u[i, j]
                y = j - dt0 * v[i, j]

                # Clamp to grid boundaries
                x = max(0.5, min(self.N + 0.5, x))
                y = max(0.5, min(self.N + 0.5, y))

                # Find grid cell indices
                i0, j0 = int(x), int(y)
                i1, j1 = i0 + 1, j0 + 1

                # Bilinear interpolation weights
                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1

                # Bilinear interpolation of previous values
                d[i, j] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                           s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))

        self.set_boundary(b, d)

    def project(self, u, v, p, div):
        """Projection to make velocity field mass-conserving (divergence-free)"""
        h = 1.0 / self.N

        # Calculate divergence
        div[1:-1, 1:-1] = -0.5 * h * (
                u[2:, 1:-1] - u[0:-2, 1:-1] +
                v[1:-1, 2:] - v[1:-1, 0:-2]
        )
        p[1:-1, 1:-1] = 0

        self.set_boundary(0, div)
        self.set_boundary(0, p)

        # Poisson-pressure equation (Gauss-Seidel relaxation)
        for k in range(20):
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] + p[0:-2, 1:-1] + p[2:, 1:-1] +
                             p[1:-1, 0:-2] + p[1:-1, 2:]) / 4

            self.set_boundary(0, p)

        # Velocity update
        u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[0:-2, 1:-1]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, 0:-2]) / h

        self.set_boundary(1, u)
        self.set_boundary(2, v)

    def density_step(self, density_sources=None):
        """Evolve density field one step"""
        if density_sources is not None:
            self.add_source(self.density, density_sources)

        # Swap density arrays
        self.density_prev, self.density = self.density.copy(), self.density_prev

        # Diffuse density
        self.diffuse(0, self.density, self.density_prev, self.diffusion)

        # Advect density
        self.density_prev, self.density = self.density.copy(), self.density_prev
        self.advect(0, self.density, self.density_prev, self.u, self.v)

    def velocity_step(self, u_forces=None, v_forces=None):
        """Evolve velocity field one step"""
        if u_forces is not None:
            self.add_source(self.u, u_forces)
        if v_forces is not None:
            self.add_source(self.v, v_forces)

        # Swap velocity arrays
        self.u_prev, self.u = self.u.copy(), self.u_prev
        self.v_prev, self.v = self.v.copy(), self.v_prev

        # Diffuse velocity (viscosity)
        self.diffuse(1, self.u, self.u_prev, self.viscosity)
        self.diffuse(2, self.v, self.v_prev, self.viscosity)

        # Project to make velocity divergence-free
        p = np.zeros(self.size)
        div = np.zeros(self.size)
        self.project(self.u, self.v, p, div)

        # Swap velocity arrays
        self.u_prev, self.u = self.u.copy(), self.u_prev
        self.v_prev, self.v = self.v.copy(), self.v_prev

        # Advect velocity
        self.advect(1, self.u, self.u_prev, self.u_prev, self.v_prev)
        self.advect(2, self.v, self.v_prev, self.u_prev, self.v_prev)

        # Project again to ensure incompressibility
        self.project(self.u, self.v, p, div)

    def step(self, density_sources=None, u_forces=None, v_forces=None):
        """Perform one simulation step"""
        self.velocity_step(u_forces, v_forces)
        self.density_step(density_sources)


class FluidSimulationGame:
    def __init__(self, width=800, height=800, N=100):
        """Initialize the Pygame-based fluid simulation visualization"""
        # Initialize Pygame
        pygame.init()

        # Set up the display
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Fluid Simulation")

        # Create fluid simulator
        self.N = N
        self.cell_size = min(width, height) / N
        self.simulator = FluidSimulator(N=N, diffusion=0.0001, viscosity=0.00005, dt=0.2)

        # Create blue to red color map
        self.create_colormap()

        # Setup simulation and interaction variables
        self.last_mouse_pos = None
        self.mouse_down = False
        self.paused = False
        self.show_velocity = False
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.font = pygame.font.SysFont(None, 18)  # Reduced font size

        # Add some initial density in center
        self.add_initial_density()

    def create_colormap(self):
        """Create a blue to red colormap for density visualization"""
        self.colormap = []
        for i in range(256):
            # Blue (cold) to red (hot) gradient
            if i < 128:  # Blue to purple
                r = int(i * 2)
                g = 0
                b = 255
            else:  # Purple to red
                r = 255
                g = 0
                b = int(255 - (i - 128) * 2)
            self.colormap.append((r, g, b))

    def add_initial_density(self):
        """Add some initial density to the center of the simulation"""
        center = (self.N + 2) // 2
        radius = self.N // 8

        for i in range(center - radius, center + radius):
            for j in range(center - radius, center + radius):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if dist < radius:
                    self.simulator.density[i, j] = 1.0 * (1 - dist / radius)

    def handle_input(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.show_velocity = not self.show_velocity
                elif event.key == pygame.K_c:
                    # Clear the simulation
                    self.simulator.density.fill(0)
                    self.simulator.u.fill(0)
                    self.simulator.v.fill(0)
                elif event.key == pygame.K_r:
                    # Add random forces
                    self.add_random_forces()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_down = True
                    self.last_mouse_pos = pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    self.mouse_down = False
                    self.last_mouse_pos = None

        # Handle continuous mouse movement
        if self.mouse_down:
            mouse_pos = pygame.mouse.get_pos()
            if self.last_mouse_pos:
                self.add_user_interaction(self.last_mouse_pos, mouse_pos)
            self.last_mouse_pos = mouse_pos

    def add_user_interaction(self, last_pos, current_pos):
        """Add user interactions (velocity and density) based on mouse movement"""
        # Convert screen coordinates to grid coordinates
        x1, y1 = int(last_pos[0] / self.cell_size) + 1, int(last_pos[1] / self.cell_size) + 1
        x2, y2 = int(current_pos[0] / self.cell_size) + 1, int(current_pos[1] / self.cell_size) + 1

        # Ensure coordinates are within bounds
        x1 = max(1, min(self.N, x1))
        y1 = max(1, min(self.N, y1))
        x2 = max(1, min(self.N, x2))
        y2 = max(1, min(self.N, y2))

        # Calculate velocity based on mouse movement
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy) + 0.001  # Avoid division by zero

        # Create force and density arrays
        u_force = np.zeros(self.simulator.size)
        v_force = np.zeros(self.simulator.size)
        density_source = np.zeros(self.simulator.size)

        # Add force and density in a small radius around the mouse position
        radius = max(2, int(self.N / 50))
        strength = 5.0

        for i in range(x2 - radius, x2 + radius):
            for j in range(y2 - radius, y2 + radius):
                if 1 <= i <= self.N and 1 <= j <= self.N:
                    dist = np.sqrt((i - x2) ** 2 + (j - y2) ** 2)
                    if dist < radius:
                        factor = 1.0 - dist / radius
                        u_force[i, j] = strength * dx / length * factor
                        v_force[i, j] = strength * dy / length * factor
                        density_source[i, j] = 0.5 * factor

        # Apply forces and density
        self.simulator.add_source(self.simulator.u, u_force)
        self.simulator.add_source(self.simulator.v, v_force)
        self.simulator.add_source(self.simulator.density, density_source)

    def add_random_forces(self):
        """Add random forces to create interesting patterns"""
        u_force = np.zeros(self.simulator.size)
        v_force = np.zeros(self.simulator.size)
        density_source = np.zeros(self.simulator.size)

        # Add several force points
        for _ in range(5):
            # Random position
            x = np.random.randint(self.N // 4, self.N * 3 // 4) + 1
            y = np.random.randint(self.N // 4, self.N * 3 // 4) + 1

            # Random direction
            angle = np.random.random() * 2 * np.pi
            dx = np.cos(angle) * 10
            dy = np.sin(angle) * 10

            # Add force and density in a small radius
            radius = max(3, int(self.N / 30))

            for i in range(x - radius, x + radius):
                for j in range(y - radius, y + radius):
                    if 1 <= i <= self.N and 1 <= j <= self.N:
                        dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                        if dist < radius:
                            factor = 1.0 - dist / radius
                            u_force[i, j] = dx * factor
                            v_force[i, j] = dy * factor
                            density_source[i, j] = 0.8 * factor

        # Apply forces and density
        self.simulator.add_source(self.simulator.u, u_force)
        self.simulator.add_source(self.simulator.v, v_force)
        self.simulator.add_source(self.simulator.density, density_source)

    def draw(self):
        """Render the fluid simulation"""
        self.screen.fill((0, 0, 0))

        # Draw density field
        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                density = self.simulator.density[i, j]
                if density > 0.001:  # Only draw non-zero density
                    # Map density to color
                    color_idx = min(255, int(density * 255))
                    color = self.colormap[color_idx]

                    # Draw cell
                    rect = pygame.Rect(
                        (i - 1) * self.cell_size,
                        (j - 1) * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(self.screen, color, rect)

        # Draw velocity field (if enabled)
        if self.show_velocity:
            # Draw fewer arrows for clarity
            skip = max(1, self.N // 25)
            scale = 5.0  # Scale factor for velocity vectors

            for i in range(1, self.N + 1, skip):
                for j in range(1, self.N + 1, skip):
                    u = self.simulator.u[i, j]
                    v = self.simulator.v[i, j]
                    speed = np.sqrt(u * u + v * v)

                    if speed > 0.01:  # Only draw significant velocities
                        # Calculate arrow endpoints
                        start_x = (i - 0.5) * self.cell_size
                        start_y = (j - 0.5) * self.cell_size
                        end_x = start_x + u * scale * self.cell_size
                        end_y = start_y + v * scale * self.cell_size

                        # Draw the arrow
                        pygame.draw.line(self.screen, (255, 255, 255),
                                         (start_x, start_y), (end_x, end_y), 1)

        # Draw status text in two parts to fit better
        status1 = f"FPS: {int(self.clock.get_fps())} | {'PAUSED' if self.paused else 'RUNNING'} | {'Velocity: ON' if self.show_velocity else 'Velocity: OFF'}"
        status2 = "[Space]: Pause | [V]: Toggle Velocity | [C]: Clear | [R]: Random Forces"

        text_surface1 = self.font.render(status1, True, (255, 255, 255))
        text_surface2 = self.font.render(status2, True, (255, 255, 255))

        self.screen.blit(text_surface1, (10, 10))
        self.screen.blit(text_surface2, (10, 30))

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        while True:
            self.handle_input()

            if not self.paused:
                # Step the simulation
                self.simulator.step()

            self.draw()
            self.clock.tick(self.fps)


if __name__ == "__main__":
    # Set simulation size (N×N grid)
    N = 100

    # Create and run the simulation
    sim_game = FluidSimulationGame(width=800, height=800, N=N)
    sim_game.run()