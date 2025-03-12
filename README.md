# Fluid Simulation

## Overview
This project is a real-time 2D fluid simulation using numerical methods to approximate the Navier-Stokes equations. The simulation is implemented in Python with Pygame for visualization.

## Features
- Real-time fluid simulation with velocity and density fields
- User interaction to add forces and density
- Pygame-based visualization
- Adjustable parameters (diffusion, viscosity, time step)
- Velocity and density visualization

## Mathematical Background

The simulation is based on the Navier-Stokes equations, which describe fluid motion:

```
∂u/∂t + (u ⋅ ∇)u = - (1/ρ) ∇p + ν ∇²u + f
```

where:
- `u` is the velocity field
- `p` is the pressure
- `ρ` is the density
- `ν` is the viscosity
- `f` represents external forces

Additionally, the incompressibility condition is enforced:

```
∇ ⋅ u = 0
```

### Discretization

The simulation uses a grid-based discretization with numerical techniques:
- **Semi-Lagrangian advection**: Traces fluid particles backward in time for stability.
- **Gauss-Seidel relaxation**: Solves diffusion and pressure projection iteratively.
- **Pressure projection**: Ensures incompressibility by solving a Poisson equation.

## Installation

### Prerequisites
Ensure you have Python installed along with the required dependencies:

```sh
pip install numpy pygame
```

### Running the Simulation
To start the simulation, run:

```sh
python fluidsim.py
```

## Usage

### Controls
- **Left Mouse Click**: Add fluid density and velocity.
- **Space**: Pause/unpause simulation.
- **V**: Toggle velocity visualization.
- **C**: Clear simulation.
- **R**: Add random forces.
- **Escape**: Quit the simulation.

## Code Structure
- `FluidSimulator`: Handles fluid dynamics calculations.
- `FluidSimulationGame`: Manages Pygame visualization and user interactions.
