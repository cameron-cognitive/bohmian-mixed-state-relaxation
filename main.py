#!/usr/bin/env python3
"""
Main entry point for the Bohmian mixed-state relaxation simulation.

This script coordinates the entire simulation process including:
- Setting up the physical system parameters
- Constructing pure states and mixed states
- Generating non-equilibrium initial conditions
- Running the time evolution
- Computing the H-function
- Producing animations and plots
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation
from src.visualization import animate_evolution, plot_h_function


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simulate Bohmian relaxation in a 2D box")
    parser.add_argument(
        "--modes", type=int, default=4, help="Number of modes in each pure state"
    )
    parser.add_argument(
        "--particles", type=int, default=1000, help="Number of particles to simulate"
    )
    parser.add_argument(
        "--tmax", type=float, default=2.0, help="Maximum simulation time (in box periods)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Time step for integration"
    )
    parser.add_argument(
        "--aligned-phases", action="store_true", help="Use aligned phases for partial convergence"
    )
    parser.add_argument(
        "--random-phases", action="store_true", help="Use random phases for full convergence"
    )
    parser.add_argument(
        "--pure-state", action="store_true", help="Simulate pure state (not mixed)"
    )
    parser.add_argument(
        "--grid-size", type=int, default=100, help="Grid size for position space discretization"
    )
    parser.add_argument(
        "--coarse-grain", type=int, nargs="*", default=[5, 10, 20],
        help="Coarse graining levels to analyze"
    )
    parser.add_argument(
        "--animate", action="store_true", help="Generate animations of the simulation"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    """Main simulation function."""
    args = parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Initialize the physical system (2D infinite square well)
    system = InfiniteSquareWell2D(
        L=np.pi,  # Length of the box side
        hbar=1.0,  # Reduced Planck constant
        m=1.0,     # Particle mass
        Nx=args.grid_size,  # Number of grid points in x
        Ny=args.grid_size   # Number of grid points in y
    )
    
    # Define time parameters
    dt = args.dt
    t_max = args.tmax * 2.0 * np.pi  # Expressed in terms of box periods
    time_steps = int(t_max / dt)
    time_values = np.linspace(0, t_max, time_steps)
    
    # Set up quantum states
    if args.pure_state:
        # For pure state simulation (Valentini's original setup)
        mode_indices = []
        for i in range(args.modes):
            nx = np.random.randint(1, 4)  # Keep mode range modest
            ny = np.random.randint(1, 4)
            mode_indices.append((nx, ny))
        
        # Set phases according to command line flags
        if args.aligned_phases:
            phases = np.zeros(args.modes)  # All phases aligned to zero
        elif args.random_phases:
            phases = 2 * np.pi * np.random.random(args.modes)  # Random phases
        else:
            # Default: first mode at 0, rest at π/2 intervals
            phases = np.array([0] + [i * np.pi/2 for i in range(1, args.modes)])
        
        # Equal amplitudes for simplicity
        amplitudes = np.ones(args.modes) / np.sqrt(args.modes)
        
        # Create the pure quantum state
        quantum_state = PureState(system, mode_indices, amplitudes, phases)
        
    else:
        # For mixed state simulation (our extension)
        # Define two different pure states
        mode_indices_1 = []
        mode_indices_2 = []
        
        # Generate two sets of modes with minimal overlap
        for i in range(args.modes // 2):
            nx1 = np.random.randint(1, 3)
            ny1 = np.random.randint(1, 3)
            mode_indices_1.append((nx1, ny1))
            
            # Make second set different
            nx2 = np.random.randint(2, 4)
            ny2 = np.random.randint(2, 4)
            mode_indices_2.append((nx2, ny2))
        
        # Set phases according to command line flags
        if args.aligned_phases:
            phases_1 = np.zeros(len(mode_indices_1))  # All phases aligned
            phases_2 = np.zeros(len(mode_indices_2))  # All phases aligned
        elif args.random_phases:
            phases_1 = 2 * np.pi * np.random.random(len(mode_indices_1))
            phases_2 = 2 * np.pi * np.random.random(len(mode_indices_2))
        else:
            # Default phases
            phases_1 = np.array([0] + [i * np.pi/2 for i in range(1, len(mode_indices_1))])
            phases_2 = np.array([np.pi/4] + [(i+1) * np.pi/2 for i in range(1, len(mode_indices_2))])
        
        # Equal amplitudes within each state
        amplitudes_1 = np.ones(len(mode_indices_1)) / np.sqrt(len(mode_indices_1))
        amplitudes_2 = np.ones(len(mode_indices_2)) / np.sqrt(len(mode_indices_2))
        
        # Create individual pure states
        psi_1 = PureState(system, mode_indices_1, amplitudes_1, phases_1)
        psi_2 = PureState(system, mode_indices_2, amplitudes_2, phases_2)
        
        # Create mixed state (50/50 mixture)
        weights = [0.5, 0.5]
        quantum_state = MixedState(system, [psi_1, psi_2], weights)

    # Create the simulation handler
    simulation = BohmianRelaxation(
        system=system,
        quantum_state=quantum_state,
        n_particles=args.particles,
        dt=dt,
        t_max=t_max
    )
    
    # Generate non-equilibrium initial conditions via backward evolution
    print("Generating non-equilibrium initial conditions...")
    start_time = time.time()
    simulation.generate_initial_conditions()
    
    # Run the simulation
    print(f"Running forward simulation with {args.particles} particles...")
    results = simulation.run_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Calculate H-function and its coarse-grained variants
    h_values = simulation.calculate_h_function()
    h_coarse_values = {}
    for cg_level in args.coarse_grain:
        h_coarse_values[cg_level] = simulation.calculate_h_function(coarse_grain=cg_level)
    
    # Plot results
    print("Generating plots...")
    plot_h_function(time_values, h_values, h_coarse_values, 
                   title="Quantum H-Function Evolution",
                   save_path="h_function.png")
    
    # Generate animations if requested
    if args.animate:
        print("Generating animations...")
        animate_evolution(
            simulation=simulation,
            results=results,
            interval=100,  # milliseconds between frames
            save_path="density_evolution.mp4"
        )
        
    print("Simulation complete!")


if __name__ == "__main__":
    main()
