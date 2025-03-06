#!/usr/bin/env python3
"""
Script to run von Neumann guidance equation simulations for mixed quantum states.

This script demonstrates how non-equilibrium distributions of particles evolve
according to the von Neumann guidance equation for mixed quantum states.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation
from src.von_neumann_visualization import VonNeumannRelaxationVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run von Neumann guidance equation simulation")
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
        "--output-dir", type=str, default="von_neumann_results", help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def setup_mixed_state(system, args):
    """Set up a mixed quantum state for simulation.
    
    Args:
        system: InfiniteSquareWell2D system
        args: Command line arguments
        
    Returns:
        MixedState object
    """
    # Define three different pure states for a more complex mixed state
    mode_indices_1 = [(1, 1), (2, 1), (1, 2)]  # Lower energy modes
    mode_indices_2 = [(2, 2), (3, 1), (1, 3)]  # Medium energy modes
    mode_indices_3 = [(3, 2), (2, 3), (3, 3)]  # Higher energy modes
    
    # Set phases according to command line flags
    if args.aligned_phases:
        # All phases aligned to zero for partial relaxation
        phases_1 = np.zeros(len(mode_indices_1))
        phases_2 = np.zeros(len(mode_indices_2))
        phases_3 = np.zeros(len(mode_indices_3))
    elif args.random_phases:
        # Random phases for complete relaxation
        phases_1 = 2 * np.pi * np.random.random(len(mode_indices_1))
        phases_2 = 2 * np.pi * np.random.random(len(mode_indices_2))
        phases_3 = 2 * np.pi * np.random.random(len(mode_indices_3))
    else:
        # Default phases - some structure but not fully aligned
        phases_1 = np.array([0, np.pi/2, np.pi])
        phases_2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4])
        phases_3 = np.array([np.pi/3, 2*np.pi/3, 4*np.pi/3])
    
    # Equal amplitudes within each state
    amplitudes_1 = np.ones(len(mode_indices_1)) / np.sqrt(len(mode_indices_1))
    amplitudes_2 = np.ones(len(mode_indices_2)) / np.sqrt(len(mode_indices_2))
    amplitudes_3 = np.ones(len(mode_indices_3)) / np.sqrt(len(mode_indices_3))
    
    # Create individual pure states
    psi_1 = PureState(system, mode_indices_1, amplitudes_1, phases_1)
    psi_2 = PureState(system, mode_indices_2, amplitudes_2, phases_2)
    psi_3 = PureState(system, mode_indices_3, amplitudes_3, phases_3)
    
    # Create mixed state with unequal weights
    weights = [0.5, 0.3, 0.2]  # Weighted mixture
    mixed_state = MixedState(system, [psi_1, psi_2, psi_3], weights)
    
    return mixed_state


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the physical system (2D infinite square well)
    system = InfiniteSquareWell2D(
        L=np.pi,  # Length of the box side
        hbar=1.0,  # Reduced Planck constant
        m=1.0,     # Particle mass
        Nx=100,    # Number of grid points in x
        Ny=100     # Number of grid points in y
    )
    
    # Define time parameters
    dt = args.dt
    t_max = args.tmax * 2.0 * np.pi  # Expressed in terms of box periods
    
    # Set up mixed quantum state
    quantum_state = setup_mixed_state(system, args)
    
    # Create the simulation handler
    simulation = BohmianRelaxation(
        system=system,
        quantum_state=quantum_state,
        n_particles=args.particles,
        dt=dt,
        t_max=t_max
    )
    
    # Run the simulation
    print("Starting von Neumann mixed-state relaxation simulation...")
    print(f"Number of particles: {args.particles}")
    print(f"Maximum simulation time: {args.tmax} box periods")
    print(f"Phases: {'aligned' if args.aligned_phases else 'random' if args.random_phases else 'default'}")
    
    # Generate non-equilibrium initial conditions via backward evolution
    print("\nGenerating non-equilibrium initial conditions via backward evolution...")
    start_time = time.time()
    simulation.generate_initial_conditions()
    
    # Run the simulation
    print(f"\nRunning forward simulation with {args.particles} particles...")
    results = simulation.run_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Create the visualizer
    visualizer = VonNeumannRelaxationVisualizer(simulation, results)
    
    # Define coarse-graining levels
    coarse_graining_levels = [
        int(np.ceil(np.pi / (np.pi/8))),   # ε = π/8
        int(np.ceil(np.pi / (np.pi/16))),  # ε = π/16
        int(np.ceil(np.pi / (np.pi/32))),  # ε = π/32
        int(np.ceil(np.pi / (np.pi/64)))   # ε = π/64
    ]
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Create density matrix comparisons for key timesteps
    timesteps_to_analyze = [
        0,                               # Initial state
        int(simulation.n_timesteps * 0.1),  # Early evolution
        int(simulation.n_timesteps * 0.25), # Quarter way
        int(simulation.n_timesteps * 0.5),  # Halfway
        int(simulation.n_timesteps * 0.75), # Three quarters
        simulation.n_timesteps - 1          # Final state
    ]
    
    print("\nGenerating density matrix comparisons for key timesteps...")
    for ts in timesteps_to_analyze:
        t = ts * dt
        print(f"  - t = {t:.3f}")
        save_path = os.path.join(args.output_dir, f'density_comparison_t{ts:04d}.png')
        visualizer.create_density_matrix_comparison(ts, save_path=save_path)
    
    # 2. Create velocity field visualizations
    print("\nGenerating velocity field visualizations...")
    for ts in timesteps_to_analyze:
        t = ts * dt
        print(f"  - t = {t:.3f}")
        save_path = os.path.join(args.output_dir, f'velocity_field_t{ts:04d}.png')
        visualizer.create_velocity_field_visualization(ts, save_path=save_path)
    
    # 3. Create coarse-graining analysis for specific timesteps
    print("\nGenerating comprehensive coarse-graining analysis...")
    analysis_dir = os.path.join(args.output_dir, 'coarse_graining_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Use a smaller subset for detailed analysis
    analysis_timesteps = [0, int(simulation.n_timesteps * 0.5), simulation.n_timesteps - 1]
    visualizer.generate_coarse_graining_analysis(analysis_timesteps, analysis_dir)
    
    # 4. Create animations
    print("\nGenerating animations...")
    
    # 4.1 Animation of coarse-grained relaxation
    print("  - Creating coarse-grained relaxation animation...")
    animation_path = os.path.join(args.output_dir, 'coarse_grained_relaxation.mp4')
    visualizer.animate_coarse_grained_relaxation(animation_path, sample_frames=50)
    
    print("\nAll visualizations completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
