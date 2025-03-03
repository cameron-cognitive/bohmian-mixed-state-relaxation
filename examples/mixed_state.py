#!/usr/bin/env python3
"""
Example script demonstrating mixed-state relaxation.

This example extends Valentini's work to mixed states represented by density matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation
from src.visualization import plot_mixed_state, plot_particle_distribution, plot_h_function, animate_evolution, compare_pure_and_mixed


def run_mixed_state_example():
    """Run the mixed state relaxation example."""
    # Set up the physical system
    system = InfiniteSquareWell2D(
        L=np.pi,  # Length of the box side
        hbar=1.0,  # Reduced Planck constant
        m=1.0,     # Particle mass
        Nx=100,    # Number of grid points in x
        Ny=100     # Number of grid points in y
    )
    
    # Define simulation parameters
    n_particles = 1000  # Number of particles to simulate
    dt = 0.01          # Time step for integration
    t_max = 2.0 * np.pi  # Maximum simulation time (two box periods)
    
    # Define first pure state
    mode_indices_1 = [(1, 1), (1, 2), (2, 1), (2, 2)]  # (nx, ny) mode indices for first state
    amplitudes_1 = np.ones(len(mode_indices_1)) / np.sqrt(len(mode_indices_1))  # Equal amplitudes
    
    # For comparison, we'll create both random and aligned phase cases
    np.random.seed(42)  # For reproducibility
    phases_random_1 = 2 * np.pi * np.random.random(len(mode_indices_1))
    phases_aligned_1 = np.zeros(len(mode_indices_1))  # All phases = 0
    
    # Define second pure state (different from the first to ensure a true mixed state)
    mode_indices_2 = [(2, 2), (2, 3), (3, 2), (3, 3)]  # (nx, ny) mode indices for second state
    amplitudes_2 = np.ones(len(mode_indices_2)) / np.sqrt(len(mode_indices_2))  # Equal amplitudes
    phases_random_2 = 2 * np.pi * np.random.random(len(mode_indices_2))
    phases_aligned_2 = np.zeros(len(mode_indices_2))  # All phases = 0
    
    # Create the pure states
    psi1_random = PureState(system, mode_indices_1, amplitudes_1, phases_random_1)
    psi2_random = PureState(system, mode_indices_2, amplitudes_2, phases_random_2)
    psi1_aligned = PureState(system, mode_indices_1, amplitudes_1, phases_aligned_1)
    psi2_aligned = PureState(system, mode_indices_2, amplitudes_2, phases_aligned_2)
    
    # Create mixed states (50/50 mixture)
    mixed_state_random = MixedState(system, [psi1_random, psi2_random], [0.5, 0.5])
    mixed_state_aligned = MixedState(system, [psi1_aligned, psi2_aligned], [0.5, 0.5])
    
    # Also create pure state counterparts for comparison
    pure_state_random = psi1_random  # Just use first state
    pure_state_aligned = psi1_aligned
    
    # Visualize the mixed state probability density
    fig1 = plot_mixed_state(mixed_state_random, t=0, 
                         title="Mixed State Probability Density (Random Phases)", 
                         save_path="mixed_random_density_t0.png")
    
    fig2 = plot_mixed_state(mixed_state_aligned, t=0, 
                         title="Mixed State Probability Density (Aligned Phases)", 
                         save_path="mixed_aligned_density_t0.png")
    
    plt.close(fig1)
    plt.close(fig2)
    
    # Run simulation with random phases (full relaxation expected)
    print("\nRunning mixed state simulation with random phases:")
    simulation_mixed_random = BohmianRelaxation(
        system=system,
        quantum_state=mixed_state_random,
        n_particles=n_particles,
        dt=dt,
        t_max=t_max
    )
    
    # For comparison, run pure state with random phases
    simulation_pure_random = BohmianRelaxation(
        system=system,
        quantum_state=pure_state_random,
        n_particles=n_particles,
        dt=dt,
        t_max=t_max
    )
    
    start_time = time.time()
    print("Generating non-equilibrium initial conditions for mixed state...")
    simulation_mixed_random.generate_initial_conditions()
    
    print("Generating non-equilibrium initial conditions for pure state...")
    simulation_pure_random.generate_initial_conditions()
    
    print("Running forward simulations...")
    results_mixed_random = simulation_mixed_random.run_simulation()
    results_pure_random = simulation_pure_random.run_simulation()
    end_time = time.time()
    print(f"Simulations completed in {end_time - start_time:.2f} seconds")
    
    # Time values for plotting
    time_values = np.linspace(0, t_max, simulation_mixed_random.n_timesteps)
    
    # Compare pure and mixed state random phase cases
    fig3 = compare_pure_and_mixed(
        pure_simulation=simulation_pure_random,
        mixed_simulation=simulation_mixed_random,
        time_values=time_values,
        save_path="pure_vs_mixed_random.png"
    )
    plt.close(fig3)
    
    # Run simulation with aligned phases (partial relaxation expected)
    print("\nRunning mixed state simulation with aligned phases:")
    simulation_mixed_aligned = BohmianRelaxation(
        system=system,
        quantum_state=mixed_state_aligned,
        n_particles=n_particles,
        dt=dt,
        t_max=t_max
    )
    
    # For comparison, run pure state with aligned phases
    simulation_pure_aligned = BohmianRelaxation(
        system=system,
        quantum_state=pure_state_aligned,
        n_particles=n_particles,
        dt=dt,
        t_max=t_max
    )
    
    start_time = time.time()
    print("Generating non-equilibrium initial conditions for mixed state...")
    simulation_mixed_aligned.generate_initial_conditions()
    
    print("Generating non-equilibrium initial conditions for pure state...")
    simulation_pure_aligned.generate_initial_conditions()
    
    print("Running forward simulations...")
    results_mixed_aligned = simulation_mixed_aligned.run_simulation()
    results_pure_aligned = simulation_pure_aligned.run_simulation()
    end_time = time.time()
    print(f"Simulations completed in {end_time - start_time:.2f} seconds")
    
    # Compare pure and mixed state aligned phase cases
    fig4 = compare_pure_and_mixed(
        pure_simulation=simulation_pure_aligned,
        mixed_simulation=simulation_mixed_aligned,
        time_values=time_values,
        save_path="pure_vs_mixed_aligned.png"
    )
    plt.close(fig4)
    
    # Compare mixed state with random vs. aligned phases
    plt.figure(figsize=(10, 6))
    
    # Calculate H-functions
    h_mixed_random = simulation_mixed_random.calculate_h_function()
    h_mixed_aligned = simulation_mixed_aligned.calculate_h_function()
    
    # Normalize
    h_mixed_random_norm = h_mixed_random / h_mixed_random[0]
    h_mixed_aligned_norm = h_mixed_aligned / h_mixed_aligned[0]
    
    plt.plot(time_values, h_mixed_random_norm, 'b-', linewidth=2, 
            label='Mixed State - Random Phases')
    plt.plot(time_values, h_mixed_aligned_norm, 'r-', linewidth=2, 
            label='Mixed State - Aligned Phases')
    plt.axhline(y=0.1, color='k', linestyle='--', alpha=0.5, label='10% Residual')
    
    plt.xlabel('Time $t$')
    plt.ylabel('Normalized H-function $H(t)/H(0)$')
    plt.title('Comparison of Mixed State Relaxation with Random vs. Aligned Phases')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig("mixed_state_comparison.png", dpi=300, bbox_inches='tight')
    
    # Create an animation of the mixed state with aligned phases (likely shows partial non-convergence)
    print("\nGenerating animation...")
    anim = animate_evolution(
        simulation=simulation_mixed_aligned,
        results=results_mixed_aligned,
        interval=100,
        save_path="mixed_aligned_animation.mp4"
    )
    
    print("\nExample completed successfully! Check the output files for results.")
    return anim  # Return the animation object for interactive viewing


if __name__ == "__main__":
    anim = run_mixed_state_example()
    plt.show()  # Show the animation if running interactively
