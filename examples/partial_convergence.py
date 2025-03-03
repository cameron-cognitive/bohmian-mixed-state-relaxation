#!/usr/bin/env python3
"""
Example script demonstrating partial non-convergence to quantum equilibrium.

This example demonstrates how to achieve a ~10% residual in the H-function decay
using specific mode selections and aligned phases.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation
from src.visualization import plot_wavefunction, plot_mixed_state, plot_h_function, animate_evolution, plot_h_matrix


def run_partial_convergence_example():
    """Run the partial convergence demonstration."""
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
    
    print("Setting up states for partial non-convergence demonstration...")
    
    # Pure state case - using commensurate frequencies and aligned phases
    # Following Valentini's approach for partial non-convergence
    mode_indices_pure = [(2, 1), (2, 2), (3, 1), (3, 2)]  # Commensurate frequencies
    amplitudes_pure = np.ones(len(mode_indices_pure)) / np.sqrt(len(mode_indices_pure))  # Equal amplitudes
    phases_pure = np.zeros(len(mode_indices_pure))  # All phases aligned to zero
    
    pure_state = PureState(system, mode_indices_pure, amplitudes_pure, phases_pure)
    
    # Mixed state case - similar approach but with two sets of modes
    mode_indices_1 = [(2, 1), (2, 2)]  # First component
    mode_indices_2 = [(3, 1), (3, 2)]  # Second component
    
    amplitudes_1 = np.ones(len(mode_indices_1)) / np.sqrt(len(mode_indices_1))
    amplitudes_2 = np.ones(len(mode_indices_2)) / np.sqrt(len(mode_indices_2))
    
    # All phases aligned to zero for both components
    phases_1 = np.zeros(len(mode_indices_1))
    phases_2 = np.zeros(len(mode_indices_2))
    
    psi_1 = PureState(system, mode_indices_1, amplitudes_1, phases_1)
    psi_2 = PureState(system, mode_indices_2, amplitudes_2, phases_2)
    
    mixed_state = MixedState(system, [psi_1, psi_2], [0.5, 0.5])
    
    # Visualize the probability densities
    fig1 = plot_wavefunction(pure_state, t=0, 
                           title="Pure State with Commensurate Frequencies", 
                           save_path="partial_convergence_pure_t0.png")
    
    fig2 = plot_mixed_state(mixed_state, t=0, 
                          title="Mixed State with Commensurate Frequencies", 
                          save_path="partial_convergence_mixed_t0.png")
    
    plt.close(fig1)
    plt.close(fig2)
    
    # Run pure state simulation
    print("\nRunning pure state simulation with commensurate frequencies:")
    simulation_pure = BohmianRelaxation(
        system=system,
        quantum_state=pure_state,
        n_particles=n_particles,
        dt=dt,
        t_max=t_max
    )
    
    start_time = time.time()
    print("Generating non-equilibrium initial conditions...")
    simulation_pure.generate_initial_conditions()
    
    print("Running forward simulation...")
    results_pure = simulation_pure.run_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Calculate H-function for pure state
    h_values_pure = simulation_pure.calculate_h_function()
    
    # Run mixed state simulation
    print("\nRunning mixed state simulation with commensurate frequencies:")
    simulation_mixed = BohmianRelaxation(
        system=system,
        quantum_state=mixed_state,
        n_particles=n_particles,
        dt=dt,
        t_max=t_max
    )
    
    start_time = time.time()
    print("Generating non-equilibrium initial conditions...")
    simulation_mixed.generate_initial_conditions()
    
    print("Running forward simulation...")
    results_mixed = simulation_mixed.run_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Calculate H-function for mixed state
    h_values_mixed = simulation_mixed.calculate_h_function()
    
    # Time values for plotting
    time_values = np.linspace(0, t_max, len(h_values_pure))
    
    # Plot normalized H-functions to see the ~10% residual
    plt.figure(figsize=(10, 6))
    
    # Normalize to initial values
    h_pure_norm = h_values_pure / h_values_pure[0]
    h_mixed_norm = h_values_mixed / h_values_mixed[0]
    
    plt.plot(time_values, h_pure_norm, 'b-', linewidth=2, 
            label='Pure State')
    plt.plot(time_values, h_mixed_norm, 'r-', linewidth=2, 
            label='Mixed State')
    
    # Add horizontal lines at 0% and 10% for reference
    plt.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(y=0.1, color='k', linestyle='--', alpha=0.5, label='10% Residual')
    
    plt.xlabel('Time $t$')
    plt.ylabel('Normalized H-function $H(t)/H(0)$')
    plt.title('Partial Convergence to Quantum Equilibrium')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig("partial_convergence_comparison.png", dpi=300, bbox_inches='tight')
    
    # Visualization of the H-function matrix at the final time
    # This shows where the non-convergence is happening in configuration space
    fig3 = plot_h_matrix(simulation_pure, timestep=simulation_pure.n_timesteps - 1, coarse_grain=30,
                       title="Pure State H-Function Matrix at Final Time", 
                       save_path="pure_h_matrix_final.png")
    
    fig4 = plot_h_matrix(simulation_mixed, timestep=simulation_mixed.n_timesteps - 1, coarse_grain=30,
                       title="Mixed State H-Function Matrix at Final Time", 
                       save_path="mixed_h_matrix_final.png")
    
    plt.close(fig3)
    plt.close(fig4)
    
    # Create animation of the mixed state evolution
    print("\nGenerating animation...")
    anim = animate_evolution(
        simulation=simulation_mixed,
        results=results_mixed,
        interval=100,
        save_path="partial_convergence_animation.mp4"
    )
    
    print("\nExample completed successfully!")
    print("Final normalized H-function values:")
    print(f"  Pure state:  {h_pure_norm[-1]:.4f} (~ {h_pure_norm[-1]*100:.1f}% residual)")
    print(f"  Mixed state: {h_mixed_norm[-1]:.4f} (~ {h_mixed_norm[-1]*100:.1f}% residual)")
    print("\nCheck the output files for visualizations.")
    
    return anim  # Return the animation object for interactive viewing


if __name__ == "__main__":
    anim = run_partial_convergence_example()
    plt.show()  # Show the animation if running interactively
