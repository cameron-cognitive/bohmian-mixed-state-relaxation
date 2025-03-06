#!/usr/bin/env python3
"""
Advanced visualization for non-equilibrium Bohmian relaxation in mixed states.

This script extends the Bohmian relaxation simulation to better visualize:
1. Particle trajectories over time
2. Particle distribution compared to quantum equilibrium
3. Evolution of the density matrix
4. Relative entropy matrix at different coarse-graining scales
5. H-function evolution with different coarse-graining factors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import time
from tqdm import tqdm
import argparse

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation
from src.visualization import plot_h_function


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced visualization for non-equilibrium Bohmian relaxation")
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
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def setup_simulation(args):
    """Set up the physical system and quantum state for simulation."""
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
    
    # Set up quantum states
    if args.pure_state:
        # For pure state simulation
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
    
    return simulation, system, quantum_state, dt, t_max


def run_relaxation_simulation(simulation):
    """Run the non-equilibrium relaxation simulation."""
    print("Generating non-equilibrium initial conditions via backward evolution...")
    start_time = time.time()
    simulation.generate_initial_conditions()
    
    print(f"Running forward simulation with {simulation.n_particles} particles...")
    results = simulation.run_simulation()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    return results


def plot_particle_trajectories(simulation, results, output_dir, sample_size=50):
    """
    Plot trajectories for a sample of particles.
    
    Args:
        simulation: BohmianRelaxation simulation object
        results: Dictionary of simulation results
        output_dir: Directory to save output
        sample_size: Number of particles to show (default: 50)
    """
    # Randomly select particles to display
    selected_particles = np.random.choice(
        simulation.n_particles, min(sample_size, simulation.n_particles), replace=False
    )
    
    # Set up a bigger figure for better visibility
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot trajectories with a colormap to indicate time progression
    positions = results["positions"]
    n_timesteps = results["n_timesteps"]
    time_values = np.linspace(0, results["t_max"], n_timesteps)
    
    # Color map for time evolution (darker = earlier, lighter = later)
    cmap = plt.cm.viridis
    
    # Plot each particle's trajectory
    for p_idx in selected_particles:
        x_traj = positions[p_idx, 0, :]
        y_traj = positions[p_idx, 1, :]
        
        # Create segments for coloring by time
        points = np.array([x_traj, y_traj]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a colored line collection
        lc = plt.matplotlib.collections.LineCollection(
            segments, cmap=cmap, norm=plt.Normalize(0, results["t_max"])
        )
        lc.set_array(time_values[:-1])
        lc.set_linewidth(1.0)
        ax.add_collection(lc)
        
        # Mark the starting point
        ax.plot(x_traj[0], y_traj[0], 'ro', markersize=3)
    
    # Add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, results["t_max"]), cmap=cmap), 
                       ax=ax, label='Time')
    
    # Set plot limits and labels
    ax.set_xlim(0, simulation.system.L)
    ax.set_ylim(0, simulation.system.L)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Bohmian Particle Trajectories (sample of {len(selected_particles)} particles)')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'particle_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def animate_particles_with_density_matrix(simulation, results, output_dir, sample_frames=100):
    """
    Create an animation showing particles evolving alongside the quantum density matrix.
    
    Args:
        simulation: BohmianRelaxation simulation object
        results: Dictionary of simulation results
        output_dir: Directory to save output
        sample_frames: Number of frames to include in the animation
    """
    positions = results["positions"]
    n_timesteps = results["n_timesteps"]
    dt = results["dt"]
    
    # Create frame indices to sample (ensure start and end frames are included)
    step_size = max(1, n_timesteps // sample_frames)
    frame_indices = list(range(0, n_timesteps, step_size))
    if (n_timesteps - 1) not in frame_indices:
        frame_indices.append(n_timesteps - 1)
    
    # Set up figure with three subplots: particles, density matrix, density difference
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), 
                                        gridspec_kw={'width_ratios': [1, 1, 1]})
    
    # Initialize plots
    # 1. Particle scatter plot
    x_pos = positions[:, 0, 0]
    y_pos = positions[:, 1, 0]
    scatter = ax1.scatter(x_pos, y_pos, s=2, c='blue', alpha=0.5)
    ax1.set_xlim(0, simulation.system.L)
    ax1.set_ylim(0, simulation.system.L)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Particle Positions')
    
    # 2. Quantum density matrix
    system = simulation.system
    quantum_state = simulation.quantum_state
    X, Y = np.meshgrid(
        np.linspace(0, system.L, 100),
        np.linspace(0, system.L, 100),
        indexing='ij'
    )
    
    if isinstance(quantum_state, MixedState):
        rho_eq = quantum_state.density_matrix_diagonal(X, Y, 0)
    else:
        rho_eq = quantum_state.probability_density(X, Y, 0)
    
    im2 = ax2.imshow(rho_eq.T, origin='lower', extent=[0, system.L, 0, system.L],
                    cmap='viridis', interpolation='nearest')
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2, label='Density Matrix')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Quantum Density Matrix')
    
    # 3. Particle histogram
    hist, _, _, im3 = ax3.hist2d(
        x_pos, y_pos, bins=50, 
        range=[[0, system.L], [0, system.L]],
        cmap='plasma', density=True
    )
    
    # Add colorbar for histogram
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax3, label='Particle Density')
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Particle Distribution')
    
    # Add a global title with time information
    title = plt.suptitle(f't = 0.00', fontsize=14)
    
    # Animation update function
    def update(frame_idx):
        frame = frame_indices[frame_idx]
        t = frame * dt
        
        # Update particle positions
        x_pos = positions[:, 0, frame]
        y_pos = positions[:, 1, frame]
        scatter.set_offsets(np.column_stack([x_pos, y_pos]))
        
        # Update quantum density matrix
        if isinstance(quantum_state, MixedState):
            rho_eq = quantum_state.density_matrix_diagonal(X, Y, t)
        else:
            rho_eq = quantum_state.probability_density(X, Y, t)
        im2.set_array(rho_eq.T)
        
        # Update particle histogram
        ax3.clear()
        hist, x_edges, y_edges, im3 = ax3.hist2d(
            x_pos, y_pos, bins=50, 
            range=[[0, system.L], [0, system.L]],
            cmap='plasma', density=True
        )
        
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('Particle Distribution')
        
        # Update time in global title
        title.set_text(f't = {t:.2f}')
        
        return [scatter, im2, im3, title]
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=100, blit=False
    )
    
    # Save animation
    print(f"Saving animation to {output_dir}/particles_and_density.mp4...")
    anim.save(os.path.join(output_dir, 'particles_and_density.mp4'), dpi=150, writer='ffmpeg')
    plt.close(fig)


def animate_relative_entropy(simulation, results, output_dir, coarse_graining_levels, sample_frames=100):
    """
    Create an animation showing the relative entropy matrix (H-function matrix) evolution
    at different coarse-graining levels.
    
    Args:
        simulation: BohmianRelaxation simulation object
        results: Dictionary of simulation results
        output_dir: Directory to save output
        coarse_graining_levels: List of coarse-graining levels to visualize
        sample_frames: Number of frames to include in the animation
    """
    n_timesteps = results["n_timesteps"]
    dt = results["dt"]
    
    # Create frame indices to sample (ensure start and end frames are included)
    step_size = max(1, n_timesteps // sample_frames)
    frame_indices = list(range(0, n_timesteps, step_size))
    if (n_timesteps - 1) not in frame_indices:
        frame_indices.append(n_timesteps - 1)
    
    # Set up figure with subplots for each coarse-graining level
    n_levels = len(coarse_graining_levels)
    fig, axs = plt.subplots(1, n_levels, figsize=(5*n_levels, 6))
    
    # For a single subplot, convert to a list for consistent indexing
    if n_levels == 1:
        axs = [axs]
    
    # Initialize plots for each coarse-graining level
    im_list = []
    for i, cg_level in enumerate(coarse_graining_levels):
        # Calculate initial H-function matrix
        h_matrix = simulation.calculate_h_matrix(0, coarse_grain=cg_level)
        
        # Determine colormap limits
        abs_max = max(abs(np.min(h_matrix)), abs(np.max(h_matrix)))
        vmin, vmax = -abs_max, abs_max
        
        # Plot the H-function matrix
        im = axs[i].imshow(h_matrix.T, origin='lower', 
                          extent=[0, simulation.system.L, 0, simulation.system.L],
                          cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # Add colorbar
        plt.colorbar(im, ax=axs[i], label=r'$\rho(\ln\rho - \ln W)$')
        
        # Add labels and title
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        axs[i].set_title(f'ε = {np.pi/cg_level:.4f}')
        
        im_list.append(im)
    
    # Add a global title with time information
    title = plt.suptitle(f't = 0.00', fontsize=14)
    
    # Animation update function
    def update(frame_idx):
        frame = frame_indices[frame_idx]
        t = frame * dt
        
        # Update each H-function matrix
        for i, cg_level in enumerate(coarse_graining_levels):
            h_matrix = simulation.calculate_h_matrix(frame, coarse_grain=cg_level)
            im_list[i].set_array(h_matrix.T)
        
        # Update time in global title
        title.set_text(f't = {t:.2f}')
        
        return im_list + [title]
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=100, blit=False
    )
    
    # Save animation
    print(f"Saving animation to {output_dir}/relative_entropy_evolution.mp4...")
    anim.save(os.path.join(output_dir, 'relative_entropy_evolution.mp4'), dpi=150, writer='ffmpeg')
    plt.close(fig)


def calculate_h_functions_for_coarse_graining(simulation, time_values, coarse_graining_levels):
    """
    Calculate H-functions for different coarse-graining levels.
    
    Args:
        simulation: BohmianRelaxation simulation object
        time_values: Array of time points
        coarse_graining_levels: List of coarse-graining levels
        
    Returns:
        Dictionary mapping coarse-graining levels to H-function arrays
    """
    h_values_dict = {}
    
    # Calculate H-function for each coarse-graining level
    for cg_level in tqdm(coarse_graining_levels, desc="Calculating H-functions"):
        h_values = simulation.calculate_h_function(coarse_grain=cg_level)
        h_values_dict[cg_level] = h_values
    
    return h_values_dict


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up and run the simulation
    simulation, system, quantum_state, dt, t_max = setup_simulation(args)
    results = run_relaxation_simulation(simulation)
    
    # Time values for plotting
    time_steps = results["n_timesteps"]
    time_values = np.linspace(0, t_max, time_steps)
    
    # Define coarse-graining levels
    coarse_graining_levels = [
        int(np.ceil(np.pi / (np.pi/8))),   # ε = π/8
        int(np.ceil(np.pi / (np.pi/16))),  # ε = π/16
        int(np.ceil(np.pi / (np.pi/32))),  # ε = π/32
        int(np.ceil(np.pi / (np.pi/64)))   # ε = π/64
    ]
    
    print("Generating visualizations...")
    
    # 1. Plot particle trajectories
    plot_particle_trajectories(simulation, results, args.output_dir)
    
    # 2. Animate particles with density matrix
    animate_particles_with_density_matrix(simulation, results, args.output_dir)
    
    # 3. Animate relative entropy matrix at different coarse-graining levels
    animate_relative_entropy(simulation, results, args.output_dir, coarse_graining_levels)
    
    # 4. Calculate and plot H-functions for different coarse-graining levels
    print("Calculating H-functions for different coarse-graining levels...")
    h_values_dict = calculate_h_functions_for_coarse_graining(
        simulation, time_values, coarse_graining_levels
    )
    
    # Calculate H-function without coarse-graining (base case)
    h_values = simulation.calculate_h_function()
    
    # Plot H-function evolution
    print("Plotting H-function evolution...")
    plot_h_function(
        time_values, h_values, h_values_dict, 
        title="H-Function Evolution with Different Coarse-Graining Levels",
        save_path=os.path.join(args.output_dir, 'h_function_evolution.png')
    )
    
    print("All visualizations completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
