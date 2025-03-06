#!/usr/bin/env python3
"""
Module for visualizing the von Neumann guidance equation for mixed states.

This extends the basic visualization functionality to specifically focus on
the density matrix guidance and coarse-grained relaxation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, List, Tuple, Union, Optional

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation


class VonNeumannRelaxationVisualizer:
    """Specialized visualizer for von Neumann mixed-state relaxation.
    
    This class provides visualization tools specifically designed for
    analyzing relaxation in mixed-state Bohmian mechanics according to the
    von Neumann guidance equation.
    
    Attributes:
        simulation: BohmianRelaxation simulation object
        results: Dictionary of simulation results
        coarse_graining_levels: List of coarse-graining levels to analyze
    """
    
    def __init__(self, simulation: BohmianRelaxation, results: Dict, 
                 coarse_graining_levels: Optional[List[int]] = None):
        """Initialize the visualizer.
        
        Args:
            simulation: BohmianRelaxation simulation object
            results: Dictionary of simulation results from running the simulation
            coarse_graining_levels: Optional list of coarse-graining levels to analyze
        """
        self.simulation = simulation
        self.results = results
        
        # Set default coarse-graining levels if not provided
        if coarse_graining_levels is None:
            # Convert ε values to bin counts (ε = π/bins)
            self.coarse_graining_levels = [
                int(np.ceil(np.pi / (np.pi/8))),   # ε = π/8
                int(np.ceil(np.pi / (np.pi/16))),  # ε = π/16
                int(np.ceil(np.pi / (np.pi/32))),  # ε = π/32
                int(np.ceil(np.pi / (np.pi/64)))   # ε = π/64
            ]
        else:
            self.coarse_graining_levels = coarse_graining_levels
            
        # Extract system parameters for convenience
        self.system = simulation.system
        self.quantum_state = simulation.quantum_state
        self.n_particles = simulation.n_particles
        self.dt = results["dt"]
        self.t_max = results["t_max"]
        self.n_timesteps = results["n_timesteps"]
        self.positions = results["positions"]
        
        # Grid for density matrix evaluation
        self.X, self.Y = np.meshgrid(
            np.linspace(0, self.system.L, 100),
            np.linspace(0, self.system.L, 100),
            indexing='ij'
        )
        
        # Time array
        self.time_values = np.linspace(0, self.t_max, self.n_timesteps)
    
    def create_density_matrix_comparison(self, timestep: int, save_path: Optional[str] = None) -> plt.Figure:
        """Compare the actual particle distribution with the quantum density matrix.
        
        Args:
            timestep: Index of the timestep to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Calculate actual time
        t = timestep * self.dt
        
        # Get particle positions at this timestep
        x_pos = self.positions[:, 0, timestep]
        y_pos = self.positions[:, 1, timestep]
        
        # Calculate theoretical quantum density matrix
        if isinstance(self.quantum_state, MixedState):
            rho_theory = self.quantum_state.density_matrix_diagonal(self.X, self.Y, t)
        else:
            rho_theory = self.quantum_state.probability_density(self.X, self.Y, t)
        
        # Create figure with 2 rows, 3 columns for in-depth comparison
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), 
                                gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 0.8]})
        
        # Row 1: Visualizations
        # 1. Particle scatter plot
        axs[0, 0].scatter(x_pos, y_pos, s=2, c='blue', alpha=0.4)
        axs[0, 0].set_xlim(0, self.system.L)
        axs[0, 0].set_ylim(0, self.system.L)
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')
        axs[0, 0].set_title('Particle Positions')
        
        # 2. Particle density histogram
        bins = 50
        hist, x_edges, y_edges, im2 = axs[0, 1].hist2d(
            x_pos, y_pos, bins=bins,
            range=[[0, self.system.L], [0, self.system.L]],
            cmap='viridis', density=True
        )
        plt.colorbar(im2, ax=axs[0, 1], label='Particle Density')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')
        axs[0, 1].set_title('Particle Density Histogram')
        
        # 3. Theoretical density matrix
        im3 = axs[0, 2].imshow(rho_theory.T, origin='lower', 
                               extent=[0, self.system.L, 0, self.system.L],
                               cmap='viridis', interpolation='nearest')
        plt.colorbar(im3, ax=axs[0, 2], label='Density Matrix')
        axs[0, 2].set_xlabel('x')
        axs[0, 2].set_ylabel('y')
        axs[0, 2].set_title('Quantum Density Matrix')
        
        # Row 2: Analysis
        # 1. Difference between histogram and theory
        # First, interpolate the histogram to match the theoretical grid
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X_hist, Y_hist = np.meshgrid(x_centers, y_centers, indexing='ij')
        
        # Ensure proper normalization for comparison
        rho_theory_norm = rho_theory / np.sum(rho_theory) * np.sum(hist)
        
        # Calculate difference and relative difference
        # Interpolate histogram to match theory grid
        from scipy.interpolate import RegularGridInterpolator
        interp_hist = RegularGridInterpolator(
            (x_centers, y_centers), hist, 
            bounds_error=False, fill_value=0
        )
        
        # Get histogram values on the same grid as theory
        points = np.column_stack((self.X.flatten(), self.Y.flatten()))
        hist_on_theory_grid = interp_hist(points).reshape(self.X.shape)
        
        # Calculate absolute difference
        diff = hist_on_theory_grid - rho_theory_norm
        
        # Plot absolute difference
        vmax = max(abs(np.min(diff)), abs(np.max(diff)))
        im4 = axs[1, 0].imshow(diff.T, origin='lower',
                               extent=[0, self.system.L, 0, self.system.L],
                               cmap='RdBu_r', interpolation='nearest',
                               vmin=-vmax, vmax=vmax)
        plt.colorbar(im4, ax=axs[1, 0], label='Absolute Difference')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('y')
        axs[1, 0].set_title('Density Difference (Actual - Theory)')
        
        # 2. H-function matrix (relative entropy density)
        h_matrix = self.simulation.calculate_h_matrix(timestep, coarse_grain=30)  # Moderate coarse-graining
        vmax = max(abs(np.min(h_matrix)), abs(np.max(h_matrix)))
        im5 = axs[1, 1].imshow(h_matrix.T, origin='lower',
                               extent=[0, self.system.L, 0, self.system.L],
                               cmap='RdBu_r', interpolation='nearest',
                               vmin=-vmax, vmax=vmax)
        plt.colorbar(im5, ax=axs[1, 1], label=r'$\rho(\ln\rho - \ln W)$')
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('y')
        axs[1, 1].set_title('H-Function Matrix (Relative Entropy Density)')
        
        # 3. H-function values for different coarse-graining levels
        h_values = []
        epsilon_values = []
        for cg_level in self.coarse_graining_levels:
            h_val = self.simulation.calculate_h_function(timestep, coarse_grain=cg_level)[0]
            h_values.append(h_val)
            epsilon_values.append(np.pi / cg_level)
        
        axs[1, 2].plot(epsilon_values, h_values, 'o-', color='blue', linewidth=2, markersize=8)
        axs[1, 2].set_xlabel('Coarse-Graining Parameter ε')
        axs[1, 2].set_ylabel('H-Function Value')
        axs[1, 2].set_title('H-Function vs. Coarse-Graining Level')
        axs[1, 2].grid(True, linestyle='--', alpha=0.7)
        
        # Use log scale if values span multiple orders of magnitude
        if max(h_values) / (min(h_values) + 1e-10) > 10:
            axs[1, 2].set_yscale('log')
        
        # Add a global title
        plt.suptitle(f'Mixed State Relaxation Analysis at t = {t:.3f}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_velocity_field_visualization(self, timestep: int, 
                                           grid_size: int = 20,
                                           save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the Bohmian velocity field along with particles.
        
        Args:
            timestep: Index of the timestep to visualize
            grid_size: Number of grid points in each dimension for velocity field
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Calculate actual time
        t = timestep * self.dt
        
        # Get particle positions at this timestep
        x_pos = self.positions[:, 0, timestep]
        y_pos = self.positions[:, 1, timestep]
        
        # Create a grid for the velocity field
        x_grid = np.linspace(0, self.system.L, grid_size)
        y_grid = np.linspace(0, self.system.L, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Calculate velocity field
        vx, vy = self.system.velocity_field(self.quantum_state, t, X_grid, Y_grid)
        
        # Calculate velocity magnitude for coloring
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot particles
        ax.scatter(x_pos, y_pos, s=3, c='blue', alpha=0.5, label='Particles')
        
        # Plot velocity field
        quiver = ax.quiver(X_grid, Y_grid, vx, vy, v_mag,
                          cmap='viridis', scale=20, width=0.003)
        plt.colorbar(quiver, ax=ax, label='Velocity Magnitude')
        
        # Set plot limits and labels
        ax.set_xlim(0, self.system.L)
        ax.set_ylim(0, self.system.L)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Bohmian Velocity Field and Particles at t = {t:.3f}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def animate_coarse_grained_relaxation(self, output_path: str, sample_frames: int = 100):
        """Create an animation showing relaxation at different coarse-graining levels.
        
        Args:
            output_path: Path to save the animation
            sample_frames: Number of frames to include in the animation
        """
        # Select frames to include in the animation
        step_size = max(1, self.n_timesteps // sample_frames)
        frame_indices = list(range(0, self.n_timesteps, step_size))
        if (self.n_timesteps - 1) not in frame_indices:
            frame_indices.append(self.n_timesteps - 1)
        
        # Set up figure with subplots for each coarse-graining level
        n_levels = len(self.coarse_graining_levels)
        fig, axs = plt.subplots(2, n_levels, figsize=(4*n_levels, 8))
        
        # Initialize plots for each coarse-graining level
        im_list = []
        h_values = [[] for _ in range(n_levels)]
        time_points = []
        lines = []
        
        # Row 1: H-function matrices
        for i, cg_level in enumerate(self.coarse_graining_levels):
            # Calculate initial H-function matrix
            h_matrix = self.simulation.calculate_h_matrix(0, coarse_grain=cg_level)
            
            # Determine colormap limits
            abs_max = max(abs(np.min(h_matrix)), abs(np.max(h_matrix)))
            vmin, vmax = -abs_max, abs_max
            
            # Plot the H-function matrix
            im = axs[0, i].imshow(h_matrix.T, origin='lower', 
                                 extent=[0, self.system.L, 0, self.system.L],
                                 cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
            
            # Add colorbar
            plt.colorbar(im, ax=axs[0, i], label=r'$\rho(\ln\rho - \ln W)$')
            
            # Add labels and title
            axs[0, i].set_xlabel('x')
            axs[0, i].set_ylabel('y')
            eps_value = np.pi / cg_level
            axs[0, i].set_title(f'ε = {eps_value:.4f}')
            
            im_list.append(im)
            
            # Initialize H-function plot
            line, = axs[1, i].plot([], [], 'b-', linewidth=2)
            lines.append(line)
            axs[1, i].set_xlabel('Time')
            axs[1, i].set_ylabel('H-function')
            axs[1, i].set_title(f'H-function Evolution (ε = {eps_value:.4f})')
            axs[1, i].grid(True, linestyle='--', alpha=0.7)
        
        # Add a global title with time information
        title = plt.suptitle(f't = 0.00', fontsize=14)
        
        # Precalculate H-functions for all timesteps and coarse-graining levels
        print("Precalculating H-function values...")
        for frame in frame_indices:
            t = frame * self.dt
            time_points.append(t)
            
            for i, cg_level in enumerate(self.coarse_graining_levels):
                h_val = self.simulation.calculate_h_function(frame, coarse_grain=cg_level)[0]
                h_values[i].append(h_val)
        
        # Set axis limits for H-function plots
        for i in range(n_levels):
            y_min = min(h_values[i])
            y_max = max(h_values[i])
            margin = 0.1 * (y_max - y_min)
            axs[1, i].set_xlim(0, max(time_points))
            axs[1, i].set_ylim(y_min - margin, y_max + margin)
        
        # Animation update function
        def update(frame_idx):
            frame = frame_indices[frame_idx]
            t = frame * self.dt
            
            # Update each H-function matrix
            for i, cg_level in enumerate(self.coarse_graining_levels):
                h_matrix = self.simulation.calculate_h_matrix(frame, coarse_grain=cg_level)
                im_list[i].set_array(h_matrix.T)
                
                # Update H-function plot
                lines[i].set_data(time_points[:frame_idx+1], h_values[i][:frame_idx+1])
            
            # Update time in global title
            title.set_text(f't = {t:.2f}')
            
            return im_list + lines + [title]
        
        # Create the animation
        print(f"Creating animation with {len(frame_indices)} frames...")
        anim = animation.FuncAnimation(
            fig, update, frames=len(frame_indices), interval=200, blit=False
        )
        
        # Save animation
        print(f"Saving animation to {output_path}...")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
        anim.save(output_path, dpi=150, writer='ffmpeg')
        plt.close(fig)
        
        print("Animation saved successfully!")
    
    def generate_coarse_graining_analysis(self, timesteps: List[int], output_dir: str):
        """Generate comprehensive coarse-graining analysis for specific timesteps.
        
        Args:
            timesteps: List of timestep indices to analyze
            output_dir: Directory to save output files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating coarse-graining analysis...")
        
        # Analyze each specified timestep
        for ts in timesteps:
            t = ts * self.dt
            print(f"Processing timestep {ts} (t = {t:.3f})...")
            
            # 1. Create density matrix comparison
            save_path = os.path.join(output_dir, f'density_comparison_t{ts:04d}.png')
            self.create_density_matrix_comparison(ts, save_path=save_path)
            
            # 2. Create velocity field visualization
            save_path = os.path.join(output_dir, f'velocity_field_t{ts:04d}.png')
            self.create_velocity_field_visualization(ts, save_path=save_path)
            
            # 3. Create detailed coarse-graining sweep
            # Use more levels for the detailed analysis
            detailed_cg_levels = np.logspace(np.log10(4), np.log10(100), 20).astype(int)
            detailed_cg_levels = np.unique(detailed_cg_levels)  # Remove duplicates
            
            # Calculate H-function for each level
            h_values = []
            epsilon_values = []
            
            for cg_level in detailed_cg_levels:
                h_val = self.simulation.calculate_h_function(ts, coarse_grain=cg_level)[0]
                h_values.append(h_val)
                epsilon_values.append(np.pi / cg_level)
            
            # Plot H-function vs. epsilon
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epsilon_values, h_values, 'o-', color='blue', linewidth=2, markersize=6)
            ax.set_xlabel('Coarse-Graining Parameter ε')
            ax.set_ylabel('H-Function Value')
            ax.set_title(f'H-Function vs. Coarse-Graining Level at t = {t:.3f}')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add logarithmic x-axis for better visualization
            ax.set_xscale('log')
            
            # Use log scale for y-axis if values span multiple orders of magnitude
            if max(h_values) / (min(h_values) + 1e-10) > 10:
                ax.set_yscale('log')
            
            # Save figure
            save_path = os.path.join(output_dir, f'coarse_graining_sweep_t{ts:04d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print("Coarse-graining analysis completed!")
