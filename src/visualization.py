#!/usr/bin/env python3
"""
Module for visualizing Bohmian relaxation simulation results.

This includes functions for plotting distributions, H-functions, and generating animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from typing import Dict, List, Optional, Tuple, Union

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation


def plot_wavefunction(pure_state: PureState, t: float, title: Optional[str] = None, 
                     save_path: Optional[str] = None) -> plt.Figure:
    """Plot the probability density of a pure quantum state.
    
    Args:
        pure_state: Pure quantum state to visualize
        t: Time at which to evaluate the wavefunction
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Get system grid
    X, Y = pure_state.system.X, pure_state.system.Y
    
    # Calculate probability density
    prob_density = pure_state.probability_density(X, Y, t)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot probability density
    im = ax.pcolormesh(X, Y, prob_density, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='Probability density $|\psi|^2$')
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title is None:
        title = f'Probability Density at t = {t:.2f}'
    ax.set_title(title)
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_mixed_state(mixed_state: MixedState, t: float, title: Optional[str] = None, 
                    save_path: Optional[str] = None) -> plt.Figure:
    """Plot the diagonal elements of a mixed quantum state density matrix.
    
    Args:
        mixed_state: Mixed quantum state to visualize
        t: Time at which to evaluate the density matrix
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Get system grid
    X, Y = mixed_state.system.X, mixed_state.system.Y
    
    # Calculate diagonal elements of the density matrix
    rho_diag = mixed_state.density_matrix_diagonal(X, Y, t)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot diagonal density matrix elements
    im = ax.pcolormesh(X, Y, rho_diag, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='Probability density $\\rho(x,y)$')
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title is None:
        title = f'Mixed State Probability Density at t = {t:.2f}'
    ax.set_title(title)
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_particle_distribution(simulation: BohmianRelaxation, timestep: int, 
                              bins: int = 50, title: Optional[str] = None, 
                              save_path: Optional[str] = None) -> plt.Figure:
    """Plot the histogram of particle positions at a given timestep.
    
    Args:
        simulation: Bohmian relaxation simulation object
        timestep: Index of the timestep to visualize
        bins: Number of bins in each dimension for the histogram
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Extract particle positions at the given timestep
    x_positions = simulation.positions[:, 0, timestep]
    y_positions = simulation.positions[:, 1, timestep]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot 2D histogram
    hist, x_edges, y_edges, im = ax.hist2d(
        x_positions, y_positions, 
        bins=bins, 
        range=[[0, simulation.system.L], [0, simulation.system.L]],
        cmap='viridis',
        density=True
    )
    plt.colorbar(im, ax=ax, label='Normalized particle density')
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title is None:
        t = timestep * simulation.dt
        title = f'Particle Distribution at t = {t:.2f}'
    ax.set_title(title)
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_h_function(time_values: np.ndarray, h_values: np.ndarray, 
                   h_coarse_values: Optional[Dict[int, np.ndarray]] = None, 
                   title: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
    """Plot the H-function evolution over time.
    
    Args:
        time_values: Array of time points
        h_values: Array of H-function values
        h_coarse_values: Optional dictionary mapping coarse-graining levels to H-function arrays
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot main H-function
    ax.plot(time_values, h_values, 'k-', linewidth=2, label='$H(t)$')
    
    # Plot coarse-grained variants if provided
    if h_coarse_values is not None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(h_coarse_values)))
        for i, (cg_level, h_cg) in enumerate(h_coarse_values.items()):
            ax.plot(time_values, h_cg, '-', color=colors[i], linewidth=1.5, 
                   label=f'$H_{{{cg_level}}}(t)$ (CG={cg_level})')
    
    # Normalize to initial value
    if np.abs(h_values[0]) > 1e-10:  # Avoid division by zero
        ax_twin = ax.twinx()
        ax_twin.plot(time_values, h_values / h_values[0], 'r--', linewidth=1.5, 
                   label='$H(t)/H(0)$')
        ax_twin.set_ylabel('Normalized H-function $H(t)/H(0)$', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax_twin.set_ylim(0, 1.05)
        
        # Add both legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax.legend(loc='best')
    
    # Add labels and title
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('H-function $H(t)$')
    if title is None:
        title = 'Evolution of the Quantum H-Function'
    ax.set_title(title)
    
    # Use logarithmic scale for y-axis if values span multiple orders of magnitude
    if np.max(h_values) / (np.min(h_values) + 1e-10) > 100:
        ax.set_yscale('log')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout and save if requested
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_h_matrix(simulation: BohmianRelaxation, timestep: int, coarse_grain: Optional[int] = None, 
                 title: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
    """Plot the H-function matrix at a given timestep.
    
    Args:
        simulation: Bohmian relaxation simulation object
        timestep: Index of the timestep to visualize
        coarse_grain: Optional number of bins for coarse-graining
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Calculate the H-function matrix
    h_matrix = simulation.calculate_h_matrix(timestep, coarse_grain=coarse_grain)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine vmin, vmax for better visualization
    # We want to see both positive and negative values
    abs_max = max(abs(np.min(h_matrix)), abs(np.max(h_matrix)))
    vmin, vmax = -abs_max, abs_max
    
    # Plot H-function matrix
    im = ax.imshow(h_matrix.T, origin='lower', extent=[0, simulation.system.L, 0, simulation.system.L],
                  cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='$\rho(\ln\rho - \ln W)$')
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title is None:
        t = timestep * simulation.dt
        cg_text = f", CG={coarse_grain}" if coarse_grain is not None else ""
        title = f'H-Function Matrix at t = {t:.2f}{cg_text}'
    ax.set_title(title)
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def animate_evolution(simulation: BohmianRelaxation, results: Dict, 
                     interval: int = 100, save_path: Optional[str] = None) -> animation.Animation:
    """Create an animation of the particle distribution evolution.
    
    Args:
        simulation: Bohmian relaxation simulation object
        results: Dictionary of simulation results
        interval: Time interval between frames in milliseconds
        save_path: Optional path to save the animation
        
    Returns:
        Matplotlib animation object
    """
    # Extract necessary data
    positions = results["positions"]
    dt = results["dt"]
    n_timesteps = results["n_timesteps"]
    
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Initial histogram for particle distribution
    x_positions = positions[:, 0, 0]
    y_positions = positions[:, 1, 0]
    hist, x_edges, y_edges, im1 = ax1.hist2d(
        x_positions, y_positions, 
        bins=50, 
        range=[[0, simulation.system.L], [0, simulation.system.L]],
        cmap='viridis',
        density=True
    )
    cb1 = plt.colorbar(im1, ax=ax1, label='Particle density')
    
    # Initial H-function matrix
    h_matrix = simulation.calculate_h_matrix(0, coarse_grain=30)
    abs_max = max(abs(np.min(h_matrix)), abs(np.max(h_matrix)))
    vmin, vmax = -abs_max, abs_max
    
    im2 = ax2.imshow(h_matrix.T, origin='lower', 
                    extent=[0, simulation.system.L, 0, simulation.system.L],
                    cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
    cb2 = plt.colorbar(im2, ax=ax2, label='$\rho(\ln\rho - \ln W)$')
    
    # Set labels and titles
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Particle Distribution')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('H-Function Matrix')
    
    # Add a global title with time information
    title = plt.suptitle(f't = 0.00', fontsize=14)
    
    # Animation update function
    def update(frame):
        # Update particle distribution
        ax1.clear()
        x_pos = positions[:, 0, frame]
        y_pos = positions[:, 1, frame]
        hist, x_edges, y_edges, im1 = ax1.hist2d(
            x_pos, y_pos, 
            bins=50, 
            range=[[0, simulation.system.L], [0, simulation.system.L]],
            cmap='viridis',
            density=True
        )
        
        # Update H-function matrix
        h_matrix = simulation.calculate_h_matrix(frame, coarse_grain=30)
        im2.set_data(h_matrix.T)
        
        # Update labels and titles
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Particle Distribution')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('H-Function Matrix')
        
        # Update time in global title
        t = frame * dt
        title.set_text(f't = {t:.2f}')
        
        return [im1, im2, title]
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=range(0, n_timesteps, max(1, n_timesteps//100)), 
                                 interval=interval, blit=False)
    
    # Save if requested
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, dpi=150, writer='ffmpeg')
    
    return anim


def compare_pure_and_mixed(pure_simulation: BohmianRelaxation, mixed_simulation: BohmianRelaxation,
                         time_values: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
    """Compare the H-function evolution for pure and mixed states.
    
    Args:
        pure_simulation: Bohmian relaxation simulation with pure state
        mixed_simulation: Bohmian relaxation simulation with mixed state
        time_values: Array of time points
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Calculate H-functions
    h_pure = pure_simulation.calculate_h_function()
    h_mixed = mixed_simulation.calculate_h_function()
    
    # Normalize to initial values
    h_pure_norm = h_pure / h_pure[0] if abs(h_pure[0]) > 1e-10 else h_pure
    h_mixed_norm = h_mixed / h_mixed[0] if abs(h_mixed[0]) > 1e-10 else h_mixed
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot absolute H-functions
    ax1.plot(time_values, h_pure, 'b-', linewidth=2, label='Pure State')
    ax1.plot(time_values, h_mixed, 'r-', linewidth=2, label='Mixed State')
    
    # Plot normalized H-functions
    ax2.plot(time_values, h_pure_norm, 'b-', linewidth=2, label='Pure State')
    ax2.plot(time_values, h_mixed_norm, 'r-', linewidth=2, label='Mixed State')
    
    # Add horizontal line at 10% level for reference
    ax2.axhline(y=0.1, color='k', linestyle='--', alpha=0.5, label='10% Residual')
    
    # Add labels, titles, and legends
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('H-function $H(t)$')
    ax1.set_title('Absolute H-Function')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Normalized H-function $H(t)/H(0)$')
    ax2.set_title('Normalized H-Function')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 1.05)
    
    # Add overall title
    plt.suptitle('Comparison of Pure and Mixed State Relaxation', fontsize=16)
    
    # Adjust layout and save if requested
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
