#!/usr/bin/env python3
"""
Script to perform comprehensive coarse-graining studies for the von Neumann relaxation.

This script runs multiple simulations with various coarse-graining parameters
and analyzes how relaxation behavior changes with the coarse-graining scale.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import subprocess
import multiprocessing
from tqdm import tqdm

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation
from src.von_neumann_visualization import VonNeumannRelaxationVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run coarse-graining study")
    parser.add_argument(
        "--particles", type=int, default=5000, 
        help="Number of particles to simulate (default: 5000)"
    )
    parser.add_argument(
        "--tmax", type=float, default=2.0, 
        help="Maximum simulation time in box periods (default: 2.0)"
    )
    parser.add_argument(
        "--parallel", action="store_true", 
        help="Run simulations in parallel"
    )
    parser.add_argument(
        "--output-dir", type=str, default="coarse_graining_study", 
        help="Output directory for results (default: coarse_graining_study)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=None, 
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def run_single_simulation(epsilon_value, args, base_output_dir):
    """Run a single simulation with a specific coarse-graining parameter.
    
    Args:
        epsilon_value: Coarse-graining parameter value (π/n)
        args: Command line arguments
        base_output_dir: Base output directory
        
    Returns:
        Path to output directory for this simulation
    """
    # Convert epsilon to bins: ε = π/bins
    n_value = int(np.pi / epsilon_value)
    output_dir = os.path.join(base_output_dir, f"epsilon_pi_over_{n_value}")
    
    # Prepare command line arguments
    cmd = [
        "python", "run_von_neumann_simulation.py",
        "--particles", str(args.particles),
        "--tmax", str(args.tmax),
        "--output-dir", output_dir
    ]
    
    # Add random seed if specified
    if args.random_seed is not None:
        cmd.extend(["--seed", str(args.random_seed)])
    
    # Run the simulation
    print(f"Running simulation with ε = π/{n_value}...")
    subprocess.run(cmd, check=True)
    
    return output_dir


def plot_combined_h_functions(output_dirs, epsilon_values, combined_output_dir):
    """Plot combined H-functions from multiple simulations.
    
    Args:
        output_dirs: List of output directories from simulations
        epsilon_values: List of epsilon values used
        combined_output_dir: Directory to save combined plots
    """
    # Create directory for combined plots
    os.makedirs(combined_output_dir, exist_ok=True)
    
    # Extract H-function data from each simulation
    h_data = {}
    time_values = None
    
    for i, (output_dir, epsilon) in enumerate(zip(output_dirs, epsilon_values)):
        # Find the H-function file
        h_function_file = os.path.join(output_dir, 'h_function_evolution.npy')
        
        if os.path.exists(h_function_file):
            # Load data
            data = np.load(h_function_file, allow_pickle=True).item()
            h_values = data['h_values']
            if time_values is None:
                time_values = data['time_values']
            
            # Store data
            h_data[f"ε = π/{int(np.pi/epsilon)}"] = h_values
    
    if time_values is not None and h_data:
        # Plot combined H-functions
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use a color gradient
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(h_data)))
        
        for i, (label, h_values) in enumerate(h_data.items()):
            ax.plot(time_values, h_values, '-', color=colors[i], linewidth=2, label=label)
        
        # Add labels and title
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('H-Function', fontsize=12)
        ax.set_title('H-Function Evolution for Different Coarse-Graining Levels', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Use log scale if values span multiple orders of magnitude
        all_values = np.concatenate([h for h in h_data.values()])
        if max(all_values) / (min(all_values) + 1e-10) > 10:
            ax.set_yscale('log')
        
        # Save figure
        save_path = os.path.join(combined_output_dir, 'combined_h_functions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Also create a plot showing convergence rate vs. epsilon
    if h_data:
        # Calculate convergence rate for each epsilon
        convergence_rates = {}
        for label, h_values in h_data.items():
            # Calculate the rate as slope of log(H) vs. time
            # Use the middle portion of the curve to avoid initial transients
            middle_start = len(h_values) // 4
            middle_end = 3 * len(h_values) // 4
            
            if h_values[middle_start] > 0 and h_values[middle_end] > 0:
                log_h_start = np.log(h_values[middle_start])
                log_h_end = np.log(h_values[middle_end])
                time_start = time_values[middle_start]
                time_end = time_values[middle_end]
                
                rate = (log_h_end - log_h_start) / (time_end - time_start)
                convergence_rates[label] = -rate  # Negate so positive is faster convergence
        
        # Extract epsilon values from labels
        if convergence_rates:
            epsilon_labels = list(convergence_rates.keys())
            epsilon_values = [np.pi / int(label.split('/')[-1]) for label in epsilon_labels]
            rates = [convergence_rates[label] for label in epsilon_labels]
            
            # Sort by epsilon
            sorted_indices = np.argsort(epsilon_values)
            epsilon_values = [epsilon_values[i] for i in sorted_indices]
            rates = [rates[i] for i in sorted_indices]
            
            # Plot convergence rate vs. epsilon
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epsilon_values, rates, 'o-', color='blue', linewidth=2, markersize=8)
            
            # Add labels and title
            ax.set_xlabel('Coarse-Graining Parameter ε', fontsize=12)
            ax.set_ylabel('Convergence Rate (-d/dt log H)', fontsize=12)
            ax.set_title('Relaxation Rate vs. Coarse-Graining Level', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Use log-log scale
            ax.set_xscale('log')
            
            # Save figure
            save_path = os.path.join(combined_output_dir, 'convergence_rate_vs_epsilon.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed if specified
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    # Define coarse-graining parameters to study
    # Use a range of epsilon values from π/4 to π/128
    n_values = [4, 8, 16, 32, 64, 128]
    epsilon_values = [np.pi / n for n in n_values]
    
    # Run simulations
    output_dirs = []
    
    if args.parallel and len(epsilon_values) > 1:
        # Run simulations in parallel
        pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(epsilon_values)))
        
        # Prepare arguments for each simulation
        sim_args = [(epsilon, args, args.output_dir) for epsilon in epsilon_values]
        
        # Run simulations in parallel
        output_dirs = pool.starmap(run_single_simulation, sim_args)
        pool.close()
        pool.join()
    else:
        # Run simulations sequentially
        for epsilon in epsilon_values:
            output_dir = run_single_simulation(epsilon, args, args.output_dir)
            output_dirs.append(output_dir)
    
    # Analyze and plot combined results
    print("\nAnalyzing combined results...")
    plot_combined_h_functions(output_dirs, epsilon_values, args.output_dir)
    
    print("\nAll simulations and analysis completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
