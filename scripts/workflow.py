#!/usr/bin/env python3
"""
Workflow script for Bohmian mixed-state relaxation simulation.

This script provides a complete workflow for:
1. Cloning the repository
2. Setting up the environment
3. Running tests
4. Running simulations
5. Generating visualizations
"""

import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'numpy', 
        'matplotlib', 
        'scipy', 
        'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("All required packages installed.")

def clone_repository(repo_url, target_dir=None):
    """Clone the GitHub repository."""
    if target_dir is None:
        target_dir = 'bohmian-mixed-state-relaxation'
    
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists. Skipping clone.")
        return target_dir
    
    print(f"Cloning repository from {repo_url}...")
    subprocess.check_call(['git', 'clone', repo_url, target_dir])
    print(f"Repository cloned to {target_dir}")
    
    return target_dir

def run_tests(repo_dir, test_type='all', verbose=True):
    """Run the test suite."""
    original_dir = os.getcwd()
    os.chdir(repo_dir)
    
    try:
        print(f"Running {test_type} tests...")
        cmd = [sys.executable, '-m', 'tests.run_tests', '--type', test_type]
        
        if not verbose:
            cmd.append('--quiet')
        
        result = subprocess.run(cmd, capture_output=not verbose)
        success = result.returncode == 0
        
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed.")
            if not verbose:
                print(result.stdout.decode('utf-8'))
                print(result.stderr.decode('utf-8'))
        
        return success
    
    finally:
        os.chdir(original_dir)

def run_simulation(repo_dir, output_dir='output', simulation_type='mixed'):
    """Run a Bohmian mechanics simulation."""
    original_dir = os.getcwd()
    os.chdir(repo_dir)
    
    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Add repo directory to the Python path
        sys.path.insert(0, os.getcwd())
        
        # Import simulation components
        from src.system import InfiniteSquareWell2D
        from src.quantum_state import PureState, MixedState
        from src.relaxation import BohmianRelaxation
        from src.von_neumann_visualization import VonNeumannRelaxationVisualizer
        
        print(f"Running {simulation_type} state simulation...")
        
        # Create system
        L = 1.0  # Box size
        system = InfiniteSquareWell2D(L)
        
        # Create quantum state
        if simulation_type == 'pure':
            # Superposition of ground and excited states
            quantum_state = PureState(L, [
                (1, 1, 1/np.sqrt(2)),
                (2, 1, 1/np.sqrt(2))
            ])
            print("Created pure state (superposition of ground and excited states)")
        else:
            # Mixed state of two pure states
            state1 = PureState(L, [(1, 1, 1.0)])
            state2 = PureState(L, [(2, 1, 1.0)])
            quantum_state = MixedState(L, [
                (state1, 0.7),
                (state2, 0.3)
            ])
            print("Created mixed state (70% ground state, 30% excited state)")
        
        # Create relaxation simulation
        n_particles = 1000
        relaxation = BohmianRelaxation(system, quantum_state, n_particles)
        print(f"Initialized simulation with {n_particles} particles")
        
        # Run simulation
        t_max = 2.0
        dt = 0.05
        print(f"Running simulation from t=0 to t={t_max} with dt={dt}...")
        results = relaxation.run_simulation(t_max, dt)
        print("Simulation completed!")
        
        # Create visualizer
        visualizer = VonNeumannRelaxationVisualizer(relaxation, results)
        
        # Generate visualizations
        print("Generating visualizations...")
        
        # Density matrix comparison at different times
        timesteps = [0, int(len(results['positions'][0, 0]) // 4), int(len(results['positions'][0, 0]) // 2), -1]
        for i, ts in enumerate(timesteps):
            t = ts * dt
            print(f"Creating density visualization for t={t:.2f}...")
            fig = visualizer.create_density_matrix_comparison(ts)
            fig.savefig(output_path / f'density_comparison_t{i}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        # Velocity field visualization
        print("Creating velocity field visualization...")
        fig = visualizer.create_velocity_field_visualization(timesteps[1])
        fig.savefig(output_path / 'velocity_field.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # H-function evolution
        print("Calculating and plotting H-function evolution...")
        visualizer.calculate_and_save_h_functions(str(output_path))
        
        print(f"All visualizations saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error running simulation: {e}")
        return False
    
    finally:
        os.chdir(original_dir)

def main():
    """Main function to parse arguments and run the workflow."""
    parser = argparse.ArgumentParser(description='Bohmian Mixed-State Relaxation Workflow')
    parser.add_argument('--repo-url', type=str, 
                      default='https://github.com/cameron-cognitive/bohmian-mixed-state-relaxation.git',
                      help='URL of the GitHub repository')
    parser.add_argument('--target-dir', type=str, default=None,
                      help='Target directory for the cloned repository')
    parser.add_argument('--skip-tests', action='store_true',
                      help='Skip running tests')
    parser.add_argument('--test-type', choices=['basic', 'velocity', 'all'], default='all',
                      help='Type of tests to run (default: all)')
    parser.add_argument('--simulation-type', choices=['pure', 'mixed'], default='mixed',
                      help='Type of quantum state to simulate (default: mixed)')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save simulation output')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("Bohmian Mixed-State Relaxation Workflow")
    print("=" * 80)
    
    # Check requirements
    print("\n[1/4] Checking requirements...")
    check_requirements()
    
    # Clone repository
    print("\n[2/4] Cloning repository...")
    repo_dir = clone_repository(args.repo_url, args.target_dir)
    
    # Run tests
    if not args.skip_tests:
        print("\n[3/4] Running tests...")
        success = run_tests(repo_dir, args.test_type)
        if not success:
            print("Tests failed. Proceeding with simulation anyway...")
    else:
        print("\n[3/4] Skipping tests...")
    
    # Run simulation
    print("\n[4/4] Running simulation...")
    run_simulation(repo_dir, args.output_dir, args.simulation_type)
    
    print("\nWorkflow completed!")
    print(f"Simulation results saved to {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()