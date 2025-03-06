#!/usr/bin/env python3
"""
Setup script for Bohmian mixed-state relaxation simulation.

This script:
1. Checks for dependencies
2. Prepares the environment
3. Creates a basic configuration
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json

DEPENDENCIES = ['numpy', 'matplotlib', 'scipy', 'tqdm']

def check_python_version():
    """Check if the Python version is adequate."""
    required_version = (3, 6)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current Python version: {current_version[0]}.{current_version[1]}")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + DEPENDENCIES)
        print("✅ All dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies.")
        return False

def create_config(config_dir=None):
    """Create a configuration file with default settings."""
    if config_dir is None:
        config_dir = os.path.join(os.path.expanduser("~"), ".bohmian_sim")
    
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.json")
    
    default_config = {
        "simulation": {
            "box_size": 1.0,
            "n_particles": 1000,
            "t_max": 2.0,
            "dt": 0.05,
            "random_seed": 42
        },
        "visualization": {
            "dpi": 200,
            "color_map": "viridis",
            "coarse_graining_levels": [4, 8, 16, 32]
        },
        "paths": {
            "output_dir": "output",
            "data_dir": "data"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    
    print(f"✅ Configuration file created at {config_path}")
    return config_path

def create_directories():
    """Create necessary directories for the project."""
    dirs = ['output', 'data']
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Created directory: {d}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup for Bohmian mixed-state relaxation simulation')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--config-dir', type=str, default=None, help='Custom config directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Setting up Bohmian Mixed-State Relaxation Simulation")
    print("=" * 60)
    
    # Check Python version
    print("\n[1/4] Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    print("\n[2/4] Installing dependencies...")
    if args.skip_deps:
        print("Skipping dependency installation.")
    else:
        if not install_dependencies():
            print("Warning: Some dependencies could not be installed.")
            print("You may need to install them manually.")
    
    # Create configuration
    print("\n[3/4] Creating configuration...")
    config_path = create_config(args.config_dir)
    
    # Create directories
    print("\n[4/4] Creating directories...")
    create_directories()
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print(f"Configuration file: {config_path}")
    print("You can now run simulations using the workflow script.")
    print("=" * 60)

if __name__ == "__main__":
    main()