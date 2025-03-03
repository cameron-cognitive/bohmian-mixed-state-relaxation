# Bohmian Mixed-State Relaxation Simulation

This repository implements a numerical simulation of quantum relaxation in Bohmian mechanics, extending Valentini's 2D infinite square well model from pure quantum states to mixed states (density matrices).

## Background

In Bohmian mechanics (also known as de Broglie-Bohm theory or pilot-wave theory), quantum systems consist of actual particles guided by a wave function. Valentini and Westman (2005) showed how an initially non-equilibrium distribution of particle positions in a 2D box can relax toward the standard Born rule distribution (|ψ|²).

This project extends their work to mixed quantum states, represented by density matrices W, and investigates whether relaxation to quantum equilibrium is complete or partial under various conditions.

## Features

- Simulation of 2D box with various eigenstate superpositions
- Mixed state (density matrix) evolution
- Non-equilibrium initial conditions via backward-time evolution
- Runge-Kutta 4 integration of Bohmian trajectories
- Computation of matrix-based relative entropy H-function:
  H(t) = Tr[ρ(t)(ln ρ(t) - ln W(t))]
- Coarse-graining analysis at multiple scales
- Visualization of relaxation dynamics
- Identification of conditions leading to partial non-convergence

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy, Matplotlib, SciPy, tqdm

### Setup

```bash
# Clone the repository
git clone https://github.com/cameron-cognitive/bohmian-mixed-state-relaxation.git
cd bohmian-mixed-state-relaxation

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using make
make setup
```

## Usage

### Command Line Interface

The main simulation can be run with various parameters:

```bash
python main.py --modes 4 --particles 1000 --tmax 2.0 --aligned-phases
```

Key parameters:
- `--modes`: Number of modes in each pure state (default: 4)
- `--particles`: Number of particles to simulate (default: 1000)
- `--tmax`: Maximum simulation time in box periods (default: 2.0)
- `--aligned-phases`: Use aligned phases for partial convergence
- `--random-phases`: Use random phases for full convergence
- `--pure-state`: Simulate pure state (not mixed)
- `--animate`: Generate animations of the simulation

### Running Examples

The repository includes several example scripts demonstrating different aspects of quantum relaxation:

```bash
# Run with make
make pure_state       # Run pure state example
make mixed_state      # Run mixed state example
make partial_convergence  # Run partial convergence example
make examples         # Run all examples

# Or run directly
python examples/pure_state.py
python examples/mixed_state.py
python examples/partial_convergence.py
```

### Jupyter Notebook

An interactive tutorial is available as a Jupyter notebook:

```bash
# Start Jupyter notebook server
jupyter notebook notebooks/bohmian_relaxation_demo.ipynb

# Or using make
make notebooks
```

## Project Structure

- `src/`: Core simulation code
  - `system.py`: Defines the 2D infinite square well system
  - `quantum_state.py`: Implements pure and mixed quantum states
  - `relaxation.py`: Handles Bohmian relaxation simulation
  - `visualization.py`: Provides plotting and animation functions
- `examples/`: Example scripts demonstrating various scenarios
- `notebooks/`: Interactive Jupyter notebooks for exploration
- `main.py`: Command-line interface to the simulation

## Key Results

1. **Full Relaxation**: With random phases and sufficient modes, both pure and mixed states fully relax to quantum equilibrium.
2. **Partial Non-Convergence**: With aligned phases and specific mode selections (especially commensurate frequencies), ~10% residual non-equilibrium can be observed, matching Valentini's findings.
3. **Mixed vs. Pure**: The mixed state formalism correctly extends the pure state case, with the H-function properly tracking the relaxation to the density matrix diagonal.

## References

- Valentini, A., & Westman, H. (2005). Dynamical origin of quantum probabilities. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 461(2053), 253-272.