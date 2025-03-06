# Bohmian Mixed-State Relaxation

This repository contains simulation code for studying quantum relaxation in Bohmian mechanics, extending the traditional approach to include mixed quantum states described by density matrices.

## Overview

Bohmian mechanics (also known as de Broglie-Bohm theory) is a deterministic interpretation of quantum mechanics where particles have definite positions guided by a quantum potential. This project explores how an initial non-equilibrium distribution of Bohmian particles approaches quantum equilibrium in a 2D infinite square well, with a special focus on mixed quantum states as described by the von Neumann guidance equation.

Key features:
- Implementation of pure and mixed quantum states
- 2D infinite square well system with configurable parameters
- Bohmian relaxation simulation with the von Neumann guidance equation
- Comprehensive visualization tools for particle distributions and velocity fields
- Coarse-grained relaxation metrics (H-function) at different scales

## Prerequisites

- Python 3.6 or higher
- NumPy (for numerical calculations)
- Matplotlib (for visualizations)
- SciPy (for integration and interpolation)
- tqdm (for progress bars)

## Installation

### Option 1: Clone and setup manually

```bash
# Clone the repository
git clone https://github.com/cameron-cognitive/bohmian-mixed-state-relaxation.git
cd bohmian-mixed-state-relaxation

# Install dependencies
pip install numpy matplotlib scipy tqdm
```

### Option 2: Use the setup script

```bash
# Clone the repository
git clone https://github.com/cameron-cognitive/bohmian-mixed-state-relaxation.git
cd bohmian-mixed-state-relaxation

# Run the setup script
python setup.py
```

## Running Tests

The repository includes comprehensive tests to ensure all components work correctly. Tests are particularly focused on verifying the velocity field calculations.

```bash
# Run all tests
python -m tests.run_tests

# Run only basic tests
python -m tests.run_tests --type basic

# Run only velocity field tests
python -m tests.run_tests --type velocity

# Run tests with minimal output
python -m tests.run_tests --quiet
```

## Running Simulations

### Option 1: Use the workflow script

The easiest way to run a simulation is to use the provided workflow script, which handles the entire process from repository cloning to visualization:

```bash
# Download the workflow script
curl -O https://raw.githubusercontent.com/cameron-cognitive/bohmian-mixed-state-relaxation/main/scripts/workflow.py

# Run a mixed state simulation
python workflow.py

# Run a pure state simulation
python workflow.py --simulation-type pure

# Specify custom output directory
python workflow.py --output-dir custom_output
```

### Option 2: Manual simulation

For more control over the simulation parameters, you can write your own script:

```python
from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState
from src.relaxation import BohmianRelaxation
from src.von_neumann_visualization import VonNeumannRelaxationVisualizer

# Create a system
L = 1.0  # Box size
system = InfiniteSquareWell2D(L)

# Create a mixed quantum state
state1 = PureState(L, [(1, 1, 1.0)])  # Ground state
state2 = PureState(L, [(2, 1, 1.0)])  # First excited state in x
mixed_state = MixedState(L, [(state1, 0.7), (state2, 0.3)])

# Create a relaxation simulation
n_particles = 1000
relaxation = BohmianRelaxation(system, mixed_state, n_particles)

# Run simulation
t_max = 2.0
dt = 0.05
results = relaxation.run_simulation(t_max, dt)

# Create visualizer and generate visualizations
visualizer = VonNeumannRelaxationVisualizer(relaxation, results)

# Create density matrix comparison
fig = visualizer.create_density_matrix_comparison(0)  # For t=0
fig.savefig('density_comparison_t0.png')

# Create velocity field visualization
fig = visualizer.create_velocity_field_visualization(0)
fig.savefig('velocity_field.png')

# Calculate and plot H-function evolution
visualizer.calculate_and_save_h_functions('output')
```

## Project Structure

- `src/`: Core simulation code
  - `quantum_state.py`: Implementation of pure and mixed quantum states
  - `system.py`: System definitions (e.g., 2D infinite square well)
  - `relaxation.py`: Bohmian relaxation simulation engine
  - `von_neumann_visualization.py`: Visualization tools for relaxation
- `tests/`: Comprehensive test suite
  - `test_suite.py`: Basic tests for all components
  - `test_velocity_field.py`: Specialized tests for velocity field calculations
  - `run_tests.py`: Test runner script
- `scripts/`: Utility scripts
  - `setup.py`: Setup script for the repository
  - `workflow.py`: Complete workflow script for simulations
- `examples/`: Example simulations and scripts
- `output/`: Default directory for simulation output

## Key Concepts

### Mixed States and von Neumann Guidance

In standard Bohmian mechanics, particle velocities are determined by the gradient of the wave function's phase. For mixed states described by density matrices, we use the von Neumann guidance equation:

```
v(x,t) = ∇S(x,t)/m
```

where S is related to the phase of the density matrix in the position representation.

### Quantum Relaxation

Quantum relaxation refers to the process by which a non-equilibrium distribution of Bohmian particles approaches the quantum equilibrium distribution (where particle density equals |ψ|²).

### H-Function

The H-function is a measure of how far the system is from quantum equilibrium, defined as:

```
H(t) = ∫ ρ(x,t) ln[ρ(x,t)/|ψ(x,t)|²] dx
```

where ρ(x,t) is the particle density and |ψ(x,t)|² is the quantum probability density.

### Coarse-Graining

Coarse-graining involves analyzing the system at different scales, which can reveal interesting scale-dependent relaxation properties.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project extends the work on quantum relaxation by Antony Valentini and others to the domain of mixed quantum states.