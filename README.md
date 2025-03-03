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

```bash
git clone https://github.com/cameron-cognitive/bohmian-mixed-state-relaxation.git
cd bohmian-mixed-state-relaxation
# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --modes 4 --particles 1000 --tmax 2.0 --aligned-phases
```

See documentation for additional arguments and examples.

## Examples

The `examples` directory contains several demonstration scenarios:

- `pure_state.py`: Reproduces Valentini's original pure-state results
- `mixed_state.py`: Extends to a 50/50 mixture of two pure states
- `partial_convergence.py`: Demonstrates conditions with ~10% residual non-equilibrium

## References

- Valentini, A., & Westman, H. (2005). Dynamical origin of quantum probabilities. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 461(2053), 253-272.
