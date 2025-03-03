#!/usr/bin/env python3
"""
Module implementing the Bohmian quantum relaxation simulation.

This includes particle trajectory integration, H-function calculation, and related utilities.
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from tqdm import tqdm

from src.system import InfiniteSquareWell2D
from src.quantum_state import PureState, MixedState


class BohmianRelaxation:
    """Simulate quantum relaxation in Bohmian mechanics.
    
    This class handles the evolution of an ensemble of Bohmian particles,
    including generation of non-equilibrium initial conditions, forward evolution,
    and calculation of the H-function.
    
    Attributes:
        system (InfiniteSquareWell2D): Reference to the physical system
        quantum_state (Union[PureState, MixedState]): Quantum state (pure or mixed)
        n_particles (int): Number of particles in the ensemble
        dt (float): Time step for numerical integration
        t_max (float): Maximum simulation time
        positions (np.ndarray): Array of particle positions (n_particles, 2, n_timesteps)
        particle_indices (np.ndarray): For mixed states, indices of which pure state guides each particle
    """
    
    def __init__(self, system: InfiniteSquareWell2D, 
                 quantum_state: Union[PureState, MixedState], 
                 n_particles: int = 1000, 
                 dt: float = 0.01, 
                 t_max: float = 2.0 * np.pi):
        """Initialize the Bohmian relaxation simulation.
        
        Args:
            system: Reference to the 2D infinite square well system
            quantum_state: Quantum state (pure or mixed)
            n_particles: Number of particles in the ensemble
            dt: Time step for numerical integration
            t_max: Maximum simulation time
        """
        self.system = system
        self.quantum_state = quantum_state
        self.n_particles = n_particles
        self.dt = dt
        self.t_max = t_max
        
        # Calculate number of time steps
        self.n_timesteps = int(t_max / dt) + 1
        
        # Initialize arrays for particle positions
        # Shape: (n_particles, 2 (x,y), n_timesteps)
        self.positions = np.zeros((n_particles, 2, self.n_timesteps))
        
        # For mixed states, assign particles to pure states
        if isinstance(quantum_state, MixedState):
            self.particle_indices = quantum_state.assign_particles_to_pure_states(n_particles)
        else:
            self.particle_indices = None
    
    def sample_from_distribution(self, distribution: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample positions from a 2D probability distribution.
        
        Args:
            distribution: 2D array of probability densities
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, 2) containing (x,y) positions
        """
        # Flatten the distribution and coordinates
        x_grid, y_grid = self.system.X, self.system.Y
        flat_dist = distribution.flatten()
        flat_x = x_grid.flatten()
        flat_y = y_grid.flatten()
        
        # Normalize the distribution
        flat_dist = flat_dist / np.sum(flat_dist)
        
        # Generate random indices according to the distribution
        indices = np.random.choice(len(flat_dist), size=n_samples, p=flat_dist)
        
        # Get the corresponding positions
        samples = np.column_stack((flat_x[indices], flat_y[indices]))
        
        return samples
    
    def rk4_step(self, x: float, y: float, t: float, dt: float, 
                 state_idx: Optional[int] = None) -> Tuple[float, float]:
        """Perform a single RK4 integration step for a particle trajectory.
        
        Args:
            x: x-coordinate of the particle
            y: y-coordinate of the particle
            t: Current time
            dt: Time step
            state_idx: For mixed states, index of the pure state guiding this particle
            
        Returns:
            Tuple (new_x, new_y) of updated particle position
        """
        # Helper function to calculate velocities at a specific position and time
        def get_velocity(x_val, y_val, t_val):
            # Use a small grid centered at the particle position
            dx = self.system.dx
            dy = self.system.dy
            X_local = np.array([[x_val - dx, x_val, x_val + dx]])
            Y_local = np.array([[y_val - dy, y_val, y_val + dy]])
            
            # Calculate velocities
            if isinstance(self.quantum_state, MixedState) and state_idx is not None:
                # Use the specific pure state for this particle
                pure_state = self.quantum_state.pure_states[state_idx]
                vx, vy = pure_state.velocity_field(X_local, Y_local, t_val)
            else:
                # Pure state or default mixed state behavior
                vx, vy = self.quantum_state.velocity_field(X_local, Y_local, t_val)
            
            # Extract the central value (corresponding to the particle position)
            return vx[0, 1], vy[0, 1]
        
        # RK4 algorithm
        k1_x, k1_y = get_velocity(x, y, t)
        
        k2_x, k2_y = get_velocity(x + 0.5 * dt * k1_x, 
                                 y + 0.5 * dt * k1_y, 
                                 t + 0.5 * dt)
        
        k3_x, k3_y = get_velocity(x + 0.5 * dt * k2_x, 
                                 y + 0.5 * dt * k2_y, 
                                 t + 0.5 * dt)
        
        k4_x, k4_y = get_velocity(x + dt * k3_x, 
                                 y + dt * k3_y, 
                                 t + dt)
        
        # Update position
        new_x = x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_y = y + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        
        # Apply boundary conditions (reflect at walls)
        L = self.system.L
        if new_x < 0:
            new_x = -new_x
        elif new_x > L:
            new_x = 2*L - new_x
            
        if new_y < 0:
            new_y = -new_y
        elif new_y > L:
            new_y = 2*L - new_y
        
        return new_x, new_y
    
    def generate_initial_conditions(self) -> None:
        """Generate non-equilibrium initial conditions via backward evolution.
        
        This implements the backward-time method described in the Valentini paper:
        1. Start with an equilibrium distribution at t = t_max
        2. Evolve backwards in time to t = 0 to get a non-equilibrium distribution
        """
        # Get the equilibrium distribution at t = t_max
        X, Y = self.system.X, self.system.Y
        if isinstance(self.quantum_state, MixedState):
            eq_distribution = self.quantum_state.density_matrix_diagonal(X, Y, self.t_max)
        else:
            eq_distribution = self.quantum_state.probability_density(X, Y, self.t_max)
        
        # Sample initial positions from the equilibrium distribution
        initial_positions = self.sample_from_distribution(eq_distribution, self.n_particles)
        
        # Store the final positions (will be t = t_max after backward evolution)
        self.positions[:, 0, -1] = initial_positions[:, 0]  # x coordinates
        self.positions[:, 1, -1] = initial_positions[:, 1]  # y coordinates
        
        # Backward evolution from t = t_max to t = 0
        # Note: We use negative time steps for backward evolution
        for i in tqdm(range(self.n_timesteps - 1, 0, -1), desc="Backward evolution"):
            t = i * self.dt
            for p in range(self.n_particles):
                x, y = self.positions[p, 0, i], self.positions[p, 1, i]
                
                # Get the pure state index for this particle (if using mixed state)
                state_idx = self.particle_indices[p] if self.particle_indices is not None else None
                
                # RK4 step with negative dt for backward evolution
                new_x, new_y = self.rk4_step(x, y, t, -self.dt, state_idx)
                
                # Store the new position
                self.positions[p, 0, i-1] = new_x
                self.positions[p, 1, i-1] = new_y
    
    def run_simulation(self) -> Dict:
        """Run the forward Bohmian simulation.
        
        This evolves the non-equilibrium initial distribution forward in time
        according to the Bohmian guidance equation.
        
        Returns:
            Dictionary containing simulation results and metadata
        """
        # Initial positions are already stored in self.positions[:, :, 0]
        
        # Forward evolution from t = 0 to t = t_max
        for i in tqdm(range(1, self.n_timesteps), desc="Forward evolution"):
            t = i * self.dt
            for p in range(self.n_particles):
                x, y = self.positions[p, 0, i-1], self.positions[p, 1, i-1]
                
                # Get the pure state index for this particle (if using mixed state)
                state_idx = self.particle_indices[p] if self.particle_indices is not None else None
                
                # RK4 step forward
                new_x, new_y = self.rk4_step(x, y, t - self.dt, self.dt, state_idx)
                
                # Store the new position
                self.positions[p, 0, i] = new_x
                self.positions[p, 1, i] = new_y
        
        # Return results
        results = {
            "positions": self.positions,
            "particle_indices": self.particle_indices,
            "dt": self.dt,
            "t_max": self.t_max,
            "n_timesteps": self.n_timesteps
        }
        
        return results
    
    def calculate_density_histogram(self, timestep: int, bins: int = 50) -> np.ndarray:
        """Calculate a histogram of particle positions at a given timestep.
        
        Args:
            timestep: Index of the timestep to analyze
            bins: Number of bins in each dimension for the histogram
            
        Returns:
            2D array representing the particle density histogram
        """
        # Extract particle positions at the given timestep
        x_positions = self.positions[:, 0, timestep]
        y_positions = self.positions[:, 1, timestep]
        
        # Calculate histogram
        H, x_edges, y_edges = np.histogram2d(
            x_positions, y_positions, 
            bins=bins, 
            range=[[0, self.system.L], [0, self.system.L]]
        )
        
        # Normalize the histogram
        bin_area = (self.system.L / bins) ** 2
        H = H / (self.n_particles * bin_area)
        
        return H
    
    def calculate_h_function(self, timestep: Optional[int] = None, coarse_grain: Optional[int] = None) -> Union[float, np.ndarray]:
        """Calculate the H-function (relative entropy) at the specified timestep(s).
        
        For a mixed state, this calculates Tr[ρ(ln ρ - ln W)], where ρ is the actual
        particle distribution and W is the quantum equilibrium distribution (diagonal
        of the density matrix).
        
        Args:
            timestep: Index of the timestep to analyze, or None for all timesteps
            coarse_grain: Number of bins in each dimension for coarse-graining, or None for default
            
        Returns:
            H-function value at the specified timestep, or array of values for all timesteps
        """
        # Set default coarse-graining level if not specified
        if coarse_grain is None:
            coarse_grain = min(50, self.system.Nx // 2)  # Default to 50 or half the grid size
        
        # Initialize array for H-function values
        if timestep is None:
            h_values = np.zeros(self.n_timesteps)
            timesteps = range(self.n_timesteps)
        else:
            h_values = np.zeros(1)
            timesteps = [timestep]
        
        # Small value to avoid log(0)
        epsilon = 1e-12
        
        # Calculate H-function for each timestep
        for i, ts in enumerate(timesteps):
            t = ts * self.dt
            
            # Calculate actual particle distribution
            rho = self.calculate_density_histogram(ts, bins=coarse_grain)
            
            # Calculate quantum equilibrium distribution
            X, Y = np.meshgrid(
                np.linspace(0, self.system.L, coarse_grain),
                np.linspace(0, self.system.L, coarse_grain),
                indexing='ij'
            )
            
            if isinstance(self.quantum_state, MixedState):
                w = self.quantum_state.density_matrix_diagonal(X, Y, t)
            else:
                w = self.quantum_state.probability_density(X, Y, t)
            
            # Ensure proper normalization of w
            w = w / np.sum(w) * np.sum(rho)
            
            # Calculate the integrand: ρ(ln ρ - ln W)
            integrand = rho * (np.log(rho + epsilon) - np.log(w + epsilon))
            
            # Sum to get the H-function
            h_value = np.sum(integrand) * (self.system.L / coarse_grain) ** 2
            
            # Store the result
            if timestep is None:
                h_values[ts] = h_value
            else:
                h_values[0] = h_value
        
        return h_values
    
    def calculate_h_matrix(self, timestep: int, coarse_grain: Optional[int] = None) -> np.ndarray:
        """Calculate the H-function matrix (relative entropy density) at the specified timestep.
        
        This returns the matrix ρ(ln ρ - ln W) at each point, which when summed gives the H-function.
        
        Args:
            timestep: Index of the timestep to analyze
            coarse_grain: Number of bins in each dimension for coarse-graining, or None for default
            
        Returns:
            2D array representing the H-function matrix
        """
        # Set default coarse-graining level if not specified
        if coarse_grain is None:
            coarse_grain = min(50, self.system.Nx // 2)  # Default to 50 or half the grid size
        
        t = timestep * self.dt
        
        # Calculate actual particle distribution
        rho = self.calculate_density_histogram(timestep, bins=coarse_grain)
        
        # Calculate quantum equilibrium distribution
        X, Y = np.meshgrid(
            np.linspace(0, self.system.L, coarse_grain),
            np.linspace(0, self.system.L, coarse_grain),
            indexing='ij'
        )
        
        if isinstance(self.quantum_state, MixedState):
            w = self.quantum_state.density_matrix_diagonal(X, Y, t)
        else:
            w = self.quantum_state.probability_density(X, Y, t)
        
        # Ensure proper normalization of w
        w = w / np.sum(w) * np.sum(rho)
        
        # Small value to avoid log(0)
        epsilon = 1e-12
        
        # Calculate the H-function matrix: ρ(ln ρ - ln W)
        h_matrix = rho * (np.log(rho + epsilon) - np.log(w + epsilon))
        
        return h_matrix
