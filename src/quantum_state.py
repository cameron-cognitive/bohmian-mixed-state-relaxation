#!/usr/bin/env python3
"""
Module defining the quantum state classes for the Bohmian simulation.

This includes implementations for pure states and mixed states.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional

from src.system import InfiniteSquareWell2D


class PureState:
    """Pure quantum state represented as a superposition of energy eigenstates.
    
    This class represents a pure quantum state |Ψ⟩ as a superposition of energy
    eigenstates with specified amplitudes and phases.
    
    Attributes:
        system (InfiniteSquareWell2D): Reference to the physical system
        mode_indices (List[Tuple[int, int]]): List of (nx, ny) mode indices
        amplitudes (np.ndarray): Array of mode amplitudes
        phases (np.ndarray): Array of mode phases
        energies (np.ndarray): Array of mode energies
    """
    
    def __init__(self, system: InfiniteSquareWell2D, 
                 mode_indices: List[Tuple[int, int]], 
                 amplitudes: np.ndarray, 
                 phases: np.ndarray):
        """Initialize a pure state in the 2D box.
        
        Args:
            system: Reference to the 2D infinite square well system
            mode_indices: List of (nx, ny) mode indices
            amplitudes: Array of mode amplitudes (should be normalized)
            phases: Array of mode phases in radians
        """
        self.system = system
        self.mode_indices = mode_indices
        self.amplitudes = amplitudes
        self.phases = phases
        
        # Calculate energies for each mode
        self.energies = np.array([system.energy(nx, ny) for nx, ny in mode_indices])
        
        # Check normalization
        norm = np.sum(np.abs(amplitudes)**2)
        if not np.isclose(norm, 1.0, rtol=1e-10):
            print(f"Warning: Input amplitudes not normalized (sum of |a|^2 = {norm}). Normalizing...")
            self.amplitudes = amplitudes / np.sqrt(norm)
    
    def evaluate(self, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        """Evaluate the wavefunction at a given time.
        
        The time-dependent wavefunction is:
        Ψ(x,y,t) = Σ_k a_k exp(i θ_k) exp(-i E_k t/ħ) φ_k(x,y)
        
        Args:
            X: Grid of x coordinates
            Y: Grid of y coordinates
            t: Time at which to evaluate the wavefunction
            
        Returns:
            Complex array representing the wavefunction values
        """
        # Initialize wavefunction with zeros
        psi = np.zeros_like(X, dtype=complex)
        
        # Sum over all modes
        for idx, (nx, ny) in enumerate(self.mode_indices):
            # Calculate the spatial part (eigenfunction)
            phi = self.system.eigenfunction(nx, ny, X, Y)
            
            # Calculate the complex amplitude with time dependence
            amplitude = self.amplitudes[idx]
            phase = self.phases[idx]
            energy = self.energies[idx]
            
            # Time evolution factor
            time_factor = np.exp(-1j * energy * t / self.system.hbar)
            
            # Add this mode's contribution
            psi += amplitude * np.exp(1j * phase) * time_factor * phi
        
        return psi
    
    def probability_density(self, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        """Calculate the probability density |Ψ|² at a given time.
        
        Args:
            X: Grid of x coordinates
            Y: Grid of y coordinates
            t: Time at which to evaluate the probability density
            
        Returns:
            Real array representing |Ψ|² values
        """
        psi = self.evaluate(X, Y, t)
        return np.abs(psi)**2
    
    def velocity_field(self, X: np.ndarray, Y: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the Bohmian velocity field at a given time.
        
        Args:
            X: Grid of x coordinates
            Y: Grid of y coordinates
            t: Time at which to evaluate the velocity field
            
        Returns:
            Tuple (vx, vy) of velocity components
        """
        # Use the system's velocity_field method with our wavefunction
        return self.system.velocity_field(self.evaluate, t, X, Y)


class MixedState:
    """Mixed quantum state represented as a statistical mixture of pure states.
    
    This class represents a mixed quantum state ρ = Σ_k w_k |Ψ_k⟩⟨Ψ_k| as a
    weighted sum of pure state density matrices.
    
    Attributes:
        system (InfiniteSquareWell2D): Reference to the physical system
        pure_states (List[PureState]): List of pure states in the mixture
        weights (np.ndarray): Array of statistical weights (probabilities)
    """
    
    def __init__(self, system: InfiniteSquareWell2D, 
                 pure_states: List[PureState], 
                 weights: np.ndarray):
        """Initialize a mixed state in the 2D box.
        
        Args:
            system: Reference to the 2D infinite square well system
            pure_states: List of pure states in the mixture
            weights: Array of statistical weights (should sum to 1)
        """
        self.system = system
        self.pure_states = pure_states
        self.weights = np.array(weights)
        
        # Check normalization of weights
        weight_sum = np.sum(weights)
        if not np.isclose(weight_sum, 1.0, rtol=1e-10):
            print(f"Warning: Weights do not sum to 1 (sum = {weight_sum}). Normalizing...")
            self.weights = weights / weight_sum
    
    def density_matrix_diagonal(self, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        """Calculate the diagonal of the density matrix in position representation.
        
        This gives the position probability density ρ(x,y,t) = ⟨x,y|ρ(t)|x,y⟩
        For a mixed state, this is the weighted sum of the pure state probabilities.
        
        Args:
            X: Grid of x coordinates
            Y: Grid of y coordinates
            t: Time at which to evaluate the density matrix
            
        Returns:
            Real array representing the diagonal density matrix elements
        """
        # Initialize density matrix diagonal with zeros
        rho_diag = np.zeros_like(X, dtype=float)
        
        # Sum over all pure states with appropriate weights
        for idx, pure_state in enumerate(self.pure_states):
            weight = self.weights[idx]
            prob_density = pure_state.probability_density(X, Y, t)
            rho_diag += weight * prob_density
        
        return rho_diag
    
    def velocity_field(self, X: np.ndarray, Y: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the Bohmian velocity field for the mixed state.
        
        For mixed states in Bohmian mechanics, there's not a single well-defined
        velocity field like for pure states. Instead, we need to consider an ensemble
        of particles guided by different pure states according to their weights.
        
        Here we use a pragmatic approach by returning a velocity field based on
        some chosen rule. Two common approaches are:
        1. Return the velocity field of the most heavily weighted pure state
        2. Return a randomly selected pure state's velocity field with probability
           given by the weights
        
        In this implementation, we use option 1 for deterministic behavior.
        
        Args:
            X: Grid of x coordinates
            Y: Grid of y coordinates
            t: Time at which to evaluate the velocity field
            
        Returns:
            Tuple (vx, vy) of velocity components
        """
        # Find the pure state with the highest weight
        max_weight_idx = np.argmax(self.weights)
        max_weight_state = self.pure_states[max_weight_idx]
        
        # Return its velocity field
        return max_weight_state.velocity_field(X, Y, t)
    
    def velocity_field_probabilistic(self, X: np.ndarray, Y: np.ndarray, t: float, 
                                   particle_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate a probabilistic velocity field for the mixed state.
        
        This method assigns each particle to a pure state according to the
        weights of the mixed state, then returns the appropriate velocity
        for each particle based on its assigned pure state.
        
        Args:
            X: Array of particle x-coordinates
            Y: Array of particle y-coordinates
            t: Time at which to evaluate the velocity field
            particle_indices: Array of indices indicating which pure state
                             each particle follows (should be pre-assigned)
            
        Returns:
            Tuple (vx, vy) of velocity components for each particle
        """
        if X.shape != particle_indices.shape:
            raise ValueError("Particle positions and indices arrays must have the same shape")
        
        # Initialize velocity arrays
        vx = np.zeros_like(X)
        vy = np.zeros_like(Y)
        
        # Calculate velocities for each particle based on its assigned pure state
        for idx, pure_state in enumerate(self.pure_states):
            # Find particles assigned to this pure state
            mask = (particle_indices == idx)
            
            if np.any(mask):
                # Extract positions of these particles
                X_subset = X[mask]
                Y_subset = Y[mask]
                
                # Calculate velocities for this subset
                vx_subset, vy_subset = pure_state.velocity_field(X_subset, Y_subset, t)
                
                # Assign velocities to the appropriate particles
                vx[mask] = vx_subset
                vy[mask] = vy_subset
        
        return vx, vy
    
    def assign_particles_to_pure_states(self, n_particles: int) -> np.ndarray:
        """Assign particles to pure states according to the mixed state weights.
        
        Args:
            n_particles: Total number of particles to assign
            
        Returns:
            Array of indices indicating which pure state each particle follows
        """
        # Calculate the number of particles for each pure state
        particles_per_state = np.floor(n_particles * self.weights).astype(int)
        
        # Handle any remaining particles due to rounding
        remaining = n_particles - np.sum(particles_per_state)
        if remaining > 0:
            # Assign remaining particles to states with highest fractional parts
            fractions = n_particles * self.weights - particles_per_state
            indices = np.argsort(fractions)[-int(remaining):]  # Get indices of highest fractions
            for idx in indices:
                particles_per_state[idx] += 1
        
        # Create the assignment array
        assignments = np.zeros(n_particles, dtype=int)
        current_idx = 0
        
        for state_idx, n_state_particles in enumerate(particles_per_state):
            end_idx = current_idx + n_state_particles
            assignments[current_idx:end_idx] = state_idx
            current_idx = end_idx
        
        # Shuffle the assignments to avoid any bias
        np.random.shuffle(assignments)
        
        return assignments
