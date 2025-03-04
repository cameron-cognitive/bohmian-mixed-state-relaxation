#!/usr/bin/env python3
"""
Module defining the 2D infinite square well physical system.

This includes the geometry, energy eigenstates, and relevant physical quantities.
"""

import numpy as np
from typing import Tuple, List, Callable


class InfiniteSquareWell2D:
    """2D infinite square well (box) quantum system.
    
    This class represents a particle in a 2D infinite square well potential,
    with methods for calculating wavefunctions, energies, and related quantities.
    
    Attributes:
        L (float): Side length of the square well
        hbar (float): Reduced Planck constant
        m (float): Particle mass
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        x_vals (np.ndarray): Grid points in x-direction
        y_vals (np.ndarray): Grid points in y-direction
        X (np.ndarray): 2D mesh grid of x-values
        Y (np.ndarray): 2D mesh grid of y-values
    """
    
    def __init__(self, L: float = np.pi, hbar: float = 1.0, m: float = 1.0, 
                 Nx: int = 100, Ny: int = 100):
        """Initialize the 2D box system.
        
        Args:
            L: Side length of the square well (default: π)
            hbar: Reduced Planck constant (default: 1.0)
            m: Particle mass (default: 1.0)
            Nx: Number of grid points in x-direction (default: 100)
            Ny: Number of grid points in y-direction (default: 100)
        """
        self.L = L
        self.hbar = hbar
        self.m = m
        self.Nx = Nx
        self.Ny = Ny
        
        # Create grid for position space
        self.x_vals = np.linspace(0, L, Nx)
        self.y_vals = np.linspace(0, L, Ny)
        self.X, self.Y = np.meshgrid(self.x_vals, self.y_vals, indexing='ij')
        
        # Grid spacing
        self.dx = L / (Nx - 1)
        self.dy = L / (Ny - 1)
    
    def eigenfunction(self, nx: int, ny: int, X: np.ndarray = None, Y: np.ndarray = None) -> np.ndarray:
        """Calculate the energy eigenfunction for quantum numbers (nx, ny).
        
        The eigenfunction is φ_{nx,ny}(x,y) = (2/L)·sin(nx·π·x/L)·sin(ny·π·y/L)
        
        Args:
            nx: Quantum number in x-direction (positive integer)
            ny: Quantum number in y-direction (positive integer)
            X: Optional custom x-coordinates (uses self.X if None)
            Y: Optional custom y-coordinates (uses self.Y if None)
            
        Returns:
            2D array containing the eigenfunction evaluated on the grid
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
            
        # Normalization factor
        norm = 2.0 / self.L
        
        # Calculate the eigenfunction
        sin_x = np.sin(nx * np.pi * X / self.L)
        sin_y = np.sin(ny * np.pi * Y / self.L)
        
        return norm * sin_x * sin_y
    
    def energy(self, nx: int, ny: int) -> float:
        """Calculate the energy eigenvalue for quantum numbers (nx, ny).
        
        The energy is E_{nx,ny} = (π²·ħ²)/(2m·L²)·(nx² + ny²)
        
        Args:
            nx: Quantum number in x-direction (positive integer)
            ny: Quantum number in y-direction (positive integer)
            
        Returns:
            Energy eigenvalue
        """
        return (np.pi**2 * self.hbar**2) / (2.0 * self.m * self.L**2) * (nx**2 + ny**2)
    
    def gradient_eigenfunction(self, nx: int, ny: int, X: np.ndarray = None, Y: np.ndarray = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the gradient of the eigenfunction for quantum numbers (nx, ny).
        
        Args:
            nx: Quantum number in x-direction (positive integer)
            ny: Quantum number in y-direction (positive integer)
            X: Optional custom x-coordinates (uses self.X if None)
            Y: Optional custom y-coordinates (uses self.Y if None)
            
        Returns:
            Tuple (∂φ/∂x, ∂φ/∂y) containing the partial derivatives
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
            
        # Normalization factor
        norm = 2.0 / self.L
        
        # Calculate the partial derivatives
        # ∂φ/∂x = (2/L)·(nx·π/L)·cos(nx·π·x/L)·sin(ny·π·y/L)
        # ∂φ/∂y = (2/L)·sin(nx·π·x/L)·(ny·π/L)·cos(ny·π·y/L)
        
        sin_x = np.sin(nx * np.pi * X / self.L)
        sin_y = np.sin(ny * np.pi * Y / self.L)
        cos_x = np.cos(nx * np.pi * X / self.L)
        cos_y = np.cos(ny * np.pi * Y / self.L)
        
        dx_factor = nx * np.pi / self.L
        dy_factor = ny * np.pi / self.L
        
        dphi_dx = norm * dx_factor * cos_x * sin_y
        dphi_dy = norm * sin_x * dy_factor * cos_y
        
        return dphi_dx, dphi_dy
    
    def laplacian_eigenfunction(self, nx: int, ny: int, X: np.ndarray = None, Y: np.ndarray = None) -> np.ndarray:
        """Calculate the Laplacian of the eigenfunction for quantum numbers (nx, ny).
        
        Args:
            nx: Quantum number in x-direction (positive integer)
            ny: Quantum number in y-direction (positive integer)
            X: Optional custom x-coordinates (uses self.X if None)
            Y: Optional custom y-coordinates (uses self.Y if None)
            
        Returns:
            2D array containing the Laplacian ∇²φ evaluated on the grid
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
            
        # For the infinite square well, we can use the fact that
        # ∇²φ = -(nx²+ny²)·(π/L)²·φ
        phi = self.eigenfunction(nx, ny, X, Y)
        factor = -(nx**2 + ny**2) * (np.pi / self.L)**2
        
        return factor * phi
    
    def velocity_field(self, wavefunction: Callable, t: float, X: np.ndarray = None, Y: np.ndarray = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the Bohmian velocity field for a given wavefunction.
        
        The velocity is v = (ħ/m)·Im[∇ψ/ψ]
        
        Args:
            wavefunction: Function that takes (X, Y, t) and returns complex wavefunction values
            t: Time at which to evaluate the velocity field
            X: Optional custom x-coordinates (uses self.X if None)
            Y: Optional custom y-coordinates (uses self.Y if None)
            
        Returns:
            Tuple (vx, vy) containing the velocity components
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
            
        # Small value to avoid division by zero
        epsilon = 1e-12
        
        # Evaluate wavefunction at the current time
        psi = wavefunction(X, Y, t)
        
        # Calculate numerical derivatives
        dx = self.dx
        dy = self.dy
        
        # Use central difference for interior points
        # and forward/backward difference at boundaries
        dpsi_dx = np.zeros_like(psi, dtype=complex)
        dpsi_dy = np.zeros_like(psi, dtype=complex)
        
        # x-derivative (central difference for interior points)
        dpsi_dx[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dx)
        # Forward difference at x=0
        dpsi_dx[0, :] = (psi[1, :] - psi[0, :]) / dx
        # Backward difference at x=L
        dpsi_dx[-1, :] = (psi[-1, :] - psi[-2, :]) / dx
        
        # y-derivative (central difference for interior points)
        dpsi_dy[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dy)
        # Forward difference at y=0
        dpsi_dy[:, 0] = (psi[:, 1] - psi[:, 0]) / dy
        # Backward difference at y=L
        dpsi_dy[:, -1] = (psi[:, -1] - psi[:, -2]) / dy
        
        # Calculate Bohmian velocities
        # v = (ħ/m)·Im[∇ψ/ψ]
        prefactor = self.hbar / self.m
        
        # Use masked arrays to handle division by zero
        mask = (np.abs(psi) < epsilon)
        psi_masked = np.ma.array(psi, mask=mask)
        
        # Calculate velocity components
        vx = prefactor * np.ma.getdata((dpsi_dx / psi_masked).imag) # Changed np.ma.imag to .imag
        vy = prefactor * np.ma.getdata((dpsi_dy / psi_masked).imag)
        
        # Fill any masked points with zeros
        vx[mask] = 0.0
        vy[mask] = 0.0
        
        return vx, vy
