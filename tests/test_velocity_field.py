#!/usr/bin/env python3
"""
Specialized tests for the velocity field calculations in Bohmian mechanics.

This module provides in-depth tests specifically for velocity field calculations
to ensure the correctness of the quantum guiding equations for both pure and mixed states.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from src.quantum_state import PureState, MixedState
from src.system import InfiniteSquareWell2D


class VelocityFieldTests(unittest.TestCase):
    """Comprehensive tests for velocity field calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.L = 1.0
        self.system = InfiniteSquareWell2D(self.L)
        
        # Create various quantum states for testing
        # Ground state (should have zero velocity)
        self.ground_state = PureState(self.L, [(1, 1, 1.0)])
        
        # Superposition of ground and first excited state in x direction
        self.x_superposition = PureState(self.L, [
            (1, 1, 1/np.sqrt(2)),
            (2, 1, 1/np.sqrt(2))
        ])
        
        # Superposition of ground and first excited state in y direction
        self.y_superposition = PureState(self.L, [
            (1, 1, 1/np.sqrt(2)),
            (1, 2, 1/np.sqrt(2))
        ])
        
        # Superposition with phase difference
        self.phase_superposition = PureState(self.L, [
            (1, 1, 1/np.sqrt(2)),
            (2, 1, 1j/np.sqrt(2))  # 90° phase difference
        ])
        
        # Mixed state of two pure states
        self.mixed_state = MixedState(self.L, [
            (self.ground_state, 0.5),
            (self.x_superposition, 0.5)
        ])
        
        # Create a grid of points for analysis
        self.grid_size = 20
        x = np.linspace(0.05, 0.95, self.grid_size)  # Avoid boundaries
        y = np.linspace(0.05, 0.95, self.grid_size)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
    
    def calculate_analytical_velocity(self, state: PureState, t: float, 
                                      x: float, y: float) -> Tuple[float, float]:
        """Calculate velocity directly from the Bohmian guidance equation."""
        # Get wave function and its spatial derivatives
        psi = state.wave_function(x, y, t)
        
        # Calculate derivatives using finite differences
        h = 1e-6  # Step size for numerical differentiation
        psi_x_plus = state.wave_function(x+h, y, t)
        psi_x_minus = state.wave_function(x-h, y, t)
        psi_y_plus = state.wave_function(x, y+h, t)
        psi_y_minus = state.wave_function(x, y-h, t)
        
        # Central difference approximation
        d_psi_dx = (psi_x_plus - psi_x_minus) / (2*h)
        d_psi_dy = (psi_y_plus - psi_y_minus) / (2*h)
        
        # Apply Bohmian velocity equation: v = Im(∇ψ/ψ)
        vx = np.imag(d_psi_dx / psi)
        vy = np.imag(d_psi_dy / psi)
        
        return vx, vy
    
    def test_ground_state_zero_velocity(self):
        """Test that ground state has zero velocity field."""
        t = 0
        vx, vy = self.system.velocity_field(self.ground_state, t, self.X, self.Y)
        
        # For ground state, velocity should be zero everywhere
        np.testing.assert_allclose(vx, np.zeros_like(vx), atol=1e-10)
        np.testing.assert_allclose(vy, np.zeros_like(vy), atol=1e-10)
    
    def test_x_superposition_velocity(self):
        """Test velocity field for x-direction superposition."""
        t = 0
        vx, vy = self.system.velocity_field(self.x_superposition, t, self.X, self.Y)
        
        # For this superposition, vx should be non-zero but vy should be zero
        self.assertTrue(np.any(np.abs(vx) > 1e-10))
        np.testing.assert_allclose(vy, np.zeros_like(vy), atol=1e-10)
        
        # Test a specific point against analytical formula
        x_test, y_test = 0.25, 0.5
        analytical_vx, analytical_vy = self.calculate_analytical_velocity(
            self.x_superposition, t, x_test, y_test
        )
        
        # Get system's calculation for comparison
        system_vx, system_vy = self.system.velocity_field(
            self.x_superposition, t, np.array([[x_test]]), np.array([[y_test]])
        )
        
        # Compare
        self.assertAlmostEqual(system_vx[0, 0], analytical_vx, places=6)
        self.assertAlmostEqual(system_vy[0, 0], analytical_vy, places=6)
