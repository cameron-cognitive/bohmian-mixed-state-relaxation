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
    
    def test_y_superposition_velocity(self):
        """Test velocity field for y-direction superposition."""
        t = 0
        vx, vy = self.system.velocity_field(self.y_superposition, t, self.X, self.Y)
        
        # For this superposition, vy should be non-zero but vx should be zero
        np.testing.assert_allclose(vx, np.zeros_like(vx), atol=1e-10)
        self.assertTrue(np.any(np.abs(vy) > 1e-10))
        
        # Test a specific point against analytical formula
        x_test, y_test = 0.5, 0.25
        analytical_vx, analytical_vy = self.calculate_analytical_velocity(
            self.y_superposition, t, x_test, y_test
        )
        
        # Get system's calculation for comparison
        system_vx, system_vy = self.system.velocity_field(
            self.y_superposition, t, np.array([[x_test]]), np.array([[y_test]])
        )
        
        # Compare
        self.assertAlmostEqual(system_vx[0, 0], analytical_vx, places=6)
        self.assertAlmostEqual(system_vy[0, 0], analytical_vy, places=6)
    
    def test_phase_superposition_velocity(self):
        """Test velocity field for superposition with phase difference."""
        t = 0
        vx, vy = self.system.velocity_field(self.phase_superposition, t, self.X, self.Y)
        
        # The imaginary component in the superposition should create a richer velocity field
        self.assertTrue(np.any(np.abs(vx) > 1e-10))
        
        # Test a specific point against analytical formula
        x_test, y_test = 0.25, 0.5
        analytical_vx, analytical_vy = self.calculate_analytical_velocity(
            self.phase_superposition, t, x_test, y_test
        )
        
        # Get system's calculation for comparison
        system_vx, system_vy = self.system.velocity_field(
            self.phase_superposition, t, np.array([[x_test]]), np.array([[y_test]])
        )
        
        # Compare
        self.assertAlmostEqual(system_vx[0, 0], analytical_vx, places=6)
        self.assertAlmostEqual(system_vy[0, 0], analytical_vy, places=6)
    
    def test_time_dependent_velocity(self):
        """Test time dependence of velocity field."""
        # Create a grid point
        x_test, y_test = 0.25, 0.5
        X_test = np.array([[x_test]])
        Y_test = np.array([[y_test]])
        
        # Calculate velocity at different times
        t_values = [0.0, 0.1, 0.2, 0.3]
        vx_values = []
        vy_values = []
        
        for t in t_values:
            vx, vy = self.system.velocity_field(self.phase_superposition, t, X_test, Y_test)
            vx_values.append(vx[0, 0])
            vy_values.append(vy[0, 0])
        
        # For states with energy difference, velocity should change with time
        # Check that velocities at different times are not all the same
        self.assertTrue(len(set(vx_values)) > 1)
    
    def test_mixed_state_velocity(self):
        """Test velocity field for mixed states."""
        t = 0
        
        # Calculate velocity field for mixed state
        mixed_vx, mixed_vy = self.system.velocity_field(self.mixed_state, t, self.X, self.Y)
        
        # Calculate velocity fields for component states
        ground_vx, ground_vy = self.system.velocity_field(self.ground_state, t, self.X, self.Y)
        super_vx, super_vy = self.system.velocity_field(self.x_superposition, t, self.X, self.Y)
        
        # Mixed state velocity should not simply be the average of component velocities
        # because the von Neumann guidance law is nonlinear
        avg_vx = 0.5 * ground_vx + 0.5 * super_vx
        avg_vy = 0.5 * ground_vy + 0.5 * super_vy
        
        # Check that mixed state velocity is not just the average
        self.assertFalse(np.allclose(mixed_vx, avg_vx, atol=1e-5))
        
        # But mixed state velocity should still be finite and well-defined
        self.assertTrue(np.all(np.isfinite(mixed_vx)))
        self.assertTrue(np.all(np.isfinite(mixed_vy)))
    
    def test_velocity_field_visualization(self):
        """Test visualization of velocity fields for verification."""
        t = 0
        
        # Create a figure with velocity field plots for visual inspection
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot ground state (should be zeros)
        vx, vy = self.system.velocity_field(self.ground_state, t, self.X, self.Y)
        vmag = np.sqrt(vx**2 + vy**2)
        axes[0, 0].quiver(self.X, self.Y, vx, vy, vmag, scale=20)
        axes[0, 0].set_title("Ground State Velocity Field")
        
        # Plot x-direction superposition
        vx, vy = self.system.velocity_field(self.x_superposition, t, self.X, self.Y)
        vmag = np.sqrt(vx**2 + vy**2)
        axes[0, 1].quiver(self.X, self.Y, vx, vy, vmag, scale=20)
        axes[0, 1].set_title("X-Superposition Velocity Field")
        
        # Plot y-direction superposition
        vx, vy = self.system.velocity_field(self.y_superposition, t, self.X, self.Y)
        vmag = np.sqrt(vx**2 + vy**2)
        axes[1, 0].quiver(self.X, self.Y, vx, vy, vmag, scale=20)
        axes[1, 0].set_title("Y-Superposition Velocity Field")
        
        # Plot phase superposition
        vx, vy = self.system.velocity_field(self.phase_superposition, t, self.X, self.Y)
        vmag = np.sqrt(vx**2 + vy**2)
        axes[1, 1].quiver(self.X, self.Y, vx, vy, vmag, scale=20)
        axes[1, 1].set_title("Phase Superposition Velocity Field")
        
        # Set up axes for all plots
        for ax in axes.flat:
            ax.set_xlim(0, self.L)
            ax.set_ylim(0, self.L)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        plt.tight_layout()
        
        # This is primarily for visual inspection, so just check that the plot was created
        self.assertIsInstance(fig, plt.Figure)
    
    def test_energy_conservation(self):
        """Test that velocity field conserves energy along trajectories."""
        from scipy.integrate import solve_ivp
        
        # Define ODE for particle trajectory
        def trajectory_ode(t, y):
            x, y_pos = y
            # Ensure we're inside the box
            x = max(min(x, 0.99*self.L), 0.01*self.L)
            y_pos = max(min(y_pos, 0.99*self.L), 0.01*self.L)
            
            vx, vy = self.system.velocity_field(
                self.phase_superposition, t, 
                np.array([[x]]), np.array([[y_pos]])
            )
            return [vx[0, 0], vy[0, 0]]
        
        # Initial position
        x0, y0 = 0.25, 0.5
        
        # Solve trajectory
        t_span = (0, 1.0)
        solution = solve_ivp(trajectory_ode, t_span, [x0, y0], 
                             method='RK45', rtol=1e-8, atol=1e-8)
        
        # Extract trajectory
        t_values = solution.t
        x_values = solution.y[0]
        y_values = solution.y[1]
        
        # Calculate quantum potential along trajectory
        Q_values = []
        for i in range(len(t_values)):
            t = t_values[i]
            x = x_values[i]
            y = y_values[i]
            
            # Calculate wave function
            psi = self.phase_superposition.wave_function(x, y, t)
            
            # Calculate Laplacian of psi using finite differences
            h = 1e-4
            psi_xx = (self.phase_superposition.wave_function(x+h, y, t) - 
                      2*psi + 
                      self.phase_superposition.wave_function(x-h, y, t)) / h**2
            
            psi_yy = (self.phase_superposition.wave_function(x, y+h, t) - 
                      2*psi + 
                      self.phase_superposition.wave_function(x, y-h, t)) / h**2
            
            # Quantum potential Q = -0.5 * (∇²ψ/ψ)
            Q = -0.5 * (psi_xx + psi_yy) / psi
            Q_values.append(np.real(Q))
        
        # Calculate kinetic energy along trajectory
        KE_values = []
        for i in range(len(t_values)):
            t = t_values[i]
            x = x_values[i]
            y = y_values[i]
            
            # Get velocity
            vx, vy = self.system.velocity_field(
                self.phase_superposition, t, 
                np.array([[x]]), np.array([[y]])
            )
            
            # Kinetic energy KE = 0.5 * v²
            KE = 0.5 * (vx[0, 0]**2 + vy[0, 0]**2)
            KE_values.append(KE)
        
        # Total energy = KE + Q should be conserved
        E_values = np.array(KE_values) + np.array(Q_values)
        
        # Check if energy is approximately conserved
        # We allow for some numerical error due to finite difference approximations
        E_mean = np.mean(E_values)
        E_std = np.std(E_values)
        
        # Energy fluctuations should be small compared to mean energy
        self.assertLess(E_std / abs(E_mean), 0.1)


if __name__ == '__main__':
    unittest.main()