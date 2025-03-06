#!/usr/bin/env python3
"""
Test suite for the Bohmian mixed-state relaxation simulation.

This module provides comprehensive tests for all components of the simulation:
- Quantum states (pure and mixed)
- System definitions and dynamics
- Relaxation calculations
- Visualization functionality
"""

import unittest
import numpy as np
import os
import tempfile
from matplotlib.figure import Figure

# Import modules to be tested
from src.quantum_state import PureState, MixedState
from src.system import InfiniteSquareWell2D
from src.relaxation import BohmianRelaxation
from src.von_neumann_visualization import VonNeumannRelaxationVisualizer


class TestPureState(unittest.TestCase):
    """Tests for the PureState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple pure state (ground state of infinite square well)
        self.L = 1.0  # Box size
        self.n_x = 1  # Quantum number in x direction
        self.n_y = 1  # Quantum number in y direction
        self.pure_state = PureState(self.L, [(self.n_x, self.n_y, 1.0)])
    
    def test_wave_function(self):
        """Test wave function calculation."""
        # Test at a few points
        x, y = 0.25, 0.25
        t = 0
        
        # Calculate expected value for ground state
        expected = np.sin(self.n_x * np.pi * x / self.L) * np.sin(self.n_y * np.pi * y / self.L)
        expected *= np.sqrt(4 / (self.L ** 2))  # Normalization for 2D
        
        # Get actual value
        actual = self.pure_state.wave_function(x, y, t)
        
        # Compare with tolerance
        self.assertAlmostEqual(actual, expected, places=10)
    
    def test_probability_density(self):
        """Test probability density calculation."""
        # Create grid of points
        x = np.linspace(0, self.L, 10)
        y = np.linspace(0, self.L, 10)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t = 0
        
        # Calculate probability density
        density = self.pure_state.probability_density(X, Y, t)
        
        # Check it's normalized (integral should be approximately 1)
        integral = np.sum(density) * (self.L / 10) ** 2
        self.assertAlmostEqual(integral, 1.0, places=3)
        
        # Check maximum is at expected location (center for ground state)
        max_idx = np.unravel_index(np.argmax(density), density.shape)
        self.assertEqual(max_idx, (5, 5))  # Middle of 10x10 grid
    
    def test_phase(self):
        """Test phase calculation."""
        x, y = 0.25, 0.25
        t = 0
        
        # For a real-valued ground state, phase should be 0
        phase = self.pure_state.phase(x, y, t)
        self.assertAlmostEqual(phase, 0.0, places=10)
        
        # Create a state with time-dependent phase
        energy = (self.n_x**2 + self.n_y**2) * (np.pi**2 / (2 * self.L**2))
        t = 1.0
        expected_phase = -energy * t
        phase = self.pure_state.phase(x, y, t)
        self.assertAlmostEqual(phase, expected_phase, places=10)
    
    def test_superposition(self):
        """Test superposition of states."""
        # Create a superposition of ground and first excited state
        superposition = PureState(self.L, [(1, 1, 1/np.sqrt(2)), (2, 1, 1/np.sqrt(2))])
        
        # Check normalization
        x = np.linspace(0, self.L, 20)
        y = np.linspace(0, self.L, 20)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t = 0
        
        density = superposition.probability_density(X, Y, t)
        integral = np.sum(density) * (self.L / 20) ** 2
        self.assertAlmostEqual(integral, 1.0, places=3)


class TestMixedState(unittest.TestCase):
    """Tests for the MixedState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.L = 1.0
        
        # Create a pure state first for comparison
        self.pure_state = PureState(self.L, [(1, 1, 1.0)])
        
        # Create a mixed state with just one pure state (should behave like pure state)
        self.mixed_state_single = MixedState(self.L, [
            (self.pure_state, 1.0)
        ])
        
        # Create a more complex mixed state
        pure_state2 = PureState(self.L, [(2, 1, 1.0)])
        self.mixed_state = MixedState(self.L, [
            (self.pure_state, 0.7),
            (pure_state2, 0.3)
        ])
    
    def test_single_state_equivalence(self):
        """Test that a mixed state with one pure state behaves like that pure state."""
        x = np.linspace(0, self.L, 10)
        y = np.linspace(0, self.L, 10)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t = 0
        
        pure_density = self.pure_state.probability_density(X, Y, t)
        mixed_density = self.mixed_state_single.density_matrix_diagonal(X, Y, t)
        
        # Compare densities
        np.testing.assert_allclose(pure_density, mixed_density, rtol=1e-10)
    
    def test_density_matrix_normalization(self):
        """Test that the density matrix is properly normalized."""
        x = np.linspace(0, self.L, 20)
        y = np.linspace(0, self.L, 20)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t = 0
        
        density = self.mixed_state.density_matrix_diagonal(X, Y, t)
        integral = np.sum(density) * (self.L / 20) ** 2
        self.assertAlmostEqual(integral, 1.0, places=3)
    
    def test_mixed_state_properties(self):
        """Test properties of a mixed state."""
        x = np.linspace(0, self.L, 10)
        y = np.linspace(0, self.L, 10)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t = 0
        
        # Calculate density from each component state
        pure1_density = self.pure_state.probability_density(X, Y, t)
        pure2_density = PureState(self.L, [(2, 1, 1.0)]).probability_density(X, Y, t)
        
        # Mixed state should be weighted sum
        expected_mixed = 0.7 * pure1_density + 0.3 * pure2_density
        actual_mixed = self.mixed_state.density_matrix_diagonal(X, Y, t)
        
        np.testing.assert_allclose(actual_mixed, expected_mixed, rtol=1e-10)


class TestInfiniteSquareWell2D(unittest.TestCase):
    """Tests for the InfiniteSquareWell2D class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.L = 1.0
        self.system = InfiniteSquareWell2D(self.L)
        
        # Create a pure state for testing
        self.pure_state = PureState(self.L, [(1, 1, 1.0)])
        
        # Create a more complex superposition state for testing velocity
        self.superposition = PureState(self.L, [
            (1, 1, 1/np.sqrt(2)),
            (2, 1, 1/np.sqrt(2))
        ])
    
    def test_energy_levels(self):
        """Test energy level calculation."""
        # Calculate energy for a few states
        E_11 = self.system.energy(1, 1)
        E_21 = self.system.energy(2, 1)
        E_22 = self.system.energy(2, 2)
        
        # Expected energies (in units where ℏ=1 and m=1)
        expected_E_11 = (np.pi**2 / 2) * (1**2 + 1**2) / (self.L**2)
        expected_E_21 = (np.pi**2 / 2) * (2**2 + 1**2) / (self.L**2)
        expected_E_22 = (np.pi**2 / 2) * (2**2 + 2**2) / (self.L**2)
        
        self.assertAlmostEqual(E_11, expected_E_11, places=10)
        self.assertAlmostEqual(E_21, expected_E_21, places=10)
        self.assertAlmostEqual(E_22, expected_E_22, places=10)
    
    def test_ground_state_velocity_field(self):
        """Test velocity field for ground state (should be zero)."""
        # Create a grid
        x = np.linspace(0.1, 0.9, 5)  # Avoid boundaries
        y = np.linspace(0.1, 0.9, 5)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t = 0
        
        # Calculate velocity field
        vx, vy = self.system.velocity_field(self.pure_state, t, X, Y)
        
        # For ground state, velocity should be zero everywhere
        np.testing.assert_allclose(vx, np.zeros_like(vx), atol=1e-10)
        np.testing.assert_allclose(vy, np.zeros_like(vy), atol=1e-10)
    
    def test_superposition_velocity_field(self):
        """Test velocity field for a superposition state."""
        # Create a grid
        x = np.linspace(0.1, 0.9, 5)  # Avoid boundaries
        y = np.linspace(0.1, 0.9, 5)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t = 0
        
        # Calculate velocity field
        vx, vy = self.system.velocity_field(self.superposition, t, X, Y)
        
        # For this superposition, vx should be non-zero but vy should be zero
        self.assertTrue(np.any(np.abs(vx) > 1e-10))
        np.testing.assert_allclose(vy, np.zeros_like(vy), atol=1e-10)
        
        # Test a specific point - analytical check
        # For a superposition of (1,1) and (2,1), calculate expected velocity
        x_test, y_test = 0.25, 0.5
        
        # Wave function components
        psi_11 = np.sin(np.pi * x_test / self.L) * np.sin(np.pi * y_test / self.L)
        psi_21 = np.sin(2 * np.pi * x_test / self.L) * np.sin(np.pi * y_test / self.L)
        
        # Derivatives
        d_psi_11_dx = (np.pi / self.L) * np.cos(np.pi * x_test / self.L) * np.sin(np.pi * y_test / self.L)
        d_psi_21_dx = (2 * np.pi / self.L) * np.cos(2 * np.pi * x_test / self.L) * np.sin(np.pi * y_test / self.L)
        
        # Full wave function and its derivative
        psi = (psi_11 + psi_21) / np.sqrt(2)
        d_psi_dx = (d_psi_11_dx + d_psi_21_dx) / np.sqrt(2)
        
        # Expected velocity at this point
        expected_vx = np.imag(d_psi_dx / psi)
        
        # Get actual velocity at this point
        actual_vx, _ = self.system.velocity_field(self.superposition, t, np.array([[x_test]]), np.array([[y_test]]))
        
        # Compare
        self.assertAlmostEqual(actual_vx[0, 0], expected_vx, places=5)
    
    def test_mixed_state_velocity_field(self):
        """Test velocity field for a mixed state."""
        # Create a mixed state
        pure_state1 = PureState(self.L, [(1, 1, 1.0)])
        pure_state2 = PureState(self.L, [(2, 1, 1.0)])
        mixed_state = MixedState(self.L, [
            (pure_state1, 0.5),
            (pure_state2, 0.5)
        ])
        
        # Create a grid
        x = np.linspace(0.1, 0.9, 5)
        y = np.linspace(0.1, 0.9, 5)
        X, Y = np.meshgrid(x, y, indexing='ij')
        t = 0
        
        # Calculate mixed state velocity field
        vx, vy = self.system.velocity_field(mixed_state, t, X, Y)
        
        # The mixed state velocity should not be the same as either pure state
        vx1, vy1 = self.system.velocity_field(pure_state1, t, X, Y)
        vx2, vy2 = self.system.velocity_field(pure_state2, t, X, Y)
        
        # Make sure it's not identical to either pure state 
        self.assertFalse(np.allclose(vx, vx1, atol=1e-5))
        self.assertFalse(np.allclose(vx, vx2, atol=1e-5))
        
        # For von Neumann guidance, velocity should be related to gradient of rho / rho
        # This is harder to test directly, but we can verify it's finite and well-behaved
        self.assertTrue(np.all(np.isfinite(vx)))
        self.assertTrue(np.all(np.isfinite(vy)))


class TestBohmianRelaxation(unittest.TestCase):
    """Tests for the BohmianRelaxation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.L = 1.0
        self.system = InfiniteSquareWell2D(self.L)
        
        # Create a pure state
        self.pure_state = PureState(self.L, [(1, 1, 1.0)])
        
        # Create a mixed state
        pure_state2 = PureState(self.L, [(2, 1, 1.0)])
        self.mixed_state = MixedState(self.L, [
            (self.pure_state, 0.7),
            (pure_state2, 0.3)
        ])
        
        # Create a small relaxation simulation
        self.n_particles = 100
        self.t_max = 0.1
        self.dt = 0.01
        self.relaxation = BohmianRelaxation(
            self.system, self.mixed_state, self.n_particles
        )
    
    def test_initialize_particles(self):
        """Test particle initialization."""
        # Initialize particles with a specified distribution
        particles = self.relaxation.initialize_particles(distribution='uniform')
        
        # Check shape
        self.assertEqual(particles.shape, (self.n_particles, 2))
        
        # Check all particles are within the box
        self.assertTrue(np.all(particles >= 0))
        self.assertTrue(np.all(particles <= self.L))
    
    def test_run_simulation(self):
        """Test running a short simulation."""
        # Run a short simulation
        results = self.relaxation.run_simulation(self.t_max, self.dt)
        
        # Check results contain expected keys
        self.assertIn('positions', results)
        self.assertIn('dt', results)
        self.assertIn('t_max', results)
        self.assertIn('n_timesteps', results)
        
        # Check shape of positions array
        positions = results['positions']
        n_timesteps = int(self.t_max / self.dt) + 1
        self.assertEqual(positions.shape, (self.n_particles, 2, n_timesteps))
        
        # Ensure particles remain in the box
        self.assertTrue(np.all(positions >= 0))
        self.assertTrue(np.all(positions <= self.L))
    
    def test_calculate_h_function(self):
        """Test H-function calculation."""
        # First run a simulation
        results = self.relaxation.run_simulation(self.t_max, self.dt)
        self.relaxation.results = results
        
        # Calculate H-function
        h_values = self.relaxation.calculate_h_function()
        
        # Check shape
        n_timesteps = int(self.t_max / self.dt) + 1
        self.assertEqual(len(h_values), n_timesteps)
        
        # Check values are finite and non-negative
        self.assertTrue(np.all(np.isfinite(h_values)))
        self.assertTrue(np.all(h_values >= 0))
    
    def test_calculate_h_matrix(self):
        """Test H-function matrix calculation."""
        # First run a simulation
        results = self.relaxation.run_simulation(self.t_max, self.dt)
        self.relaxation.results = results
        
        # Calculate H-matrix for the first timestep
        h_matrix = self.relaxation.calculate_h_matrix(0, coarse_grain=10)
        
        # Check shape - should be a 2D grid
        self.assertEqual(len(h_matrix.shape), 2)
        self.assertEqual(h_matrix.shape[0], 10)
        self.assertEqual(h_matrix.shape[1], 10)
        
        # Check values are finite
        self.assertTrue(np.all(np.isfinite(h_matrix)))


class TestVonNeumannRelaxationVisualizer(unittest.TestCase):
    """Tests for the VonNeumannRelaxationVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a system and state
        self.L = 1.0
        self.system = InfiniteSquareWell2D(self.L)
        
        # Create a mixed state
        pure_state1 = PureState(self.L, [(1, 1, 1.0)])
        pure_state2 = PureState(self.L, [(2, 1, 1.0)])
        self.mixed_state = MixedState(self.L, [
            (pure_state1, 0.7),
            (pure_state2, 0.3)
        ])
        
        # Create a small relaxation simulation
        self.n_particles = 50
        self.t_max = 0.05
        self.dt = 0.01
        self.relaxation = BohmianRelaxation(
            self.system, self.mixed_state, self.n_particles
        )
        
        # Run a short simulation
        self.results = self.relaxation.run_simulation(self.t_max, self.dt)
        
        # Create visualizer
        self.visualizer = VonNeumannRelaxationVisualizer(
            self.relaxation, self.results, coarse_graining_levels=[4, 8]
        )
    
    def test_create_density_matrix_comparison(self):
        """Test density matrix comparison visualization."""
        # Create visualization for the first timestep
        fig = self.visualizer.create_density_matrix_comparison(0)
        
        # Check it's a matplotlib figure
        self.assertIsInstance(fig, Figure)
        
        # Check it has the right number of axes (2 rows, 3 columns)
        self.assertEqual(len(fig.axes), 6)
    
    def test_create_velocity_field_visualization(self):
        """Test velocity field visualization."""
        # Create visualization for the first timestep
        fig = self.visualizer.create_velocity_field_visualization(0)
        
        # Check it's a matplotlib figure
        self.assertIsInstance(fig, Figure)
        
        # Check it has one axis
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_h_function_evolution(self):
        """Test H-function evolution plot."""
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Plot H-function evolution
            self.visualizer.plot_h_function_evolution(tmpdirname)
            
            # Check file was created
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, 'h_function_evolution.png')))
    
    def test_plot_convergence_rate_vs_epsilon(self):
        """Test convergence rate plot."""
        # Calculate H-functions
        h_coarse_values = {}
        for cg_level in self.visualizer.coarse_graining_levels:
            h_cg = self.relaxation.calculate_h_function(coarse_grain=cg_level)
            h_coarse_values[cg_level] = h_cg
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Plot convergence rate
            self.visualizer.plot_convergence_rate_vs_epsilon(tmpdirname, h_coarse_values)
            
            # Check file was created
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, 'convergence_rate_vs_epsilon.png')))


if __name__ == '__main__':
    unittest.main()