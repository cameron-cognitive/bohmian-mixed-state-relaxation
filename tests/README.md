# Testing Documentation

This directory contains comprehensive tests for the Bohmian mixed-state relaxation simulation. The tests ensure that all components of the simulation work correctly, with special emphasis on velocity field calculations.

## Test Structure

The test suite is organized into multiple components:

1. **Basic Tests** (in `test_suite.py`)
   - `TestPureState`: Tests for pure quantum state functionality
   - `TestMixedState`: Tests for mixed quantum state functionality 
   - `TestInfiniteSquareWell2D`: Tests for the 2D infinite square well system
   - `TestBohmianRelaxation`: Tests for the relaxation simulation
   - `TestVonNeumannRelaxationVisualizer`: Tests for visualization tools

2. **Specialized Velocity Field Tests** (in `test_velocity_field.py`)
   - `VelocityFieldTests`: In-depth tests for quantum velocity fields
   - Includes analytical verification of velocity calculations
   - Tests time dependence, energy conservation, and mixed-state behavior

## Running Tests

You can run the tests using the provided `run_tests.py` script:

```bash
# Run all tests
python -m tests.run_tests

# Run only basic tests
python -m tests.run_tests --type basic

# Run only velocity field tests
python -m tests.run_tests --type velocity

# Run tests with minimal output
python -m tests.run_tests --quiet

# Save test results to a file
python -m tests.run_tests --output results/test_results.txt
```

## Key Test Cases

### Velocity Field Verification

The velocity field tests are particularly important as they verify the correctness of the quantum guiding equations:

1. **Ground State Test**: Ensures velocity is zero for ground state
2. **Superposition Tests**: Verifies velocity fields for various superpositions
3. **Analytical Comparison**: Compares velocity calculations with direct analytical formula
4. **Time Dependence**: Tests evolution of velocity field over time
5. **Mixed State Behavior**: Tests the von Neumann guidance for mixed states
6. **Energy Conservation**: Verifies that particle trajectories conserve energy

### Other Important Tests

1. **Wave Function Tests**: Ensures wave functions are correctly calculated
2. **Density Matrix Tests**: Validates density matrix calculations for mixed states
3. **H-Function Tests**: Checks relaxation metrics calculation
4. **Visualization Tests**: Ensures visualization tools work correctly

## Adding New Tests

When adding new quantum states, systems, or features to the simulation, you should add corresponding tests:

1. For new quantum states: Add test cases to `TestPureState` or `TestMixedState`
2. For new system types: Create a new test class similar to `TestInfiniteSquareWell2D`
3. For velocity field changes: Add test cases to `VelocityFieldTests`

## Continuous Integration

These tests are designed to be run in a CI/CD pipeline. You can configure your CI system to run:

```bash
python -m tests.run_tests --type all --quiet --output test-results.txt
```

This will perform all tests and save the results for reporting.