#!/usr/bin/env python3
"""
Test runner for the Bohmian mixed-state relaxation simulation.

This script runs all tests for the Bohmian mechanics simulation,
including specific tests for velocity field calculations.
"""

import unittest
import sys
import os
import argparse
from datetime import datetime

# Import test modules
from tests.test_suite import (
    TestPureState,
    TestMixedState,
    TestInfiniteSquareWell2D,
    TestBohmianRelaxation,
    TestVonNeumannRelaxationVisualizer
)
from tests.test_velocity_field import VelocityFieldTests


def run_tests(test_type=None, verbose=True):
    """Run specified tests with optional verbosity.
    
    Args:
        test_type: Optional test type to run ('basic', 'velocity', or 'all')
        verbose: Whether to show verbose output
    
    Returns:
        Result of the test run (success or failure)
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    if test_type == 'basic' or test_type == 'all' or test_type is None:
        # Add basic tests
        suite.addTest(unittest.makeSuite(TestPureState))
        suite.addTest(unittest.makeSuite(TestMixedState))
        suite.addTest(unittest.makeSuite(TestInfiniteSquareWell2D))
        suite.addTest(unittest.makeSuite(TestBohmianRelaxation))
        suite.addTest(unittest.makeSuite(TestVonNeumannRelaxationVisualizer))
    
    if test_type == 'velocity' or test_type == 'all':
        # Add specialized velocity field tests
        suite.addTest(unittest.makeSuite(VelocityFieldTests))
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run tests for Bohmian mixed-state relaxation simulation')
    parser.add_argument('--type', choices=['basic', 'velocity', 'all'], default='all',
                      help='Type of tests to run (default: all)')
    parser.add_argument('--quiet', action='store_true',
                      help='Run tests with minimal output')
    parser.add_argument('--output', type=str,
                      help='Path to save test results (optional)')
    
    args = parser.parse_args()
    
    # Print test header
    if not args.quiet:
        print(f"=== Running {args.type} tests for Bohmian Mixed-State Relaxation ===")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    # Run tests
    success = run_tests(args.type, verbose=not args.quiet)
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(f"Test run: {args.type}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Result: {'Success' if success else 'Failure'}\n")
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())