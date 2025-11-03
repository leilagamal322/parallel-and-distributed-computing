"""
Installation Test Script
========================
Verifies that all required dependencies are properly installed.
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing required package imports...")
    print("-" * 50)
    
    failed = []
    
    # Test NumPy
    try:
        import numpy as np
        print("OK NumPy:", np.__version__)
    except ImportError as e:
        print("X NumPy: FAILED -", str(e))
        failed.append("numpy")
    
    # Test Pygame
    try:
        import pygame
        print("OK Pygame:", pygame.version.ver)
    except ImportError as e:
        print("X Pygame: FAILED -", str(e))
        failed.append("pygame")
    
    # Test PyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("OK PyCUDA: OK")
        print(f"  GPU Device: {cuda.Device(0).name()}")
        print(f"  Compute Capability: {cuda.Device(0).compute_capability()}")
    except ImportError as e:
        print("X PyCUDA: FAILED -", str(e))
        print("  Note: PyCUDA requires CUDA Toolkit to be installed")
        failed.append("pycuda")
    except Exception as e:
        print("X PyCUDA: FAILED -", str(e))
        failed.append("pycuda")
    
    # Test Matplotlib
    try:
        import matplotlib
        print("OK Matplotlib:", matplotlib.__version__)
    except ImportError as e:
        print("X Matplotlib: FAILED -", str(e))
        failed.append("matplotlib")
    
    print("-" * 50)
    
    if failed:
        print(f"\nX Some packages failed to import: {', '.join(failed)}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\nOK All required packages are installed correctly!")
        return True


def test_project_files():
    """Test if project files exist."""
    print("\nTesting project file structure...")
    print("-" * 50)
    
    import os
    
    required_files = [
        'src/baseline_cpu_simulation.py',
        'src/gpu_simulation_pycuda.py',
        'src/performance_analysis.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"OK {file}")
        else:
            print(f"X {file} - MISSING")
            missing.append(file)
    
    print("-" * 50)
    
    if missing:
        print(f"\nX Some files are missing: {len(missing)} file(s)")
        return False
    else:
        print("\nOK All project files exist!")
        return True


if __name__ == "__main__":
    print("=" * 50)
    print("Particle Simulation - Installation Test")
    print("=" * 50)
    print()
    
    imports_ok = test_imports()
    files_ok = test_project_files()
    
    print("\n" + "=" * 50)
    if imports_ok and files_ok:
        print("OK Installation test PASSED!")
        print("\nYou can now run:")
        print("  python src/baseline_cpu_simulation.py")
        print("  python src/gpu_simulation_pycuda.py")
        print("  python src/performance_analysis.py")
    else:
        print("X Installation test FAILED!")
        print("\nPlease fix the issues above before proceeding.")
    print("=" * 50)