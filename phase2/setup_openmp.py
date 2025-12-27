"""
Setup script to compile the OpenMP C++ extension.
This script tries to compile the particle_simulation_openmp.cpp file.
"""

import os
import sys
import subprocess
import platform

def compile_openmp_extension():
    """Compile the OpenMP C++ extension."""
    src_file = os.path.join('src', 'particle_simulation_openmp.cpp')
    
    if not os.path.exists(src_file):
        print(f"ERROR: Source file not found: {src_file}")
        return False
    
    # Determine output file based on platform
    if platform.system() == 'Windows':
        output_file = os.path.join('src', 'particle_simulation_openmp.dll')
        compile_cmd = [
            'g++',
            '-fopenmp',
            '-O3',
            '-shared',
            '-o', output_file,
            src_file
        ]
    elif platform.system() == 'Darwin':
        output_file = os.path.join('src', 'libparticle_simulation_openmp.dylib')
        compile_cmd = [
            'g++',
            '-fopenmp',
            '-O3',
            '-shared',
            '-fPIC',
            '-o', output_file,
            src_file
        ]
    else:  # Linux
        output_file = os.path.join('src', 'libparticle_simulation_openmp.so')
        compile_cmd = [
            'g++',
            '-fopenmp',
            '-O3',
            '-shared',
            '-fPIC',
            '-o', output_file,
            src_file
        ]
    
    print("=" * 70)
    print("Compiling OpenMP C++ Extension")
    print("=" * 70)
    print(f"Source: {src_file}")
    print(f"Output: {output_file}")
    print(f"Command: {' '.join(compile_cmd)}")
    print()
    
    try:
        result = subprocess.run(
            compile_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("SUCCESS: OpenMP extension compiled successfully!")
        print(f"Output file: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print("ERROR: Compilation failed!")
        print(f"Return code: {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        print("\nPossible issues:")
        print("1. g++ not found in PATH")
        print("2. OpenMP not supported by your compiler")
        print("3. Missing development libraries")
        print("\nTry installing:")
        if platform.system() == 'Windows':
            print("  - MinGW-w64 with OpenMP support")
            print("  - Or use MSVC with OpenMP")
        else:
            print("  - g++ compiler with OpenMP support")
            print("  - On Ubuntu/Debian: sudo apt-get install g++ libomp-dev")
            print("  - On macOS: brew install libomp")
        return False
    except FileNotFoundError:
        print("ERROR: g++ compiler not found!")
        print("\nPlease install a C++ compiler with OpenMP support:")
        if platform.system() == 'Windows':
            print("  - Install MinGW-w64")
            print("  - Or install MSVC (Visual Studio)")
        else:
            print("  - Install g++: sudo apt-get install g++")
        return False

if __name__ == "__main__":
    success = compile_openmp_extension()
    sys.exit(0 if success else 1)

