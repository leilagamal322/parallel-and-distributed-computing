"""Check MPI setup and provide installation instructions."""
import sys

print("Checking MPI setup...")
print("="*70)

# Check mpi4py
try:
    import mpi4py
    print("[OK] mpi4py package is installed")
    try:
        from mpi4py import MPI
        version = MPI.Get_version()
        print(f"  MPI version: {version[0]}.{version[1]}")
        comm = MPI.COMM_WORLD
        print(f"  MPI Size: {comm.Get_size()}")
        print(f"  MPI Rank: {comm.Get_rank()}")
        print("\n[OK] MPI runtime is available!")
        print("You can run the experiments with:")
        print("  mpirun -n 2 python src/mpi_performance_experiments.py --latency-bandwidth")
        sys.exit(0)
    except RuntimeError as e:
        print(f"[X] MPI runtime not found")
        print("  mpi4py is installed but cannot load MPI library")
except ImportError:
    print("[X] mpi4py is not installed")
    print("  Install with: pip install mpi4py")

print("\n" + "="*70)
print("MPI INSTALLATION INSTRUCTIONS FOR WINDOWS:")
print("="*70)
print("""
1. Install MS-MPI:
   - Download from: https://www.microsoft.com/en-us/download/details.aspx?id=57467
   - Install both:
     * msmpisdk.msi (SDK)
     * msmpisetup.exe (Runtime)

2. Add to PATH (if not automatic):
   - C:\\Program Files\\Microsoft MPI\\Bin
   - C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Bin

3. Verify installation:
   - Open new terminal
   - Run: mpiexec --version

4. Reinstall mpi4py (if needed):
   - pip uninstall mpi4py
   - pip install mpi4py

Alternative: Use WSL (Windows Subsystem for Linux) and install OpenMPI there.
""")

