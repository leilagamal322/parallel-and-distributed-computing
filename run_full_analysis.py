"""Full performance analysis with plots"""
import sys
import os

# Set matplotlib to non-interactive backend before importing
import matplotlib
matplotlib.use('Agg')

# Setup
os.makedirs('plots', exist_ok=True)
sys.path.insert(0, 'src')

print("=" * 80)
print("FULL PERFORMANCE ANALYSIS")
print("=" * 80)
print()

try:
    from performance_analysis import (
        compare_performance,
        plot_speedup,
        plot_efficiency,
        plot_comparison,
        analyze_overheads,
        generate_report
    )
    print("[OK] Modules imported successfully")
except Exception as e:
    print(f"[ERROR] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Configuration
particle_counts = [100, 500, 1000, 5000, 10000]
num_frames = 100

print(f"Configuration:")
print(f"  Particle counts: {particle_counts}")
print(f"  Frames per test: {num_frames}")
print(f"  Threads: 4")
print()

# Run comparison
print("=" * 80)
print("Running benchmarks...")
print("=" * 80)
try:
    results = compare_performance(
        particle_counts,
        num_frames=num_frames,
        include_cache_analysis=False,
        num_threads=4
    )
    print("\n[OK] Benchmarks completed")
except Exception as e:
    print(f"[ERROR] Benchmark error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Generate plots
print("\n" + "=" * 80)
print("Generating plots...")
print("=" * 80)
try:
    plot_speedup(results)
    print("[OK] Speedup plot generated")
    
    plot_efficiency(results)
    print("[OK] Efficiency plot generated")
    
    plot_comparison(results)
    print("[OK] Comparison plot generated")
except Exception as e:
    print(f"[ERROR] Plot error: {e}")
    import traceback
    traceback.print_exc()

# Analyze overheads
print("\n" + "=" * 80)
print("Analyzing overheads...")
print("=" * 80)
try:
    analyze_overheads(results)
    print("[OK] Overhead analysis completed")
except Exception as e:
    print(f"[ERROR] Overhead analysis error: {e}")

# Generate report
print("\n" + "=" * 80)
print("Generating report...")
print("=" * 80)
try:
    generate_report(results)
    print("[OK] Report generated")
except Exception as e:
    print(f"[ERROR] Report error: {e}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - plots/speedup_vs_particles.png")
print("  - plots/efficiency_vs_particles.png")
print("  - plots/cpu_vs_openmp_vs_gpu_comparison.png")
print("  - report_performance.txt")
print("\nCheck these files for detailed results!")

