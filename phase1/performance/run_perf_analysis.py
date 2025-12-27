"""Run performance analysis"""
import sys
import os

# Setup paths
os.makedirs('plots', exist_ok=True)
sys.path.insert(0, 'src')

print("Importing modules...")
try:
    from performance_analysis import (
        compare_performance,
        plot_speedup,
        plot_efficiency, 
        plot_comparison,
        analyze_overheads,
        generate_report
    )
    print("Imports successful!")
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Configuration
particle_counts = [100, 500, 1000, 5000, 10000]
num_frames = 100  # Reduced for faster testing

print("\n" + "=" * 80)
print("Performance Analysis: CPU vs OpenMP vs GPU")
print("=" * 80)
print(f"Particle counts: {particle_counts}")
print(f"Frames per test: {num_frames}")
print("\nThis may take a few minutes...\n")

try:
    # Run comparison
    print("Running benchmarks...")
    results = compare_performance(
        particle_counts,
        num_frames=num_frames,
        include_cache_analysis=False,
        num_threads=4
    )
    
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    
    plot_speedup(results)
    plot_efficiency(results)
    plot_comparison(results)
    
    print("\n" + "=" * 80)
    print("Analyzing overheads...")
    print("=" * 80)
    analyze_overheads(results)
    
    print("\n" + "=" * 80)
    print("Generating report...")
    print("=" * 80)
    generate_report(results)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - plots/speedup_vs_particles.png")
    print("  - plots/efficiency_vs_particles.png")
    print("  - plots/cpu_vs_openmp_vs_gpu_comparison.png")
    print("  - report_performance.txt")
    print("\nCheck these files for detailed results!")
    
except KeyboardInterrupt:
    print("\n\nAnalysis interrupted by user.")
    sys.exit(1)
except Exception as e:
    print(f"\n\nERROR during analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

