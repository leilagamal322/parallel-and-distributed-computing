#!/usr/bin/env python
"""Quick performance comparison test"""
import sys
import os

# Setup
os.makedirs('plots', exist_ok=True)
sys.path.insert(0, 'src')

# Import
from performance_analysis import (
    compare_performance, 
    plot_speedup, 
    plot_efficiency, 
    plot_comparison,
    analyze_overheads,
    generate_report
)

# Small test
print("=" * 80)
print("Quick Performance Comparison Test")
print("=" * 80)
print()

# Run comparison with small particle counts for quick test
particle_counts = [100, 500, 1000, 5000]
print(f"Testing with {particle_counts} particles")
print("Using 100 frames per test for speed")
print()

try:
    results = compare_performance(
        particle_counts, 
        num_frames=100,
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
    print("Overhead Analysis")
    print("=" * 80)
    analyze_overheads(results)
    
    print("\n" + "=" * 80)
    print("Generating report...")
    print("=" * 80)
    generate_report(results)
    
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print("Check the 'plots' directory for generated graphs.")
    print("Check 'report_performance.txt' for the detailed report.")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

