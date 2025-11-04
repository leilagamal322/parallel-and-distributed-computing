"""
Quick performance test script
"""
import sys
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Add src to path
sys.path.insert(0, 'src')

# Import with error handling
try:
    from performance_analysis import compare_performance, plot_speedup, plot_efficiency, plot_comparison, analyze_overheads, generate_report
    print("Imports successful!")
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    # Use smaller particle counts for faster testing
    particle_counts = [100, 500, 1000, 5000]
    
    print("\nStarting performance comparison...")
    print("This will test CPU, OpenMP, and GPU (if available)")
    print("=" * 80)
    
    try:
        # Run comparison (without cache analysis for speed)
        results = compare_performance(
            particle_counts, 
            num_frames=100,  # Reduced frames for faster testing
            include_cache_analysis=False,
            num_threads=None
        )
        
        # Generate plots
        print("\nGenerating plots...")
        plot_speedup(results)
        plot_efficiency(results)
        plot_comparison(results)
        
        # Analyze overheads
        analyze_overheads(results)
        
        # Generate text report
        generate_report(results)
        
        print("\n" + "=" * 80)
        print("Performance analysis complete!")
        print("Check the 'plots' directory for generated graphs.")
        print("Check 'report_performance.txt' for the detailed report.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

