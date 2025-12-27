"""
Cache Analysis Module
====================
Analyzes cache performance for CPU particle simulation.
Estimates cache hits/misses based on memory access patterns and data structures.
"""

import numpy as np
import sys
import platform

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class CacheAnalyzer:
    """
    Analyzes cache performance for particle simulation.
    Estimates cache hits/misses based on memory layout and access patterns.
    """
    
    # Typical cache sizes (bytes) - these are estimates for modern CPUs
    # L1 cache: 32-64 KB per core
    # L2 cache: 256 KB - 1 MB per core
    # L3 cache: 8-32 MB shared
    L1_CACHE_SIZE = 32 * 1024  # 32 KB
    L2_CACHE_SIZE = 256 * 1024  # 256 KB
    L3_CACHE_SIZE = 8 * 1024 * 1024  # 8 MB
    
    # Cache line size (typically 64 bytes)
    CACHE_LINE_SIZE = 64
    
    # Number of floats that fit in cache line
    FLOATS_PER_CACHE_LINE = CACHE_LINE_SIZE // 4  # 4 bytes per float32
    
    def __init__(self):
        """Initialize cache analyzer with system-specific cache info if available."""
        self.system_cache_info = self._get_system_cache_info()
        
    def _get_system_cache_info(self):
        """Try to get cache information from system."""
        cache_info = {
            'l1_size': self.L1_CACHE_SIZE,
            'l2_size': self.L2_CACHE_SIZE,
            'l3_size': self.L3_CACHE_SIZE,
            'cache_line_size': self.CACHE_LINE_SIZE,
            'source': 'estimated'
        }
        
        if PSUTIL_AVAILABLE:
            try:
                # psutil doesn't directly expose cache info, but we can get CPU info
                cpu_info = psutil.cpu_count(logical=False)
                cache_info['cpu_cores'] = cpu_info
            except:
                pass
        
        return cache_info
    
    def analyze_memory_layout(self, particle_system):
        """
        Analyze memory layout of particle system arrays.
        
        Args:
            particle_system: ParticleSystemCPU instance
            
        Returns:
            Dictionary with memory layout analysis
        """
        arrays = {
            'x': particle_system.x,
            'y': particle_system.y,
            'vx': particle_system.vx,
            'vy': particle_system.vy,
            'ax': particle_system.ax,
            'ay': particle_system.ay
        }
        
        total_size = 0
        contiguous_count = 0
        layout_info = {}
        
        for name, arr in arrays.items():
            size_bytes = arr.nbytes
            total_size += size_bytes
            is_contiguous = arr.flags['C_CONTIGUOUS']
            stride = arr.strides[0] if arr.ndim > 0 else 0
            
            layout_info[name] = {
                'size_bytes': size_bytes,
                'size_elements': arr.size,
                'is_contiguous': is_contiguous,
                'stride': stride,
                'dtype': arr.dtype
            }
            
            if is_contiguous:
                contiguous_count += 1
        
        return {
            'arrays': layout_info,
            'total_size_bytes': total_size,
            'total_size_kb': total_size / 1024,
            'total_size_mb': total_size / (1024 * 1024),
            'num_arrays': len(arrays),
            'contiguous_arrays': contiguous_count,
            'all_contiguous': contiguous_count == len(arrays)
        }
    
    def estimate_cache_behavior(self, particle_system, num_particles):
        """
        Estimate cache hits and misses based on memory access patterns.
        
        Args:
            particle_system: ParticleSystemCPU instance
            num_particles: Number of particles
            
        Returns:
            Dictionary with cache hit/miss estimates
        """
        layout = self.analyze_memory_layout(particle_system)
        
        # Calculate working set size
        # Each particle update accesses: x, y, vx, vy, ax, ay (6 arrays)
        # Each element is 4 bytes (float32)
        elements_per_particle = 6
        bytes_per_particle = elements_per_particle * 4
        working_set_size = num_particles * bytes_per_particle
        
        # Estimate cache misses based on working set size
        # Sequential access pattern should have good spatial locality
        # Multiple arrays accessed per particle (x, y, vx, vy, ax, ay)
        # This creates cache conflicts but sequential access helps
        
        # Calculate how many cache lines are needed
        cache_lines_needed = (working_set_size + self.CACHE_LINE_SIZE - 1) // self.CACHE_LINE_SIZE
        
        # Estimate hits/misses based on cache hierarchy
        # L1 cache analysis
        if working_set_size <= self.L1_CACHE_SIZE:
            # Fits in L1 - high hit rate
            l1_hits_est = int(num_particles * 0.95)  # 95% hit rate
            l1_misses_est = num_particles - l1_hits_est
            l1_hit_rate = 0.95
        elif working_set_size <= self.L2_CACHE_SIZE:
            # Fits in L2 - moderate hit rate
            l1_hits_est = int(num_particles * 0.70)  # 70% L1 hit
            l1_misses_est = num_particles - l1_hits_est
            l2_hits_est = int(l1_misses_est * 0.90)  # 90% L2 hit on L1 miss
            l2_misses_est = l1_misses_est - l2_hits_est
            l1_hit_rate = 0.70
            l2_hit_rate = 0.90
        else:
            # May not fit in L2 - lower hit rates
            l1_hits_est = int(num_particles * 0.50)  # 50% L1 hit
            l1_misses_est = num_particles - l1_hits_est
            l2_hits_est = int(l1_misses_est * 0.70)  # 70% L2 hit
            l2_misses_est = l1_misses_est - l2_hits_est
            l3_hits_est = int(l2_misses_est * 0.80)  # 80% L3 hit
            l3_misses_est = l2_misses_est - l3_hits_est
            l1_hit_rate = 0.50
            l2_hit_rate = 0.70
            l3_hit_rate = 0.80
        
        # Calculate cache efficiency metrics
        # Sequential access pattern has good spatial locality
        # But accessing 6 different arrays per particle creates some cache conflicts
        
        # Spatial locality: accessing consecutive elements in arrays
        spatial_locality_score = 0.85  # Good - sequential access
        
        # Temporal locality: same particles accessed each frame
        temporal_locality_score = 0.90  # Excellent - same data reused
        
        # Overall cache friendliness
        cache_friendliness = (spatial_locality_score + temporal_locality_score) / 2
        
        result = {
            'num_particles': num_particles,
            'working_set_size_bytes': working_set_size,
            'working_set_size_kb': working_set_size / 1024,
            'working_set_size_mb': working_set_size / (1024 * 1024),
            'cache_lines_needed': cache_lines_needed,
            'l1_cache_size': self.L1_CACHE_SIZE,
            'l2_cache_size': self.L2_CACHE_SIZE,
            'l3_cache_size': self.L3_CACHE_SIZE,
            'fits_in_l1': working_set_size <= self.L1_CACHE_SIZE,
            'fits_in_l2': working_set_size <= self.L2_CACHE_SIZE,
            'fits_in_l3': working_set_size <= self.L3_CACHE_SIZE,
            'spatial_locality_score': spatial_locality_score,
            'temporal_locality_score': temporal_locality_score,
            'cache_friendliness': cache_friendliness,
            'layout_info': layout
        }
        
        # Add hit/miss estimates
        if working_set_size <= self.L1_CACHE_SIZE:
            result['l1_hits_est'] = l1_hits_est
            result['l1_misses_est'] = l1_misses_est
            result['l1_hit_rate'] = l1_hit_rate
            result['total_hits'] = l1_hits_est
            result['total_misses'] = l1_misses_est
        elif working_set_size <= self.L2_CACHE_SIZE:
            result['l1_hits_est'] = l1_hits_est
            result['l1_misses_est'] = l1_misses_est
            result['l1_hit_rate'] = l1_hit_rate
            result['l2_hits_est'] = l2_hits_est
            result['l2_misses_est'] = l2_misses_est
            result['l2_hit_rate'] = l2_hit_rate
            result['total_hits'] = l1_hits_est + l2_hits_est
            result['total_misses'] = l2_misses_est
        else:
            result['l1_hits_est'] = l1_hits_est
            result['l1_misses_est'] = l1_misses_est
            result['l1_hit_rate'] = l1_hit_rate
            result['l2_hits_est'] = l2_hits_est
            result['l2_misses_est'] = l2_misses_est
            result['l2_hit_rate'] = l2_hit_rate
            result['l3_hits_est'] = l3_hits_est
            result['l3_misses_est'] = l3_misses_est
            result['l3_hit_rate'] = l3_hit_rate
            result['total_hits'] = l1_hits_est + l2_hits_est + l3_hits_est
            result['total_misses'] = l3_misses_est
        
        # Calculate hit rate
        total_accesses = num_particles * elements_per_particle
        result['total_accesses'] = total_accesses
        result['overall_hit_rate'] = result['total_hits'] / total_accesses if total_accesses > 0 else 0
        result['overall_miss_rate'] = result['total_misses'] / total_accesses if total_accesses > 0 else 0
        
        return result
    
    def analyze_cache_performance(self, particle_system, num_particles):
        """
        Comprehensive cache performance analysis.
        
        Args:
            particle_system: ParticleSystemCPU instance
            num_particles: Number of particles
            
        Returns:
            Dictionary with comprehensive cache analysis
        """
        cache_behavior = self.estimate_cache_behavior(particle_system, num_particles)
        layout = self.analyze_memory_layout(particle_system)
        
        # Calculate cache efficiency
        # Ideal: all data fits in L1
        # Good: fits in L2
        # Acceptable: fits in L3
        # Poor: requires main memory
        
        if cache_behavior['fits_in_l1']:
            efficiency_level = 'Excellent'
            efficiency_score = 1.0
        elif cache_behavior['fits_in_l2']:
            efficiency_level = 'Good'
            efficiency_score = 0.8
        elif cache_behavior['fits_in_l3']:
            efficiency_level = 'Acceptable'
            efficiency_score = 0.6
        else:
            efficiency_level = 'Poor'
            efficiency_score = 0.4
        
        # Recommendations
        recommendations = []
        
        if not cache_behavior['fits_in_l3']:
            recommendations.append("Working set exceeds L3 cache - consider reducing particle count or using GPU")
        
        if not layout['all_contiguous']:
            recommendations.append("Some arrays are not contiguous - memory layout could be optimized")
        
        if cache_behavior['cache_friendliness'] < 0.7:
            recommendations.append("Cache friendliness is low - consider restructuring data access patterns")
        
        if cache_behavior['spatial_locality_score'] < 0.8:
            recommendations.append("Spatial locality could be improved - consider array-of-structures instead of structure-of-arrays")
        
        if not recommendations:
            recommendations.append("Cache performance is optimal for current working set size")
        
        return {
            'cache_behavior': cache_behavior,
            'layout_analysis': layout,
            'efficiency_level': efficiency_level,
            'efficiency_score': efficiency_score,
            'recommendations': recommendations,
            'system_cache_info': self.system_cache_info
        }


def benchmark_cache_performance(num_particles_list=[100, 500, 1000, 5000, 10000, 50000]):
    """
    Benchmark cache performance for different particle counts.
    
    Args:
        num_particles_list: List of particle counts to test
        
    Returns:
        Dictionary mapping particle counts to cache analysis results
    """
    from baseline_cpu_simulation import ParticleSystemCPU
    
    analyzer = CacheAnalyzer()
    results = {}
    
    print("=" * 80)
    print("Cache Performance Analysis")
    print("=" * 80)
    print(f"System Cache Info: {analyzer.system_cache_info}")
    print()
    
    for num_particles in num_particles_list:
        print(f"Analyzing cache performance for {num_particles} particles...")
        
        # Create particle system
        particle_system = ParticleSystemCPU(num_particles)
        
        # Perform cache analysis
        cache_analysis = analyzer.analyze_cache_performance(particle_system, num_particles)
        
        results[num_particles] = cache_analysis
        
        # Print summary
        cb = cache_analysis['cache_behavior']
        print(f"  Working Set: {cb['working_set_size_kb']:.2f} KB ({cb['working_set_size_mb']:.4f} MB)")
        print(f"  Fits in L1: {cb['fits_in_l1']}, L2: {cb['fits_in_l2']}, L3: {cb['fits_in_l3']}")
        print(f"  Overall Hit Rate: {cb['overall_hit_rate']*100:.2f}%")
        print(f"  Cache Friendliness: {cb['cache_friendliness']*100:.1f}%")
        print(f"  Efficiency: {cache_analysis['efficiency_level']} ({cache_analysis['efficiency_score']*100:.0f}%)")
        print()
    
    return results


if __name__ == "__main__":
    # Run cache analysis benchmark
    results = benchmark_cache_performance()
    
    print("\n" + "=" * 80)
    print("Cache Analysis Summary")
    print("=" * 80)
    
    for num_particles, analysis in results.items():
        cb = analysis['cache_behavior']
        print(f"\n{num_particles} particles:")
        print(f"  Hit Rate: {cb['overall_hit_rate']*100:.2f}%")
        print(f"  Miss Rate: {cb['overall_miss_rate']*100:.2f}%")
        if 'l1_hits_est' in cb:
            print(f"  L1 Hits: {cb['l1_hits_est']}, Misses: {cb['l1_misses_est']}")
        if 'l2_hits_est' in cb:
            print(f"  L2 Hits: {cb.get('l2_hits_est', 0)}, Misses: {cb.get('l2_misses_est', 0)}")
        if 'l3_hits_est' in cb:
            print(f"  L3 Hits: {cb.get('l3_hits_est', 0)}, Misses: {cb.get('l3_misses_est', 0)}")
        print(f"  Recommendations: {', '.join(analysis['recommendations'])}")
