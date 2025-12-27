/*
 * OpenMP-Accelerated Particle Simulation
 * ======================================
 * This C++ implementation uses OpenMP for parallel particle updates.
 * Compile with: g++ -fopenmp -O3 -shared -o particle_simulation_openmp.so particle_simulation_openmp.cpp
 * (On Windows with MinGW: g++ -fopenmp -O3 -shared -o particle_simulation_openmp.dll particle_simulation_openmp.cpp)
 */

#include <vector>
#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>

// Constants matching Python implementation
const float GRAVITY = 0.5f;
const float DT = 1.0f;
const float DAMPING = 0.8f;

// Particle system structure
struct ParticleSystem {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> vx;
    std::vector<float> vy;
    int num_particles;
    float width;
    float height;
    
    ParticleSystem(int n, float w, float h) 
        : num_particles(n), width(w), height(h) {
        x.resize(n);
        y.resize(n);
        vx.resize(n);
        vy.resize(n);
    }
};

// Initialize particle system with random values
extern "C" {
    ParticleSystem* create_particle_system(int num_particles, float width, float height) {
        ParticleSystem* ps = new ParticleSystem(num_particles, width, height);
        
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> x_dist(0.0f, width);
        std::uniform_real_distribution<float> y_dist(0.0f, height / 2.0f);
        std::uniform_real_distribution<float> vx_dist(-2.0f, 2.0f);
        std::uniform_real_distribution<float> vy_dist(0.0f, 2.0f);
        
        // Parallel initialization with OpenMP
        #pragma omp parallel for
        for (int i = 0; i < num_particles; i++) {
            ps->x[i] = x_dist(gen);
            ps->y[i] = y_dist(gen);
            ps->vx[i] = vx_dist(gen);
            ps->vy[i] = vy_dist(gen);
        }
        
        return ps;
    }
    
    // Update particles using OpenMP parallelization
    void update_particles(ParticleSystem* ps) {
        if (!ps) return;
        
        // Parallel update loop - each thread processes a subset of particles
        #pragma omp parallel for
        for (int i = 0; i < ps->num_particles; i++) {
            // Update velocity with gravity
            ps->vy[i] += GRAVITY * DT;
            
            // Update position
            ps->x[i] += ps->vx[i] * DT;
            ps->y[i] += ps->vy[i] * DT;
            
            // Boundary collision with damping
            // Left wall
            if (ps->x[i] < 0.0f) {
                ps->x[i] = 0.0f;
                ps->vx[i] *= -DAMPING;
            }
            // Right wall
            if (ps->x[i] > ps->width) {
                ps->x[i] = ps->width;
                ps->vx[i] *= -DAMPING;
            }
            // Top wall
            if (ps->y[i] < 0.0f) {
                ps->y[i] = 0.0f;
                ps->vy[i] *= -DAMPING;
            }
            // Bottom wall
            if (ps->y[i] > ps->height) {
                ps->y[i] = ps->height;
                ps->vy[i] *= -DAMPING;
            }
        }
    }
    
    // Get positions for rendering
    void get_positions(ParticleSystem* ps, float* x_out, float* y_out) {
        if (!ps) return;
        
        // Copy positions (could be parallelized but memory bandwidth limited)
        #pragma omp parallel for
        for (int i = 0; i < ps->num_particles; i++) {
            x_out[i] = ps->x[i];
            y_out[i] = ps->y[i];
        }
    }
    
    // Free particle system
    void free_particle_system(ParticleSystem* ps) {
        if (ps) {
            delete ps;
        }
    }
    
    // Get number of OpenMP threads
    int get_num_threads() {
        return omp_get_max_threads();
    }
    
    // Set number of OpenMP threads
    void set_num_threads(int num_threads) {
        omp_set_num_threads(num_threads);
    }
}

