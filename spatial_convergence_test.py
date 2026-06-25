import jax

# Enable 64-bit precision to prevent round-off error accumulation
jax.config.update("jax_enable_x64", True) 

import jax.numpy as jnp
import numpy as np

# Import your original functions
from core_functions import build_quasi2d_stokes_solver, build_trajectory_generator

if __name__ == "__main__":
    # 1. Fixed Simulation Parameters
    grid_limit = 64
    Lx, Ly, Lz = grid_limit, grid_limit, grid_limit
    
    # Time step obtained from temporal convergence test
    dt = 0.1  
    T_end = 20.0
    num_steps = int(round(T_end / dt))  # 1600 steps
    
    # 2. Physics & Kinematics Setup (Stiff Scattering)
    N_swimmers = 2
    sigmas_scatter = jnp.array([5, 5])  # Two strong Pushers

    initial_positions = jnp.array([
        [0.4, 0.5 * grid_limit],  # Swimmer 1: Left side
        [0.6 * grid_limit, 0.5 * grid_limit]   # Swimmer 2: Right side
    ])

    initial_angles = jnp.array([
        0.25 * jnp.pi,   # Swimmer 1 pointing upper right
        0.75 * jnp.pi    # Swimmer 2 pointing upper left
    ])

    key = jax.random.PRNGKey(42)

    # 3. Resolutions to test (Nx, Ny, Nz) - The LAST one is our "Ground Truth"
    resolutions = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256), 
        (512, 512, 512)  # Treated as the pseudo-analytical ground truth
    ]

    # Dictionary to hold the trajectory history for each resolution
    final_states = {}

    print("Starting Stiff Scattering Spatial Convergence Test...\n")
    print("-" * 75)
    print(f"Fixed dt: {dt} ({num_steps} steps)")
    print(f"Total Physical Time: {T_end}")
    print("-" * 75)

    # Run simulations
    for (Nx, Ny, Nz) in resolutions:
        print(f"Solving for resolution: {Nx}x{Ny}x{Nz}...")
        
        # Build solvers
        solve_flow_fn = build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz)
        simulate_fn = build_trajectory_generator(
            Lx, Ly, dt, solve_flow_fn, sigmas_scatter
        )
        
        # Run simulation
        pos_hist, _ = simulate_fn(initial_positions, initial_angles, num_steps, key)
        
        # Save the ENTIRE history
        final_states[(Nx, Ny, Nz)] = pos_hist

    print("-" * 75)
    print("\nCalculating Spatial Convergence Metrics Across Entire Timeline...\n")

    # 4. Calculate Errors against the Ground Truth
    ground_truth_res = resolutions[-1]
    ground_truth_hist = final_states[ground_truth_res] # Shape: [1600, 2, 2]

    print(f"Ground Truth Resolution: {ground_truth_res[0]}x{ground_truth_res[1]}x{ground_truth_res[2]}")
    print(f"{'Resolution':<15} | {'Max Peak Error':<18} | {'Time-Avg Error':<18} | {'Max Pack Diff (%)'}")
    print("-" * 80)

    for res in resolutions[:-1]:  # Skip the last one since it's the ground truth
        test_hist = final_states[res] # Shape: [1600, 2, 2]
        
        # Because dt is fixed, we don't need to stride! We can subtract directly.
        dx = jnp.abs(test_hist[:, :, 0] - ground_truth_hist[:, :, 0])
        dy = jnp.abs(test_hist[:, :, 1] - ground_truth_hist[:, :, 1])
        
        # Minimum Image Convention for error calculation
        dx = jnp.where(dx > Lx / 2.0, Lx - dx, dx)
        dy = jnp.where(dy > Ly / 2.0, Ly - dy, dy)
        
        distances = jnp.sqrt(dx**2 + dy**2)
        
        # Mean error across the 2 swimmers at every single point in time
        error_over_time = jnp.mean(distances, axis=1)  # Shape: [1600]
        
        # Extract tracking metrics
        max_peak_error = jnp.max(error_over_time)
        time_avg_error = jnp.mean(error_over_time)
        
        # Calculate fractional difference relative to domain size (Lx)
        fractional_max_diff = (max_peak_error / 1) * 100
        
        # Print formatted results
        res_str = f"{res[0]}x{res[1]}x{res[2]}"
        print(f"{res_str:<15} | {max_peak_error.item():<18.6e} | {time_avg_error.item():<18.6e} | {fractional_max_diff.item():.6f} %")

    print("\nTest Complete.")