import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision globally
jax.config.update("jax_enable_x64", True) 

# Import your updated functions
from core_functions import build_quasi2d_stokes_solver, build_trajectory_generator

if __name__ == "__main__":
    # 1. Fixed Simulation Parameters
    grid_limit = 64
    Lx, Ly, Lz = grid_limit, grid_limit, grid_limit
    
    # Time step obtained from temporal convergence test
    dt = 0.1  
    T_end = 20.0
    num_steps = int(round(T_end / dt))  # 200 steps
    eps = 0.5
    sigma_rep = 2 * eps
    r_cutoff = 3 * sigma_rep
    dimension = 2
    
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
    
    tracking_radius = r_cutoff + eps  
    particle_radius = 0.5 * sigma_rep
    
    if dimension == 2:
        # Max particles in tracking circle (assuming ~90% optimal hexagonal packing)
        max_neighbors = int(0.9 * (tracking_radius / particle_radius)**2)
    else: # 3D
        # Max particles in tracking sphere (assuming ~74% optimal spherical packing)
        max_neighbors = int(0.74 * (tracking_radius / particle_radius)**3)

    print("Starting Stiff Scattering Spatial Convergence Test...\n")
    print("-" * 75)
    print(f"Fixed dt: {dt} ({num_steps} steps)")
    print(f"Total Physical Time: {T_end}")
    print("-" * 75)

    # Run simulations
    for (Nx, Ny, Nz) in resolutions:
        print(f"Solving for resolution: {Nx}x{Ny}x{Nz}...")
        
        # Build solvers - Unpack both the solver and the neighbor generator
        solve_flow_fn, neighbor_fn = build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz)
        
        # Pre-allocate the neighbor list cell grid structure using the starting positions            
        nbrs_init = neighbor_fn.allocate(initial_positions, extra_capacity=max_neighbors)
        
        simulate_fn = build_trajectory_generator(
            Lx, Ly, dt, solve_flow_fn, sigmas_scatter
        )
        
        # Run simulation - Pass nbrs_init as required by the updated wrapper function
        pos_hist, _ = simulate_fn(initial_positions, initial_angles, nbrs_init, num_steps, key)
        
        # Save the ENTIRE history
        final_states[(Nx, Ny, Nz)] = pos_hist

    print("-" * 75)
    print("\nCalculating Spatial Convergence Metrics Across Entire Timeline...\n")

    # 4. Calculate Errors against the Ground Truth
    ground_truth_res = resolutions[-1]
    ground_truth_hist = final_states[ground_truth_res] # Shape: [200, 2, 2]

    print(f"Ground Truth Resolution: {ground_truth_res[0]}x{ground_truth_res[1]}x{ground_truth_res[2]}")
    print(f"{'Resolution':<15} | {'Max Peak Error':<18} | {'Time-Avg Error':<18} | {'Max Pack Diff (%)'}")
    print("-" * 80)

    for res in resolutions[:-1]:  # Skip the last one since it's the ground truth
        test_hist = final_states[res]
        
        # Minimum Image Convention for error calculation
        dx = jnp.abs(test_hist[:, :, 0] - ground_truth_hist[:, :, 0])
        dy = jnp.abs(test_hist[:, :, 1] - ground_truth_hist[:, :, 1])
        
        dx = jnp.where(dx > Lx / 2.0, Lx - dx, dx)
        dy = jnp.where(dy > Ly / 2.0, Ly - dy, dy)
        
        distances = jnp.sqrt(dx**2 + dy**2)
        
        # Mean error across the 2 swimmers at every single point in time
        error_over_time = jnp.mean(distances, axis=1)
        
        # Extract tracking metrics
        max_peak_error = jnp.max(error_over_time)
        time_avg_error = jnp.mean(error_over_time)
        
        # Calculate fractional difference relative to the swimmer size
        fractional_max_diff = (max_peak_error / sigma_rep) * 100
        
        # Print formatted results
        res_str = f"{res[0]}x{res[1]}x{res[2]}"
        print(f"{res_str:<15} | {max_peak_error.item():<18.6e} | {time_avg_error.item():<18.6e} | {fractional_max_diff.item():.6f} %")

    print("\nTest Complete.")