import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision globally
jax.config.update("jax_enable_x64", True) 

# Import the updated unsteady functions
from core_functions_unsteady import build_quasi2d_stokes_solver, build_trajectory_generator

if __name__ == "__main__":
    # 1. Fixed Spatial Parameters
    grid_resolution = 128
    grid_limit = 64
    Lx, Ly, Lz = grid_limit, grid_limit, grid_limit
    Nx, Ny, Nz = grid_resolution, grid_resolution, grid_resolution
    
    # 2. Physics & Kinematics Setup
    N_swimmers = 2
    Dr_scatter = 0.0  # Zero noise to isolate hydrodynamics
    v0 = 0.5
    
    # --- UNSTEADY EXTREME CASE ---
    # Test the highest amplitude from parameter sweep to ensure dt is safe
    p_const_val = 0.125
    p_amp_val = 100.0 * p_const_val 
    omega = 2.0 * jnp.pi
    
    sigma_const = jnp.array([p_const_val, p_const_val])
    sigma_amp = jnp.array([p_amp_val, p_amp_val])
    phases = jnp.array([0.0, 0.0]) # Synchronized phases, as requested in your production script
    
    eps = 0.5
    sigma_rep = 2.0 * eps
    r_cutoff = 3.0 * sigma_rep
    
    # Total physical time (20 time units allows the swimmers to cross paths)
    T_end = 20.0 
    
    # 3. Setup Initial Conditions: Swimming towards each other
    initial_positions = jnp.array([
        [0.4 * grid_limit, 0.5 * grid_limit],  # Swimmer 1: Left side
        [0.6 * grid_limit, 0.5 * grid_limit]   # Swimmer 2: Right side
    ])

    initial_angles = jnp.array([
        0.25 * jnp.pi,   # Swimmer 1 pointing upper right
        0.75 * jnp.pi    # Swimmer 2 pointing upper left
    ])

    key = jax.random.PRNGKey(42)

    # 4. Time steps (dt) to test. The SMALLEST dt is our "Ground Truth"
    dt_values = [
        1e-1, 
        5e-2, 
        2.5e-2, 
        1.25e-2, 
        6.25e-3  # Ground truth
    ]

    final_states = {}

    print("Starting Stiff Scattering Temporal Convergence Test (Unsteady)...\n")
    print("-" * 65)
    print(f"Fixed Grid Resolution: {Nx}x{Ny}x{Nz}")
    print(f"Total Physical Time: {T_end}")
    print(f"Testing extreme amplitude: {p_amp_val} (100x baseline)")
    print("-" * 65)

    # 5. Build the spatial solver and neighbor list ONCE
    solve_flow_fn, neighbor_fn = build_quasi2d_stokes_solver(
        Lx, Ly, Lz, Nx, Ny, Nz, mu=1.0, eps=eps, v0=v0
    )
    
    # Pre-allocate neighbor list using 2D packing physics
    tracking_radius = r_cutoff + eps  
    particle_radius = 0.5 * sigma_rep
    max_neighbors = int(0.9 * (tracking_radius / particle_radius)**2)
    
    nbrs_init = neighbor_fn.allocate(initial_positions, extra_capacity=max_neighbors)

    # 6. Run simulations across different time steps
    for dt in dt_values:
        num_steps = int(round(T_end / dt))
        print(f"Solving for dt = {dt:<8} ({num_steps} steps)...")
        
        # Build trajectory generator for this specific dt with unsteady arrays
        generate_trajectory = build_trajectory_generator(
            Lx, Ly, dt, solve_flow_fn, 
            sigma_const, sigma_amp, omega, phases, 
            v0=v0, Dr=Dr_scatter
        )
        
        # Run simulation
        pos_hist, _ = generate_trajectory(
            initial_positions, initial_angles, nbrs_init, num_steps, key
        )
        
        # Extract all positions over time
        final_states[dt] = pos_hist

    print("-" * 65)
    print("\nCalculating Convergence Metrics Across Entire Timeline...\n")

    ground_truth_dt = dt_values[-1]
    ground_truth_hist = final_states[ground_truth_dt] 

    print(f"Ground Truth dt: {ground_truth_dt}")
    print(f"{'dt':<10} | {'Max Peak Error':<18} | {'Time-Avg Error':<18} | {'Max Pack Diff (%)'}")
    print("-" * 75)

    for dt in dt_values[:-1]:
        test_hist = final_states[dt] 
        
        # Calculate the stride factor to sync time steps
        stride = int(round(dt / ground_truth_dt))
        
        # Downsample the ground truth history to match the test history's timestamps
        matched_gt = ground_truth_hist[::stride]
        
        # Handle any trailing frame rounding mismatches safely
        matched_gt = matched_gt[:test_hist.shape[0]]
        
        # Calculate spatial differences across ALL frames and ALL swimmers
        dx = jnp.abs(test_hist[:, :, 0] - matched_gt[:, :, 0])
        dy = jnp.abs(test_hist[:, :, 1] - matched_gt[:, :, 1])
        
        # Periodic Boundary Conditions (Minimum Image Convention)
        dx = jnp.where(dx > Lx / 2.0, Lx - dx, dx)
        dy = jnp.where(dy > Ly / 2.0, Ly - dy, dy)
        
        distances = jnp.sqrt(dx**2 + dy**2)
        
        # Mean error across the 2 swimmers at every single point in time
        error_over_time = jnp.mean(distances, axis=1)
        
        # Extract key tracking metrics
        max_peak_error = jnp.max(error_over_time)
        time_avg_error = jnp.mean(error_over_time)
        
        # Calculate fractional difference relative to the physical swimmer size (sigma_rep)
        fractional_max_diff = (max_peak_error / sigma_rep) * 100
        
        # Print formatted results
        print(f"{dt:<10.5f} | {max_peak_error.item():<18.6e} | {time_avg_error.item():<18.6e} | {fractional_max_diff.item():.6f} %")

    print("\nTest Complete.")