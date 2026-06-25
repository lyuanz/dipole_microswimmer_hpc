import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Import your original functions
from core_functions import build_quasi2d_stokes_solver, build_trajectory_generator

if __name__ == "__main__":
    # 1. Fixed Spatial Parameters
    grid_resolution = 256
    grid_limit = 64
    Lx, Ly, Lz = grid_limit, grid_limit, grid_limit
    Nx, Ny, Nz = grid_resolution, grid_resolution, grid_resolution
    
    # 2. Physics & Kinematics Setup
    N_swimmers = 2
    Dr_scatter = 0.0  # Zero noise to isolate hydrodynamics
    sigmas_scatter = jnp.array([5, 5])  # Two strong Pushers
    
    # Total physical time based on your previous dt=0.1 and total_steps=200
    T_end = 20.0 
    
    # 3. Setup Initial Conditions: Swimming towards each other
    # I shifted the Y-coordinate to the middle of the domain (0.5 * grid_limit) 
    # just to keep the interaction safely away from the exact boundaries, 
    # though your periodic conditions handle it either way!
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
        0.1, 
        0.05, 
        0.025, 
        0.0125, 
        0.00625  # Ground truth
    ]

    final_states = {}

    print("Starting Stiff Scattering Temporal Convergence Test...\n")
    print("-" * 55)
    print(f"Fixed Grid Resolution: {Nx}x{Ny}x{Nz}")
    print(f"Total Physical Time: {T_end}")
    print("-" * 55)

    # Build the spatial solver ONCE
    solver = build_quasi2d_stokes_solver(
        Lx=Lx, Ly=Ly, Lz=Lz, 
        Nx=Nx, Ny=Ny, Nz=Nz
    )

    # Run simulations
    for dt in dt_values:
        num_steps = int(round(T_end / dt))
        print(f"Solving for dt = {dt:<8} ({num_steps} steps)...")
        
        # Build trajectory generator for this specific dt
        generate_trajectory = build_trajectory_generator(
            Lx=Lx, Ly=Ly, dt=dt, 
            solve_flow_fn=solver, sigmas=sigmas_scatter, 
            Dr=Dr_scatter
        )
        
        # Run simulation
        pos_hist, _ = generate_trajectory(
            initial_positions, initial_angles, num_steps=num_steps, prng_key=key
        )
        
        # Extract the all positions over time
        final_states[dt] = pos_hist

    print("-" * 55)
    print("\nCalculating Convergence Metrics Across Entire Timeline...\n")

    ground_truth_dt = dt_values[-1]
    ground_truth_hist = final_states[ground_truth_dt]  # Shape: [3200, 2, 2]

    print(f"Ground Truth dt: {ground_truth_dt}")
    print(f"{'dt':<10} | {'Max Peak Error':<18} | {'Time-Avg Error':<18} | {'Max Pack Diff (%)'}")
    print("-" * 75)

    for dt in dt_values[:-1]:
        test_hist = final_states[dt]  # Shape: [num_steps, 2, 2]
        
        # 1. Calculate the stride factor to sync time steps
        # e.g., if dt=0.1 and gt_dt=0.003125, stride = 32. 
        stride = int(round(dt / ground_truth_dt))
        
        # Downsample the ground truth history to match the test history's timestamps
        matched_gt = ground_truth_hist[::stride]
        
        # Handle any trailing frame rounding mismatches safely
        matched_gt = matched_gt[:test_hist.shape[0]]
        
        # 2. Calculate spatial differences across ALL frames and ALL swimmers
        dx = jnp.abs(test_hist[:, :, 0] - matched_gt[:, :, 0])
        dy = jnp.abs(test_hist[:, :, 1] - matched_gt[:, :, 1])
        
        # Periodic Boundary Conditions (Minimum Image Convention)
        dx = jnp.where(dx > Lx / 2.0, Lx - dx, dx)
        dy = jnp.where(dy > Ly / 2.0, Ly - dy, dy)
        
        # Distances shape: [num_steps, N_swimmers]
        distances = jnp.sqrt(dx**2 + dy**2)
        
        # Mean error across the 2 swimmers at every single point in time
        error_over_time = jnp.mean(distances, axis=1)  # Shape: [num_steps]
        
        # 3. Extract key tracking metrics
        max_peak_error = jnp.max(error_over_time)   # Triggers exactly during the peak jiggle!
        time_avg_error = jnp.mean(error_over_time)  # Overall trajectory deviation
        
        fractional_max_diff = (max_peak_error / 1) * 100
        
        # Print formatted results
        print(f"{dt:<10.5f} | {max_peak_error.item():<18.6e} | {time_avg_error.item():<18.6e} | {fractional_max_diff.item():.6f} %")

    print("\nTest Complete.")