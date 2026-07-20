import sys
import h5py
import numpy as np
import jax
import jax.numpy as jnp

# Import simulation builders from the new core file
from core_functions_quad import build_quasi2d_stokes_solver, build_trajectory_generator

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    # Ensure an array index was passed
    if len(sys.argv) < 2:
        print("Error: Please provide a job index (0-99).")
        sys.exit(1)
        
    job_index = int(sys.argv[1])
    
    if job_index < 0 or job_index > 99:
        print("Error: Job index out of bounds. Must be 0-99.")
        sys.exit(1)

    # 1. Base Physical & Simulation Parameters (Constant)
    domain_size = 64
    Lx, Ly, Lz = domain_size, domain_size, domain_size
    dt = 0.1
    total_t = 1000
    num_steps = int(round(total_t / dt)) 
    v0 = 0.5
    Dr = 0.0
    mu = 1.0
    eps = 0.5
    grid_size = 0.5
    total_nodes = int(round(domain_size / grid_size))
    Nx, Ny, Nz = total_nodes, total_nodes, total_nodes
    
    # Stroke frequency (irrelevant since amplitude is 0, but required by solver)
    omega = 2.0 * jnp.pi

    # Neighbor List Geometry Setup
    sigma_rep = 2.0 * eps
    r_cutoff = 3.0 * sigma_rep
    tracking_radius = r_cutoff + eps  
    particle_radius = 0.5 * sigma_rep
    max_neighbors = int(1.5 * np.pi * (tracking_radius / particle_radius)**2 / (2*np.sqrt(3)))

    # Instantiate the base flow solver components
    solve_flow_fn, neighbor_fn = build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz, mu, eps, v0)

    # 2. Define the 10x10 Parameter Grid
    # Constant dipole strength is fixed
    p_const_val = 0.125
    
    # Volumetric fractions: 10 values from 0.1 to 0.5
    vol_fractions = np.linspace(0.1, 0.5, 10)
    
    # Quadrupole constant: 10 values spanning negative (pusher-like) to positive (puller-like)
    # You can adjust this range based on the physics you want to probe.
    quad_vals = np.logspace(np.log10(0.1 * p_const_val), np.log10(10.0 * p_const_val), 10)

    # Map the 1D job index to the 2D parameter grid
    phi_idx = job_index // 10
    quad_idx = job_index % 10

    phi = vol_fractions[phi_idx]
    q_const_val = quad_vals[quad_idx]
    
    # OSCILLATORY COMPONENTS SWITCHED OFF
    p_amp_val = 0.0
    q_amp_val = 0.0

    # 3. Setup specifically for this run
    print("\n" + "-"*60)
    print(f"ARRAY JOB {job_index} | phi = {phi:.3f} | q_const = {q_const_val:.3f}")
    print("-"*60)

    N = int(phi * domain_size**2)
    print(f"Number of swimmers (N): {N}")

    # Initialize a base PRNG key and split based on job index to ensure unique noise per run
    base_key = jax.random.PRNGKey(42 + job_index)
    base_key, pos_key, angle_key, sim_key = jax.random.split(base_key, 4)

    # --- Grid Generation with 40% spacing noise ---
    num_per_side = int(jnp.ceil(jnp.sqrt(N)))
    spacing = Lx / num_per_side
    
    grid_1d = jnp.arange(num_per_side) * spacing + (spacing / 2.0)
    X, Y = jnp.meshgrid(grid_1d, grid_1d)
    grid_positions = jnp.column_stack((X.ravel(), Y.ravel()))
    
    init_positions = grid_positions[:N]
    noise = jax.random.uniform(pos_key, shape=(N, 2), minval=-0.4*spacing, maxval=0.4*spacing)
    init_positions = init_positions + noise
    
    init_angles = jax.random.uniform(angle_key, shape=(N,), minval=0.0, maxval=2.0 * jnp.pi)
    
    # Synchronized strokes: Phase offset is identically zero for all swimmers
    phases = jnp.zeros(N)
    
    # Define the per-particle property arrays for the solver
    sigma_const = jnp.ones(N) * p_const_val
    sigma_amp   = jnp.zeros(N)  # Amplitude is zero
    
    quad_const  = jnp.ones(N) * q_const_val
    quad_amp    = jnp.zeros(N)  # Amplitude is zero

    # --- Allocate Neighbors and Build Trajectory Function ---
    nbrs_init = neighbor_fn.allocate(init_positions, extra_capacity=max_neighbors)
    sim_fn = build_trajectory_generator(
        Lx, Ly, dt, solve_flow_fn, 
        sigma_const, sigma_amp, 
        quad_const, quad_amp, 
        omega, phases, v0=v0, Dr=Dr
    )

    # --- Execute Simulation ---
    print(f"Compiling/Running simulation...")
    trajectories = sim_fn(init_positions, init_angles, nbrs_init, num_steps, sim_key)
    
    positions_traj = trajectories[0]
    angles_traj = trajectories[1]
    
    positions_traj.block_until_ready()
    print("Simulation complete! Saving data...")

    # --- Unique Compressed HDF5 Export ---
    output_filename = f"swimmer_traj_quad_phi_{phi:.3f}_qconst_{q_const_val:.3f}.h5"
    
    with h5py.File(output_filename, "w") as f:
        # Simulation domain & dynamics
        f.attrs["Lx"] = Lx
        f.attrs["Ly"] = Ly
        f.attrs["Lz"] = Lz
        f.attrs["dt"] = dt
        f.attrs["total_t"] = total_t
        f.attrs["v0"] = v0
        f.attrs["Dr"] = Dr
        f.attrs["mu"] = mu
        f.attrs["eps"] = eps
        
        # Swarm properties
        f.attrs["vol_fraction"] = phi
        f.attrs["N"] = N
        f.attrs["omega"] = omega
        
        # Dipole / Quadrupole properties
        f.attrs["dipole_const"] = float(p_const_val)
        f.attrs["dipole_amp"]   = float(p_amp_val)
        f.attrs["quad_const"]   = float(q_const_val)
        f.attrs["quad_amp"]     = float(q_amp_val)
        
        # Time-series datasets
        f.create_dataset("positions", data=np.array(positions_traj), compression="gzip", compression_opts=4)
        f.create_dataset("angles", data=np.array(angles_traj), compression="gzip", compression_opts=4)
        
        # Static arrays
        f.create_dataset("phases", data=np.array(phases), compression="gzip", compression_opts=4)
        
    print(f"Successfully saved: {output_filename}")
