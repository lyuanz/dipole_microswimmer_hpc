import sys
import h5py
import numpy as np
import jax
import jax.numpy as jnp

# Import simulation builders from the core file
from core_functions import build_quasi2d_stokes_solver, build_trajectory_generator

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
    dt = 0.05
    total_t = 1000
    num_steps = int(round(total_t / dt)) 
    v0 = 0.5
    Dr = 1
    mu = 1.0
    eps = 0.5
    grid_size = 0.25
    total_nodes = int(round(domain_size / grid_size))
    Nx, Ny, Nz = total_nodes, total_nodes, total_nodes

    # Neighbor List Geometry Setup
    sigma_rep = 2.0 * eps
    r_cutoff = 3.0 * sigma_rep
    tracking_radius = r_cutoff + eps  
    particle_radius = 0.5 * sigma_rep
    max_neighbors = int(1.5 * np.pi * (tracking_radius / particle_radius)**2 / (2*np.sqrt(3)))

    # Instantiate the base flow solver components
    solve_flow_fn, neighbor_fn = build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz, mu, eps, v0)

    # 2. Define the 10x10 Parameter Grid
    vol_fractions = np.linspace(0.1, 0.5, 10)
    dipole_vals = np.logspace(np.log10(0.0125), np.log10(1.25), 10)

    # Map the 1D job index to the 2D parameter grid
    phi_idx = job_index // 10
    dipole_idx = job_index % 10

    phi = vol_fractions[phi_idx]
    p_val = dipole_vals[dipole_idx]

    # 3. Setup specifically for this run
    print("\n" + "-"*60)
    print(f"ARRAY JOB {job_index} | phi = {phi:.3f} | dipole_strength = {p_val:.3f}")
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
    dipole_strengths = jnp.ones(N) * p_val

    # --- Allocate Neighbors and Build Trajectory Function ---
    nbrs_init = neighbor_fn.allocate(init_positions, extra_capacity=max_neighbors)
    sim_fn = build_trajectory_generator(Lx, Ly, dt, solve_flow_fn, dipole_strengths, v0, Dr)

    # --- Execute Simulation ---
    print(f"Compiling/Running simulation...")
    trajectories = sim_fn(init_positions, init_angles, nbrs_init, num_steps, sim_key)
    
    positions_traj = trajectories[0]
    angles_traj = trajectories[1]
    
    positions_traj.block_until_ready()
    print("Simulation complete! Saving data...")

    # --- Unique Compressed HDF5 Export ---
    output_filename = f"swimmer_traj_phi_{phi:.3f}_p_{p_val:.3f}.h5"
    
    with h5py.File(output_filename, "w") as f:
        f.attrs["Lx"] = Lx
        f.attrs["Ly"] = Ly
        f.attrs["Lz"] = Lz
        f.attrs["dt"] = dt
        f.attrs["total_t"] = total_t
        f.attrs["v0"] = v0
        f.attrs["Dr"] = Dr
        f.attrs["mu"] = mu
        f.attrs["eps"] = eps
        f.attrs["vol_fraction"] = phi
        f.attrs["dipole_strength"] = float(p_val)
        f.attrs["job_index"] = job_index
        
        f.create_dataset("positions", data=np.array(positions_traj), compression="gzip", compression_opts=4)
        f.create_dataset("angles", data=np.array(angles_traj), compression="gzip", compression_opts=4)
        
    print(f"Successfully saved: {output_filename}")
