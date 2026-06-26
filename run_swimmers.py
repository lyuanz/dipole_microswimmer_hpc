import h5py
import numpy as np
import jax
import jax.numpy as jnp

# Import simulation builders from the core file
from core_functions import build_quasi2d_stokes_solver, build_trajectory_generator

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":

    # 1. Physical Parameters
    domain_size = 64
    vol_fraction = 0.1
    N = int(vol_fraction * domain_size**3)
    Lx, Ly, Lz = domain_size, domain_size, domain_size
    dt = 0.05
    total_t = 200
    num_steps = int(round(total_t / dt)) 
    v0 = 0.5
    Dr = 0.01
    mu = 1.0
    eps = 0.5
    grid_size = 0.25
    total_nodes = int(round(domain_size / grid_size))
    
    # Must be powers of 2 for optimal FFT efficiency
    Nx, Ny, Nz = total_nodes, total_nodes, total_nodes

    print(f"Initializing simulation for {N} swimmers across {num_steps} steps...")
    
    # 2. PRNG and Initialization
    key = jax.random.PRNGKey(42)  # Seed
    key, pos_key, angle_key = jax.random.split(key, 3)
    
    # Evenly distribute positions or randomize them
    init_positions = jax.random.uniform(pos_key, shape=(N, 2), minval=0.0, maxval=Lx)
    init_angles = jax.random.uniform(angle_key, shape=(N,), minval=0.0, maxval=2.0 * jnp.pi)
    
    # Half pushers (+1.0), half pullers (-1.0)
    dipole_strengths = 5.0 * jnp.ones(N)

    # 3. Instantiate Solvers
    solve_flow_fn = build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz, mu, eps, v0)
    sim_fn = build_trajectory_generator(Lx, Ly, dt, solve_flow_fn, dipole_strengths, v0, Dr)

    # 4. Execute the Simulation (JIT compiles on first step)
    print("Compilation and Execution started...")
    positions_traj, angles_traj = sim_fn(init_positions, init_angles, num_steps, key)
    
    # Force evaluation of JAX arrays before saving to measure true compute time
    positions_traj.block_until_ready()
    print("Simulation complete! Preparing HDF5 write operations...")

    # 5. Export to HDF5 Compressed File
    output_filename = "swimmer_trajectory.h5"
    
    with h5py.File(output_filename, "w") as f:
        # Write metadata attributes for structural reference
        f.attrs["Lx"] = Lx
        f.attrs["Ly"] = Ly
        f.attrs["Lz"] = Lz
        f.attrs["dt"] = dt
        f.attrs["total_t"] = total_t
        f.attrs["N"] = N
        f.attrs["v0"] = v0
        f.attrs["Dr"] = Dr
        f.attrs["mu"] = mu
        f.attrs["eps"] = eps
        
        # Create chunked & compressed datasets for performance
        f.create_dataset("positions", data=np.array(positions_traj), compression="gzip", compression_opts=4)
        f.create_dataset("angles", data=np.array(angles_traj), compression="gzip", compression_opts=4)
        f.create_dataset("dipole_strengths", data=np.array(dipole_strengths))

    print(f"Data successfully saved to {output_filename}")