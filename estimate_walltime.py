import time
import math
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Import simulation builders from the core file
from core_functions import build_quasi2d_stokes_solver, build_trajectory_generator

def format_time(seconds):
    """Converts seconds into Slurm's HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

if __name__ == "__main__":

    # 1. Target Parameters
    domain_size = 64
    vol_fraction = 0.6
    N = int(vol_fraction * domain_size**2)
    Lx, Ly, Lz = domain_size, domain_size, domain_size
    dt = 0.05
    total_t = 1000
    target_num_steps = int(round(total_t / dt)) 
    v0 = 0.5
    Dr = 0.01
    mu = 1.0
    eps = 0.5
    grid_size = 0.25
    total_nodes = int(round(domain_size / grid_size))
    Nx, Ny, Nz = total_nodes, total_nodes, total_nodes
    
    # 2. Neighbor List Parameter Setup
    sigma_rep = 2.0 * eps
    r_cutoff = 3.0 * sigma_rep
    tracking_radius = r_cutoff + eps  
    particle_radius = 0.5 * sigma_rep
    # 2D packing physics for maximum neighbors capacity
    max_neighbors = int(0.9 * (tracking_radius / particle_radius)**2)

    # Benchmarking parameters
    warmup_steps = 2
    benchmark_steps = 30

    print("Setting up arrays...")
    key = jax.random.PRNGKey(42)
    key, pos_key, angle_key = jax.random.split(key, 3)
    
    # Calculate how many particles we need per row/column to fit N particles
    num_per_side = int(jnp.ceil(jnp.sqrt(N)))
    
    # Calculate the exact spacing needed to fill the box evenly
    spacing = Lx / num_per_side
    
    # Create the 2D grid coordinates
    grid_1d = jnp.arange(num_per_side) * spacing + (spacing / 2.0)
    X, Y = jnp.meshgrid(grid_1d, grid_1d)
    grid_positions = jnp.column_stack((X.ravel(), Y.ravel()))
    
    # Take exactly N positions (since the grid might have a few extra slots)
    init_positions = grid_positions[:N]
    
    # Add a tiny, safe micro-noise to break perfect grid symmetry
    # (Keeps them from being perfectly locked in unstable equilibriums)
    noise = jax.random.uniform(pos_key, shape=(N, 2), minval=-0.1*spacing, maxval=0.1*spacing)
    init_positions = init_positions + noise
    
    init_angles = jax.random.uniform(angle_key, shape=(N,), minval=0.0, maxval=2.0 * jnp.pi)
    dipole_strengths = 5.0 * jnp.ones(N)

    # 3. Instantiate Solvers - Extract both solver and neighbor list function
    solve_flow_fn, neighbor_fn = build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz, mu, eps, v0)
    
    # Pre-allocate the neighbor list structure
    nbrs_init = neighbor_fn.allocate(init_positions, extra_capacity=max_neighbors)
    
    sim_fn = build_trajectory_generator(Lx, Ly, dt, solve_flow_fn, dipole_strengths, v0, Dr)

    # ---------------------------------------------------------
    # PHASE 1: JIT Compilation (Warm-up)
    # ---------------------------------------------------------
    print(f"\n[Phase 1] Forcing JIT Compilation ({warmup_steps} steps)...")
    start_compile = time.time()
    
    # Pass nbrs_init to the trajectory generator
    warmup_traj = sim_fn(init_positions, init_angles, nbrs_init, warmup_steps, key)
    warmup_traj[0].block_until_ready() 
    
    compile_time = time.time() - start_compile
    print(f"-> Compilation and warmup took: {compile_time:.2f} seconds")

    # ---------------------------------------------------------
    # PHASE 2: Benchmarking Pure Execution Time
    # ---------------------------------------------------------
    print(f"\n[Phase 2] Benchmarking execution speed ({benchmark_steps} steps)...")
    start_bench = time.time()
    
    # Pass nbrs_init to the trajectory generator
    bench_traj = sim_fn(init_positions, init_angles, nbrs_init, benchmark_steps, key)
    bench_traj[0].block_until_ready() 
    
    bench_time = time.time() - start_bench
    time_per_step = bench_time / benchmark_steps
    print(f"-> Pure execution time: {bench_time:.4f} seconds ({time_per_step:.6f} sec/step)")

    # ---------------------------------------------------------
    # PHASE 3: Extrapolation and Buffer
    # ---------------------------------------------------------
    print("\n[Phase 3] Calculating HPC Walltime Request...")
    
    # Base estimate: compilation + (time per step * total steps) + saving overhead estimate
    saving_overhead_estimate = 30.0 # generous 30 seconds for h5py saving
    projected_raw_time = compile_time + (time_per_step * target_num_steps) + saving_overhead_estimate
    
    # Add a 20% safety buffer for node variations and minor I/O delays
    safety_factor = 1.20
    recommended_time = projected_raw_time * safety_factor
    
    print(f"\n{'='*40}")
    print(f"Projected Raw Time:      {projected_raw_time:.2f} seconds")
    print(f"With 20% Safety Buffer:  {recommended_time:.2f} seconds")
    print(f"{'='*40}")
    print(f"RECOMMENDED WALLTIME:  #SBATCH --time={format_time(recommended_time)}")
    print(f"{'='*40}\n")