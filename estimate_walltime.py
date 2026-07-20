import time
import math
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Import simulation builders from your core file
from core_functions_unsteady import build_quasi2d_stokes_solver, build_trajectory_generator

def format_time(seconds):
    """Converts seconds into Slurm's HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

if __name__ == "__main__":

    # 1. Target Parameters
    domain_size = 64
    vol_fraction = 0.5
    N = int(vol_fraction * domain_size**2)
    Lx, Ly, Lz = domain_size, domain_size, domain_size
    dt = 0.1
    total_t = 1000
    target_num_steps = int(round(total_t / dt)) 
    v0 = 0.5
    Dr = 0.01
    mu = 1.0
    eps = 0.5
    
    grid_size = 0.5 
    total_nodes = int(round(domain_size / grid_size))
    Nx, Ny, Nz = total_nodes, total_nodes, total_nodes
    
    # 2. Neighbor List Parameter Setup
    sigma_rep = 2.0 * eps
    r_cutoff = 3.0 * sigma_rep
    tracking_radius = r_cutoff + eps  
    particle_radius = 0.5 * sigma_rep
    max_neighbors = int(0.9 * (tracking_radius / particle_radius)**2)

    # Benchmarking parameters (increased for stable RK4 averaging)
    warmup_steps = 3
    benchmark_steps = 100

    print("Setting up arrays...")
    key = jax.random.PRNGKey(42)
    key, pos_key, angle_key, phase_key = jax.random.split(key, 4)
    
    num_per_side = int(jnp.ceil(jnp.sqrt(N)))
    spacing = Lx / num_per_side
    
    grid_1d = jnp.arange(num_per_side) * spacing + (spacing / 2.0)
    X, Y = jnp.meshgrid(grid_1d, grid_1d)
    grid_positions = jnp.column_stack((X.ravel(), Y.ravel()))
    
    init_positions = grid_positions[:N]
    noise = jax.random.uniform(pos_key, shape=(N, 2), minval=-0.1*spacing, maxval=0.1*spacing)
    init_positions = init_positions + noise
    init_angles = jax.random.uniform(angle_key, shape=(N,), minval=0.0, maxval=2.0 * jnp.pi)
    
    # NEW: Unsteady Dipole Parameters
    sigma_const = 5.0 * jnp.ones(N)
    sigma_amp = 2.5 * jnp.ones(N)       # Amplitude of oscillation
    omega = 2.0 * jnp.pi / 2.0          # Base frequency
    phases = jax.random.uniform(phase_key, shape=(N,), minval=0.0, maxval=2.0 * jnp.pi)

    # 3. Instantiate Solvers
    solve_flow_fn, neighbor_fn = build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz, mu, eps, v0)
    nbrs_init = neighbor_fn.allocate(init_positions, extra_capacity=max_neighbors)
    
    # NEW: Pass unsteady parameters to trajectory generator
    sim_fn = build_trajectory_generator(
        Lx, Ly, dt, solve_flow_fn, 
        sigma_const, sigma_amp, omega, phases, 
        v0, Dr
    )

    # ---------------------------------------------------------
    # PHASE 1: JIT Compilation (Warm-up)
    # ---------------------------------------------------------
    print(f"\n[Phase 1] Forcing JIT Compilation ({warmup_steps} steps)...")
    start_compile = time.time()
    
    warmup_traj = sim_fn(init_positions, init_angles, nbrs_init, warmup_steps, key)
    # Block until both positions and angles are ready to ensure full graph execution
    warmup_traj[0].block_until_ready() 
    warmup_traj[1].block_until_ready()
    
    compile_time = time.time() - start_compile
    print(f"-> Compilation and warmup took: {compile_time:.2f} seconds")

    # ---------------------------------------------------------
    # PHASE 2: Benchmarking Pure Execution Time
    # ---------------------------------------------------------
    print(f"\n[Phase 2] Benchmarking execution speed ({benchmark_steps} steps)...")
    start_bench = time.time()
    
    bench_traj = sim_fn(init_positions, init_angles, nbrs_init, benchmark_steps, key)
    bench_traj[0].block_until_ready() 
    bench_traj[1].block_until_ready()
    
    bench_time = time.time() - start_bench
    time_per_step = bench_time / benchmark_steps
    print(f"-> Pure execution time: {bench_time:.4f} seconds ({time_per_step:.6f} sec/step)")

    # ---------------------------------------------------------
    # PHASE 3: Extrapolation and Buffer
    # ---------------------------------------------------------
    print("\n[Phase 3] Calculating HPC Walltime Request...")
    
    saving_overhead_estimate = 60.0 # Bumped to 60s for potentially larger trajectory files
    projected_raw_time = compile_time + (time_per_step * target_num_steps) + saving_overhead_estimate
    
    safety_factor = 1.25 # 25% buffer for GPU thermal throttling over long runs
    recommended_time = projected_raw_time * safety_factor
    
    print(f"\n{'='*40}")
    print(f"Domain: {Lx}x{Ly}, Grid: {Nx}x{Ny}x{Nz}, Particles: {N}")
    print(f"Projected Raw Time:      {projected_raw_time:.2f} seconds")
    print(f"With 25% Safety Buffer:  {recommended_time:.2f} seconds")
    print(f"{'='*40}")
    print(f"RECOMMENDED WALLTIME:  #SBATCH --time={format_time(recommended_time)}")
    print(f"{'='*40}\n")