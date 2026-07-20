import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jndi
from jax_md import space, partition
import numpy as np

jax.config.update("jax_enable_x64", True)

class AxisymmetricSwimmers2D:
    def __init__(self, positions, angles, dipole_const, dipole_amp, quad_const, quad_amp, phases):
        """
        Initialise N axisymmetric dipole-quadrupole microswimmers in a 2D plane.
        
        Parameters
        ----------
        positions : Array, shape (N, 2)
            (x, y) coordinates of each swimmer
        angles : Array, shape (N, )
            Orientation angle (theta) of each swimmer, measured from the x-axis
        dipole_const : Array, shape (N, )
            Constant scalar dipole strength of each swimmer.
        dipole_amp : Array, shape (N, )
            Amplitude of the sinusoidal dipole component.
        quad_const : Array, shape (N, )
            Constant scalar quadrupole strength (typically ~ dipole * length).
        quad_amp : Array, shape (N, )
            Amplitude of the sinusoidal quadrupole component.
        phases : Array, shape (N, )
            Initial phase offset for the sinusoidal stroke cycle of each swimmer.
        """
        self.positions = jnp.asarray(positions, dtype=jnp.float64)
        self.angles = jnp.asarray(angles, dtype=jnp.float64)
        self.dipole_const = jnp.asarray(dipole_const, dtype=jnp.float64)
        self.dipole_amp = jnp.asarray(dipole_amp, dtype=jnp.float64)
        self.quad_const = jnp.asarray(quad_const, dtype=jnp.float64)
        self.quad_amp = jnp.asarray(quad_amp, dtype=jnp.float64)
        self.phases = jnp.asarray(phases, dtype=jnp.float64)

    @property
    def count(self):
        """Returns the number of swimmers in the system"""
        return self.positions.shape[0]

    def update_states(self, new_positions, new_angles):
        """Updates kinematic states."""
        self.positions = new_positions
        self.angles = new_angles
        
def _splat_to_grid(positions, quantities, Nx, Ny, Nz, dx, dy):
    """
    Distributes particle quantities smoothly onto the 2D grid nodes using the 
    inverse of bilinear interpolation at the central z-plane slice.

    Parameters
    -----------
    positions : Array, shape (N, 2)
        Continuous sub-grid coordinates of the swimmers.
    quantities : Array, shape (C, N)
        Physical quantities (forces, dipoles) to map to the grid across C 
        channels, each channel representing one quantity.
    Nx, Ny, Nz : Integers
        Grid counts
    dx, dy : Floats
        Grid spacings
        
    Returns
    --------
    grid : Array, shape (C, Nx, Ny, Nz)
        The spiky grid containing the interpolated point-source fields.
    """
    
    x_idx = positions[:, 0] / dx
    y_idx = positions[:, 1] / dy
    
    ix = jnp.floor(x_idx).astype(jnp.int64)
    iy = jnp.floor(y_idx).astype(jnp.int64)
    
    wx = x_idx - ix
    wy = y_idx - iy
    
    ix0 = ix % Nx
    ix1 = (ix + 1) % Nx
    iy0 = iy % Ny
    iy1 = (iy + 1) % Ny
    
    iz = Nz // 2
    
    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    
    # Automatically handles any number of channels C
    C = quantities.shape[0]
    grid = jnp.zeros((C, Nx, Ny, Nz))
    grid = grid.at[:, ix0, iy0, iz].add(quantities * w00)
    grid = grid.at[:, ix1, iy0, iz].add(quantities * w10)
    grid = grid.at[:, ix0, iy1, iz].add(quantities * w01)
    grid = grid.at[:, ix1, iy1, iz].add(quantities * w11)
    
    return grid

def build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz, mu=1.0, eps=0.5, v0=0.5):
    """
    Precomputes grids and returns a JIT-compiled function to solve the flow field.

    Parameters
    -----------
    Lx, Ly, Lz : Floats
        Dimension of the simulation domain
    Nx, Ny, Nz : Integers
        Grid count in each dimension. 
    mu : Float
        Dynamic viscosity
    eps : Float
        Size of Gaussian blob
    v0 : Float
        Self-propulsion velocity of the swimmers (used to scale repulsion)

    Returns
    --------
    solve_flow : JIT-compiled function
        A function to calculate the velocity field of the fluid
    neighbor_fn : jax_md.partition.NeighborFn
        A JAX-MD state manager containing `.allocate()` and `.update()` methods. 
        It maintains a dynamic Verlet neighbor list (spatial partitioning grid) 
        to track local particle interactions, reducing the steric repulsion 
        compute complexity from O(N^2) to O(N).
    """
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

    sigma_rep = 2.0 * eps
    r_cutoff = 3.0 * sigma_rep 
    
    displacement_fn, _ = space.periodic(jnp.array([Lx, Ly]))
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=jnp.array([Lx, Ly]),
        r_cutoff=r_cutoff,
        format=partition.Dense,
        dr_threshold=eps,
        fractional_coordinates=False
    )

    kx = (2 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx))[:, None, None]
    ky = (2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy))[None, :, None]
    kz = (2 * jnp.pi * jnp.fft.fftfreq(Nz, d=dz))[None, None, :]
    
    @jax.jit(inline=False)
    def solve_flow(positions, angles, sigmas_active_dipole, sigmas_active_quad, sigmas_rep, neighbor_idx):
        """
        Compute the velocity flow field due to active pumping and steric repulsion of N 
        microswimmers. Uses `neighbor_idx` to drastically restrict pairwise operations to 
        local vicinity.
    
        Parameters
        -----------
        positions : Array, shape (N, 2)
            (x, y) coordinates of all swimmers
        angles : Array, shape (N, )
            Orientations of all swimmers with respect to the x-axis
        sigmas_active : Array, shape (N, )
            Instantaneous time-varying dipole for the hydrodynamic flow.
        sigmas_active_quad : Array, shape (N, )
            Instantaneous time-varying force quadrupole for the hydrodynamic flow.
        sigmas_rep : Array, shape (N, )
            Constant dipole used solely to set the steric repulsion radius.
        neighbor_idx : Integer Array, shape (N, max_neighbors)
            A dense matrix containing the indices of nearby interacting swimmers for 
            each particle, extracted from the JAX-MD neighbor list (`nbrs.idx`). 
            Empty neighbor slots are padded with the integer `N` to maintain a static 
            array shape for the JIT compiler.
    
        Returns
        --------
        u_fluid : Array, shape (N, 2)
            The linear velocity experienced by each swimmer due to background flow.
        omega_fluid : Array, shape (N, )
            The angular velocity experienced by each swimmer due to background flow.
        u_grid_2d : Array, shape (2, Nx, Ny)
            The velocity flow field due to all swimmers at the grid points.
        """
        N = positions.shape[0]

        # ---------------------------------------------------------------------
        # O(N) GEM-4 Repulsion via Neighbor List
        # ---------------------------------------------------------------------
        eps_rep_individual = 25 * jnp.e * (v0 + jnp.abs(sigmas_rep) / (4 * jnp.pi))

        # JAX-MD pads empty neighbor slots with the index 'N'. We append a dummy 
        # zero-state to our arrays so out-of-bounds indices route safely to 0.0.
        padded_positions = jnp.vstack([positions, jnp.zeros((1, 2))])
        padded_eps = jnp.append(eps_rep_individual, 0.0)
        
        neighbor_positions = padded_positions[neighbor_idx]
        neighbor_eps = padded_eps[neighbor_idx]
        
        dx_pair = positions[:, None, 0] - neighbor_positions[:, :, 0]
        dy_pair = positions[:, None, 1] - neighbor_positions[:, :, 1]
        
        dx_pair -= Lx * jnp.round(dx_pair / Lx)
        dy_pair -= Ly * jnp.round(dy_pair / Ly)
        
        dist_sq = dx_pair**2 + dy_pair**2
        
        valid_mask = (neighbor_idx < N) & (dist_sq > 0.0)
        safe_dist_sq = jnp.where(valid_mask, dist_sq, 1.0)
        
        eps_rep_matrix = 0.5 * (eps_rep_individual[:, None] + neighbor_eps)
        
        r_over_sig_4 = safe_dist_sq**2 / sigma_rep**4
        force_div_r = 4.0 * eps_rep_matrix * (safe_dist_sq / sigma_rep**4) * jnp.exp(-r_over_sig_4)
        force_div_r = jnp.where(valid_mask, force_div_r, 0.0)
        
        net_rep_fx = jnp.sum(force_div_r * dx_pair, axis=1)
        net_rep_fy = jnp.sum(force_div_r * dy_pair, axis=1)
        
        # ---------------------------------------------------------------------
        # Splatting and FFT Flow Solver (Dipole + Quadrupole)
        # ---------------------------------------------------------------------
        px, py = jnp.cos(angles), jnp.sin(angles)
        
        # 1. Force Dipole Tensor Components
        M_xx = sigmas_active_dipole * px * px
        M_xy = sigmas_active_dipole * px * py
        M_yy = sigmas_active_dipole * py * py

        # 2. Force Quadrupole Tensor Components
        Q_xxx = sigmas_active_quad * px * px * px
        Q_xxy = sigmas_active_quad * px * px * py
        Q_xyy = sigmas_active_quad * px * py * py
        Q_yyy = sigmas_active_quad * py * py * py

        # Pack all 9 physical channels for vectorized splatting
        quantities = jnp.stack([
            net_rep_fx, net_rep_fy, 
            M_xx, M_xy, M_yy, 
            Q_xxx, Q_xxy, Q_xyy, Q_yyy
        ], axis=0)

        spiky_grid = _splat_to_grid(positions, quantities, Nx, Ny, Nz, dx, dy)

        # Unpack and transform
        F_rep_hat_x = jnp.fft.fftn(spiky_grid[0])
        F_rep_hat_y = jnp.fft.fftn(spiky_grid[1])
        
        M_xx_hat    = jnp.fft.fftn(spiky_grid[2])
        M_xy_hat    = jnp.fft.fftn(spiky_grid[3])
        M_yy_hat    = jnp.fft.fftn(spiky_grid[4])
        
        Q_xxx_hat   = jnp.fft.fftn(spiky_grid[5])
        Q_xxy_hat   = jnp.fft.fftn(spiky_grid[6])
        Q_xyy_hat   = jnp.fft.fftn(spiky_grid[7])
        Q_yyy_hat   = jnp.fft.fftn(spiky_grid[8])

        K2 = kx**2 + ky**2 + kz**2
        K2_safe = jnp.where(K2 == 0.0, 1.0, K2)
        Phi_g_hat = jnp.exp(-K2 * (eps**2) / 2.0)

        # Dipole active forces (Derivative Identity: d/dx -> i*kx)
        F_dipole_hat_x = -1j * (kx * M_xx_hat + ky * M_xy_hat)
        F_dipole_hat_y = -1j * (kx * M_xy_hat + ky * M_yy_hat)

        # Quadrupole active forces (Double Derivative Identity: d/dx d/dy -> -kx*ky)
        F_quad_hat_x = -(kx**2 * Q_xxx_hat + 2 * kx * ky * Q_xxy_hat + ky**2 * Q_xyy_hat)
        F_quad_hat_y = -(kx**2 * Q_xxy_hat + 2 * kx * ky * Q_xyy_hat + ky**2 * Q_yyy_hat)

        F_active_hat_x = F_dipole_hat_x + F_quad_hat_x
        F_active_hat_y = F_dipole_hat_y + F_quad_hat_y

        F_hat_x = (F_rep_hat_x + F_active_hat_x) * Phi_g_hat
        F_hat_y = (F_rep_hat_y + F_active_hat_y) * Phi_g_hat
        F_hat_z = jnp.zeros_like(F_hat_x)

        # ---------------------------------------------------------------------
        # FFT Fluid Solver
        # ---------------------------------------------------------------------
        k_dot_F_hat = kx * F_hat_x + ky * F_hat_y + kz * F_hat_z
        
        U_hat_x = (F_hat_x - kx * k_dot_F_hat / K2_safe) / (mu * K2_safe)
        U_hat_y = (F_hat_y - ky * k_dot_F_hat / K2_safe) / (mu * K2_safe)
        
        U_hat_x = U_hat_x.at[0, 0, 0].set(0.0)
        U_hat_y = U_hat_y.at[0, 0, 0].set(0.0)
        
        Vort_hat_z = 1j * kx * U_hat_y - 1j * ky * U_hat_x

        volume_factor = (Nx * Ny * Nz) / (Lx * Ly * Lz)
        
        u_x_3d = jnp.fft.ifftn(U_hat_x).real * volume_factor
        u_y_3d = jnp.fft.ifftn(U_hat_y).real * volume_factor
        vort_z_3d = jnp.fft.ifftn(Vort_hat_z).real * volume_factor

        z_idx = Nz // 2
        u_x_2d = u_x_3d[:, :, z_idx]
        u_y_2d = u_y_3d[:, :, z_idx]
        vort_z_2d = vort_z_3d[:, :, z_idx]

        x_idx = positions[:, 0] / dx
        y_idx = positions[:, 1] / dy
        coords = jnp.stack([x_idx, y_idx], axis=0)
        
        u_fluid_x = jndi.map_coordinates(u_x_2d, coords, order=1, mode='wrap')
        u_fluid_y = jndi.map_coordinates(u_y_2d, coords, order=1, mode='wrap')
        vort_fluid_z = jndi.map_coordinates(vort_z_2d, coords, order=1, mode='wrap')
        
        u_fluid = jnp.stack([u_fluid_x, u_fluid_y], axis=-1)
        omega_fluid = 0.5 * vort_fluid_z
        u_grid_2d = jnp.stack([u_x_2d, u_y_2d], axis=0)

        return u_fluid, omega_fluid, u_grid_2d

    return solve_flow, neighbor_fn
    
def build_trajectory_generator(Lx, Ly, dt, solve_flow_fn, sigma_const, sigma_amp, quad_const, quad_amp, omega, phases, v0=0.5, Dr=0.01):
    """
    Returns a JIT-compiled function that simulates swimmer trajectories over time.

    Parameters
    -----------
    Lx, Ly : Floats
        Dimensions of the simulation domain
    dt : Float
        Time-step
    solve_flow_fn : JIT-compiled function
        Returns linear and angular velocity of a rigid particle due to background flow.
    sigma_const : Array, shape (N, )
        Baseline dipole strength of all swimmers.
    sigma_amp : Array, shape (N, )
        Amplitude of the sinusoidal dipole variation.
    quad_const : Array, shape (N, )
        Baseline quadrupole strength of all swimmers.
    quad_amp : Array, shape (N, )
        Amplitude of the sinusoidal quadrupole variation.
    omega : Float or Array, shape (N, )
        Angular frequency of the stroke cycle.
    phases : Array, shape (N, )
        Initial phase offset for each swimmer's stroke cycle.
    v0 : Float
        Self-propulsion velocity of all swimmers (share the same value)
    Dr : Float
        Rotational diffusivity of all swimmers (share the same value)

    Returns
    --------
    wrapper_simulation : JIT-compiled function
    """
    
    @jax.jit(static_argnames=['num_steps'])
    def wrapper_simulation(initial_positions, initial_angles, nbrs_init, num_steps, prng_key):
        """
        The function which computes the trajectory (positions and orientations) of 
        all swimmers over time.
    
        Parameters
        -----------
        initial_positions : Array, shape (N, 2)
            Initial (x, y) coordinates of all swimmers
        initial_angles : Array, shape (N, )
            Initial orientation, as measured from x-axis, of all swimmers
        nbrs_init : jax_md.partition.NeighborList
            The initial state of the spatial neighbor list, generated prior to the 
            simulation loop via `neighbor_fn.allocate(initial_positions)`. This 
            state tracks local interaction pairings and is dynamically updated inside.
        num_steps : Integer
            Number of timesteps of the simulation
        prng_key: jax.Array
            Starting pseudo-random number generator key
    
        Returns
        --------
        trajectory : tuple of 2 elements
            1st element is the history of positions, a JAX array of shape 
            (num_steps, N, 2).
            2nd element is the history of orientations, a JAX array of shape 
            (num_steps, N).
        """
        def scan_step(carry, _):
            """
            Helper function to be used as an argument for jax.lax.scan. This function 
            will be iterated to update the positions and angles of the swimmers.
        
            Parameters
            -----------
            carry : tuple
                Current (x, y) positions of all swimmers, current orientation of all swimmers, 
                current random number generator key, current neighbor list, and current time.
        
            Returns
            --------
            next_carry : tuple
                The value of carry to be used in the next iteration of scan_step.
            outputs : tuple
                The data from the current timestep to be saved, which is (x, y) 
                positions and orientations of all swimmers.
            """
            pos, current_angles, key, nbrs, t = carry
            
            nbrs = nbrs.update(pos)
            idx = nbrs.idx
            
            def rk4_step(positions, angles, dt, t_start, sigma_c, sigma_a, quad_c, quad_a, om, ph, neighbor_idx):
                a_weights = jnp.array([0.0, 0.5 * dt, 0.5 * dt, 1.0 * dt])
                b_weights = jnp.array([dt / 6.0, dt / 3.0, dt / 3.0, dt / 6.0])
                
                def rk4_stage(carry, stage_params):
                    pos_orig, ang_orig, k_prev_pos, k_prev_ang, acc_pos, acc_ang = carry
                    a, b = stage_params
                    
                    pos_eval = pos_orig + a * k_prev_pos
                    ang_eval = ang_orig + a * k_prev_ang
                    
                    t_eval = t_start + a
                    
                    # Compute instantaneous dipole and quadrupole strengths
                    sigmas_dipole_eval = sigma_c + sigma_a * jnp.sin(om * t_eval + ph)
                    sigmas_quad_eval = quad_c + quad_a * jnp.sin(om * t_eval + ph)
                    
                    # Evaluate the background fluid flow field
                    u_fluid, omega_fluid, _ = solve_flow_fn(
                        pos_eval, ang_eval, sigmas_dipole_eval, sigmas_quad_eval, sigma_c, neighbor_idx
                    )
                    
                    px_eval = jnp.cos(ang_eval)
                    py_eval = jnp.sin(ang_eval)
                    u_self = v0 * jnp.stack([px_eval, py_eval], axis=-1)
                    
                    total_u = u_fluid + u_self
                    
                    acc_pos_new = acc_pos + b * total_u
                    acc_ang_new = acc_ang + b * omega_fluid
                    
                    return (pos_orig, ang_orig, total_u, omega_fluid, acc_pos_new, acc_ang_new), None
            
                init_carry = (
                    positions, angles, 
                    jnp.zeros_like(positions), jnp.zeros_like(angles), 
                    jnp.zeros_like(positions), jnp.zeros_like(angles)
                )
                
                final_carry, _ = jax.lax.scan(rk4_stage, init_carry, (a_weights, b_weights))
                _, _, _, _, delta_pos, delta_ang = final_carry
                
                return positions + delta_pos, angles + delta_ang
            
            next_pos, next_angles_det = rk4_step(
                pos, current_angles, dt, t, 
                sigma_const, sigma_amp, quad_const, quad_amp, omega, phases, idx
            )
            
            next_pos = next_pos % jnp.array([Lx, Ly])
            
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=current_angles.shape)
            stochastic_ang = jnp.sqrt(2.0 * Dr * dt) * noise
            
            next_angles = next_angles_det + stochastic_ang
            next_angles = next_angles % (2.0 * jnp.pi)
            
            next_t = t + dt
            next_carry = (next_pos, next_angles, key, nbrs, next_t)
            outputs = (pos, current_angles)
            
            return next_carry, outputs

        init_state = (initial_positions, initial_angles, prng_key, nbrs_init, 0.0)
        _, trajectory = jax.lax.scan(scan_step, init_state, None, length=num_steps)
        
        return trajectory

    return wrapper_simulation