import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jndi
from jax_md import space, partition
import numpy as np

# Enable 64-bit precision globally
jax.config.update("jax_enable_x64", True) 

class AxisymmetricSwimmers2D:
    def __init__(self, positions, angles, dipole_strengths):
        """
        Initialise N axisymmetric dipole microswimmers in a 2D plane.
        
        Parameters
        ----------
        positions : Array, shape (N, 2)
            (x, y) coordinates of each swimmer
        angles : Array, shape (N, )
            Orientation angle (theta) of each swimmer, measured from the x-axis
        dipole_strengths : Array, shape (N, )
            Scalar dipole strength of each swimmer. 
            Positive for pusher, negative for puller.
        """
        self.positions = jnp.asarray(positions, dtype=jnp.float32)
        self.angles = jnp.asarray(angles, dtype=jnp.float32)
        self.dipole_strengths = jnp.asarray(dipole_strengths, dtype=jnp.float32)

    @property
    def count(self):
        """Returns the number of swimmers in the system"""
        return self.positions.shape[0]

    def update_states(self, new_positions, new_angles):
        """Updates kinematic states. Dipole strengths are assumed constant."""
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
    
    # Compute normalized continuous positions on the grid scale
    x_idx = positions[:, 0] / dx
    y_idx = positions[:, 1] / dy
    
    # Lower-left pixel coordinates
    ix = jnp.floor(x_idx).astype(jnp.int64)
    iy = jnp.floor(y_idx).astype(jnp.int64)
    
    # Fractional distances (weights)
    wx = x_idx - ix
    wy = y_idx - iy
    
    # Periodic boundary wrap-around for neighboring indices
    ix0 = ix % Nx
    ix1 = (ix + 1) % Nx
    iy0 = iy % Ny
    iy1 = (iy + 1) % Ny
    
    # Central plane index where swimmers are physically restricted
    iz = Nz // 2
    
    # Construct bilinear weights
    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    
    # Accumulate quantities onto the 4 nearest grid cell neighbors at the midplane slice
    C = quantities.shape[0]
    grid = jnp.zeros((C, Nx, Ny, Nz))
    grid = grid.at[:, ix0, iy0, iz].add(quantities * w00)
    grid = grid.at[:, ix1, iy0, iz].add(quantities * w10)
    grid = grid.at[:, ix0, iy1, iz].add(quantities * w01)
    grid = grid.at[:, ix1, iy1, iz].add(quantities * w11)
    
    return grid

def build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz, mu=1.0, eps=0.5, v0=0.5):
    """
    Precomputes grids and returns a JIT-compiled function to solve the flow field, 
    incorporating wet GEM-4 steric repulsion via an optimized Fourier-space convolution.
    
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

    # Dynamic length scale for GEM-4 potential
    sigma_rep = 2.0 * eps
    
    # The GEM-4 potential is exp(-(r/sigma)^4). At r = 3.0 * sigma, the force 
    # multiplier is exp(-81), which is mathematically zero for float32.
    r_cutoff = 3.0 * sigma_rep 
    
    displacement_fn, _ = space.periodic(jnp.array([Lx, Ly]))
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=jnp.array([Lx, Ly]),
        r_cutoff=r_cutoff,
        format=partition.Dense,
        dr_threshold=eps,  # Buffer padding so the list doesn't rebuild every single step
        fractional_coordinates=False
    )

    # Wavevector grids for Fourier-space solving
    kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(Nz, d=dz)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
    
    K2 = KX**2 + KY**2 + KZ**2
    K2_safe = jnp.where(K2 == 0.0, 1.0, K2)

    # Precompute the analytical continuous Fourier Transform of the Gaussian regularizer
    # F{exp(-r^2 / 2eps^2) / (2pi^1.5 * eps^3)} = exp(-k^2 * eps^2 / 2)
    Phi_g_hat = jnp.exp(-K2 * (eps**2) / 2.0)

    @jax.jit
    def solve_flow(positions, angles, sigmas, neighbor_idx):
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
        sigmas : Array, shape (N, )
            Dipole strength of all swimmers, positive for pusher, negative for puller.
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

        # Vectorized GEM-4 Pairwise Repulsion Calculation
        eps_rep_individual = 25 * jnp.e * (v0 + jnp.abs(sigmas) / (4 * jnp.pi))
        
        # ---------------------------------------------------------------------
        # O(N) GEM-4 Repulsion via Neighbor List
        # ---------------------------------------------------------------------
        # JAX-MD pads empty neighbor slots with the index 'N'. We append a dummy 
        # zero-state to our arrays so out-of-bounds indices route safely to 0.0.
        padded_positions = jnp.vstack([positions, jnp.zeros((1, 2))])
        padded_eps = jnp.append(eps_rep_individual, 0.0)
        
        # Extract only the physically relevant local neighbors (Shape: N x max_neighbors)
        neighbor_positions = padded_positions[neighbor_idx]
        neighbor_eps = padded_eps[neighbor_idx]
        
        dx_pair = positions[:, None, 0] - neighbor_positions[:, :, 0]
        dy_pair = positions[:, None, 1] - neighbor_positions[:, :, 1]
        
        # Minimum Image Convention
        dx_pair -= Lx * jnp.round(dx_pair / Lx)
        dy_pair -= Ly * jnp.round(dy_pair / Ly)
        
        dist_sq = dx_pair**2 + dy_pair**2
        
        # Mask out padding indices and self-interactions
        valid_mask = (neighbor_idx < N) & (dist_sq > 0.0)
        safe_dist_sq = jnp.where(valid_mask, dist_sq, 1.0)
        
        eps_rep_matrix = 0.5 * (eps_rep_individual[:, None] + neighbor_eps)
        
        r_over_sig_4 = safe_dist_sq**2 / sigma_rep**4
        force_div_r = 4.0 * eps_rep_matrix * (safe_dist_sq / sigma_rep**4) * jnp.exp(-r_over_sig_4)
        force_div_r = jnp.where(valid_mask, force_div_r, 0.0)
        
        # Sum local forces safely
        net_rep_fx = jnp.sum(force_div_r * dx_pair, axis=1)
        net_rep_fy = jnp.sum(force_div_r * dy_pair, axis=1)
        
        # ---------------------------------------------------------------------
        # Splatting and FFT Flow Solver
        # ---------------------------------------------------------------------
        # Calculate active orientation vectors and active dipole components
        px, py = jnp.cos(angles), jnp.sin(angles)
        M_xx = sigmas * px * px
        M_xy = sigmas * px * py
        M_yy = sigmas * py * py

        # Pack all particle physical fields into a stacked array (5 unique channels)
        # Channels: 0=Repulsive F_x, 1=Repulsive F_y, 2=Dipole M_xx, 3=Dipole M_xy, 4=Dipole M_yy
        quantities = jnp.stack([net_rep_fx, net_rep_fy, M_xx, M_xy, M_yy], axis=0)

        # Splat all particles onto the 3D grid instantly via vectorized bilinear scattering
        spiky_grid = _splat_to_grid(positions, quantities, Nx, Ny, Nz, dx, dy)

        # Unpack the spiky grid channels and apply 3D Fast Fourier Transforms
        F_rep_hat_x = jnp.fft.fftn(spiky_grid[0])
        F_rep_hat_y = jnp.fft.fftn(spiky_grid[1])
        M_xx_hat    = jnp.fft.fftn(spiky_grid[2])
        M_xy_hat    = jnp.fft.fftn(spiky_grid[3])
        M_yy_hat    = jnp.fft.fftn(spiky_grid[4])

        # Compute the active forces in Fourier space using the Derivative Identity (d/dx -> i*kx)
        F_active_hat_x = -1j * (KX * M_xx_hat + KY * M_xy_hat)
        F_active_hat_y = -1j * (KX * M_xy_hat + KY * M_yy_hat)

        # Combine fields and apply the global Gaussian Regularization kernel
        F_hat_x = (F_rep_hat_x + F_active_hat_x) * Phi_g_hat
        F_hat_y = (F_rep_hat_y + F_active_hat_y) * Phi_g_hat
        F_hat_z = jnp.zeros_like(F_hat_x)

        # ---------------------------------------------------------------------
        # FFT Fluid Solver
        # ---------------------------------------------------------------------
        k_dot_F_hat = KX * F_hat_x + KY * F_hat_y + KZ * F_hat_z
        
        U_hat_x = (F_hat_x - KX * k_dot_F_hat / K2_safe) / (mu * K2_safe)
        U_hat_y = (F_hat_y - KY * k_dot_F_hat / K2_safe) / (mu * K2_safe)
        
        U_hat_x = U_hat_x.at[0, 0, 0].set(0.0)
        U_hat_y = U_hat_y.at[0, 0, 0].set(0.0)
        
        Vort_hat_z = 1j * KX * U_hat_y - 1j * KY * U_hat_x

        # Scale by grid density to eliminate grid-size dependence
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
    
def build_trajectory_generator(Lx, Ly, dt, solve_flow_fn, sigmas, v0=0.5, Dr=0.01):
    """
    Returns a JIT-compiled function that simulates swimmer trajectories over time using 
    active self-propulsion, steric repulsion, rotational diffusion, AND hydrodynamic coupling.

    Parameters
    -----------
    Lx, Ly : Floats
        Dimensions of the simulation domain
    dt : Float
        Time-step
    solve_flow_fn : JIT-compiled function
        Returns linear and angular velocity of a rigid particle due to background flow, 
        and velocity flow field.
    sigmas : Array, shape (N, )
        Dipole strength of all swimmers, positive for pusher, negative for puller.
    v0 : Float
        Self-propulsion velocity of all swimmers (share the same value)
    Dr : Float
        Rotational diffusivity of all swimmers (share the same value)

    Returns
    --------
    wrapper_simulation : JIT-compiled function
        A function which computes the trajectory (positions and angles) of 
        all swimmers over time.
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
                Current (x, y) positions of all swimmers, current orientation of all 
                swimmers, current random number generator key, and current neighbor list.
        
            Returns
            --------
            next_carry : tuple
                The value of carry to be used in the next iteration of scan_step.
            outputs : tuple
                The data from the current timestep to be saved, which is (x, y) 
                positions and orientations of all swimmers.
            """
            pos, current_angles, key, nbrs = carry
            
            # 1. Update Neighbor List
            nbrs = nbrs.update(pos)
            
            # 2. Extract sparse neighbor indices and feed them to the solver
            u_fluid, omega_fluid, _ = solve_flow_fn(pos, current_angles, sigmas, nbrs.idx)

            # 3. Update positions (Active Propulsion + Background Flow)
            px = jnp.cos(current_angles)
            py = jnp.sin(current_angles)
            p = jnp.stack([px, py], axis=-1)
            
            next_pos = pos + (v0 * p + u_fluid) * dt
            next_pos = next_pos % jnp.array([Lx, Ly])

            # 4. Update angles
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=current_angles.shape)
            
            next_angles = current_angles + jnp.sqrt(2.0 * Dr * dt) * noise + omega_fluid * dt
            next_angles = next_angles % (2.0 * jnp.pi)
            
            next_carry = (next_pos, next_angles, key, nbrs)
            outputs = (pos, current_angles)
            return next_carry, outputs

        init_state = (initial_positions, initial_angles, prng_key, nbrs_init)
        _, trajectory = jax.lax.scan(scan_step, init_state, None, length=num_steps)
        
        return trajectory

    return wrapper_simulation