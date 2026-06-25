import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jndi
import numpy as np

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
        self.positions = jnp.asarray(positions, dtype=jnp.float64)
        self.angles = jnp.asarray(angles, dtype=jnp.float64)
        self.dipole_strengths = jnp.asarray(dipole_strengths, dtype=jnp.float64)

    @property
    def count(self):
        """Returns the number of swimmers in the system"""
        return self.positions.shape[0]

    def update_states(self, new_positions, new_angles):
        """Updates kinematic states. Dipole strengths are assumed constant."""
        self.positions = new_positions
        self.angles = new_angles
        
def build_quasi2d_stokes_solver(Lx, Ly, Lz, Nx, Ny, Nz, mu=1.0, eps=0.5, v0=0.5):
    """
    Precomputes grids and returns a JIT-compiled function to solve the flow field, 
    incorporating wet GEM-4 steric repulsion.
    
    Parameters
    -----------
    Lx, Ly, Lz : Floats
        Dimension of the simulation domain
    Nx, Ny, Nz : Integers
        Number of grid points in each dimension. 
        Should be powers of two for optimal FFT efficiency.
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
    """
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    z_mid = Lz / 2.0
    
    # 1. Dynamic length scale for GEM-4 potential
    sigma_rep = 2.0 * eps
    
    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    z = jnp.linspace(0, Lz, Nz, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(Nz, d=dz)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
    
    K2 = KX**2 + KY**2 + KZ**2
    K2_safe = jnp.where(K2 == 0.0, 1.0, K2)

    def _single_swimmer_force(pos_2d, theta, sigma, rep_force):
        """
        Compute the force density exerted by a single swimmer, combining both 
        the active force dipole and the transient repulsive point force.
    
        Parameters
        -----------
        pos_2d : Array, shape (2, )
            (x, y) coordinates of the swimmer
        theta : Float
            Orientation of the swimmer with respect to the x-axis
        sigma : Float
            Dipole strength, positive for pusher, negative for puller.
        rep_force : Array, shape (2, )
            x and y-components of the force exerted by the swimmer on the fluid 
            due to steric repulsion from all other neighbours
        Returns
        --------
        force_density : Array, shape (3, Nx, Ny, Nz)
            Force density field due to the swimmer
        """
        px, py = jnp.cos(theta), jnp.sin(theta)
        M_xx, M_xy, M_yy = sigma * px * px, sigma * px * py, sigma * py * py
        
        rx, ry, rz = X - pos_2d[0], Y - pos_2d[1], Z - z_mid

        # Minimum Image Convention for periodic boundaries
        rx -= Lx * jnp.round(rx / Lx)
        ry -= Ly * jnp.round(ry / Ly)
        rz -= Lz * jnp.round(rz / Lz)
        
        r2 = rx**2 + ry**2 + rz**2
        
        # Calculate the regularizing Gaussian blob (phi_g)
        prefactor = 1.0 / ((2.0 * jnp.pi)**1.5 * eps**3)
        phi_g = prefactor * jnp.exp(-r2 / (2.0 * eps**2))
        
        # 1. Active Dipole Force Field
        f_active_x = (M_xx * rx + M_xy * ry) * phi_g / (eps**2)
        f_active_y = (M_xy * rx + M_yy * ry) * phi_g / (eps**2)
        
        # 2. Steric Repulsive Force Field
        f_rep_x = rep_force[0] * phi_g
        f_rep_y = rep_force[1] * phi_g
        
        # Combine forces
        f_x = f_active_x + f_rep_x
        f_y = f_active_y + f_rep_y
        f_z = jnp.zeros_like(f_x)

        force_density = jnp.stack([f_x, f_y, f_z], axis=0)
        
        return force_density

    @jax.jit
    def solve_flow(positions, angles, sigmas):
        """
        Compute the velocity flow field due to active pumping and steric repulsion of N microswimmers.
        GEM-4 potential is used to compute steric repulsion.
        Linear interpolation is used to find velocity away from grid points.
    
        Parameters
        -----------
        positions : Array, shape (N, 2)
            (x, y) coordinates of all swimmers
        angles : Array, shape (N, )
            Orientations of all swimmers with respect to the x-axis
        sigmas : Array, shape (N, )
            Dipole strength of all swimmers, positive for pusher, negative for puller.
    
        Returns
        --------
        u_fluid : Array, shape (N, 2)
            The linear velocity experienced by each swimmer due to background flow.
        Omega_fluid : Array, shape (N, )
            The angular velocity experienced by each swimmer due to background flow.
        u_grid_2d : Array, shape (2, Nx, Ny)
            The velocity flow field due to all swimmers at the grid points.
        """
        
        # 2. Dynamic strength scale for GEM-4 potential
        # Calculate individual epsilon requirement based on target velocity logic
        eps_rep_individual = 25 * jnp.e * (v0 + jnp.abs(sigmas) / (4* jnp.pi))
        
        # Create a symmetric N x N matrix for pairwise repulsion strengths
        eps_rep_matrix = 0.5 * (eps_rep_individual[:, None] + eps_rep_individual[None, :])
        
        # Vectorized GEM-4 Pairwise Repulsion Calculation
        dx_pair = positions[:, 0, None] - positions[None, :, 0]
        dy_pair = positions[:, 1, None] - positions[None, :, 1]
        
        # Minimum Image Convention
        dx_pair -= Lx * jnp.round(dx_pair / Lx)
        dy_pair -= Ly * jnp.round(dy_pair / Ly)
        
        dist_sq = dx_pair**2 + dy_pair**2
        safe_dist_sq = jnp.where(dist_sq == 0.0, 1.0, dist_sq)
        
        # GEM-4 Force Magnitude calculation
        r_over_sig_4 = safe_dist_sq**2 / sigma_rep**4
        force_div_r = 4.0 * eps_rep_matrix * (safe_dist_sq / sigma_rep**4) * jnp.exp(-r_over_sig_4)
        
        # Mask self-interactions (diagonal)
        force_div_r = jnp.where(dist_sq == 0.0, 0.0, force_div_r)
        
        # Net repulsive force vector on each swimmer i
        net_rep_fx = jnp.sum(force_div_r * dx_pair, axis=1)
        net_rep_fy = jnp.sum(force_div_r * dy_pair, axis=1)
        net_rep_forces = jnp.stack([net_rep_fx, net_rep_fy], axis=-1)
        # -----------------------------------------------------------

        def add_swimmer_force(accumulated_force, i):
            """
            Calculates force for swimmer i and adds it to the running total.
    
            Parameters
            -----------
            accumulated_force : Array, shape (3, Nx, Ny, Nz)
                Total force on the fluid due to all microswimmers accounted for thus far 
                (index 0 to i-1) at the current iteration.
            i : Integer
                Index of the swimmer to be accounted for in the current iteration.
        
            Returns
            --------
            new_force : Array, shape (3, Nx, Ny, Nz)
                The updated accumulated force after considering swimmer of index i.
            None : This is supposed to be the history of accumulated force. Not needed 
            here. Left `None` for memory management.
            """
            new_force = accumulated_force + _single_swimmer_force(
                positions[i], angles[i], sigmas[i], net_rep_forces[i]
            )
            return new_force, None

        initial_force_grid = jnp.zeros((3, Nx, Ny, Nz))
        
        total_force, _ = jax.lax.scan(
            add_swimmer_force, 
            initial_force_grid, 
            jnp.arange(positions.shape[0])
        )
        
        # FFT Fluid Solver 
        F_hat_x = jnp.fft.fftn(total_force[0])
        F_hat_y = jnp.fft.fftn(total_force[1])
        F_hat_z = jnp.fft.fftn(total_force[2])
        
        k_dot_F_hat = KX * F_hat_x + KY * F_hat_y + KZ * F_hat_z
        
        U_hat_x = (F_hat_x - KX * k_dot_F_hat / K2_safe) / (mu * K2_safe)
        U_hat_y = (F_hat_y - KY * k_dot_F_hat / K2_safe) / (mu * K2_safe)
        
        U_hat_x = U_hat_x.at[0, 0, 0].set(0.0)
        U_hat_y = U_hat_y.at[0, 0, 0].set(0.0)
        
        Vort_hat_z = 1j * KX * U_hat_y - 1j * KY * U_hat_x
        
        u_x_3d = jnp.fft.ifftn(U_hat_x).real
        u_y_3d = jnp.fft.ifftn(U_hat_y).real
        vort_z_3d = jnp.fft.ifftn(Vort_hat_z).real
        
        # Slice at mid-plane
        z_idx = Nz // 2
        u_x_2d = u_x_3d[:, :, z_idx]
        u_y_2d = u_y_3d[:, :, z_idx]
        vort_z_2d = vort_z_3d[:, :, z_idx]

        # Interpolate fluid velocity back to particle positions
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

    return solve_flow
    
def build_trajectory_generator(Lx, Ly, dt, solve_flow_fn, sigmas, v0=0.5, Dr=0.01):
    """
    Returns a JIT-compiled function that simulates swimmer trajectories over time
    using active self-propulsion, rotational diffusion, AND hydrodynamic coupling.

    Parameters
    -----------
    Lx, Ly : Floats
        Dimensions of the simulation domain
    dt : Float
        Time-step
    solve_flow_fn : JIT-compiled function
        Returns linear and angular velocity of a rigid particle due to background flow
    sigmas : Array, shape (N, )
        Dipole strength of all swimmers, positive for pusher, negative for puller.
    v0 : Float
        Self-propulsion velocity of all swimmers (share the same value)
    Dr : Float
        Rotational diffusivity of all swimmers (share the same value)

    Returns
    --------
    wrapper_simulation : JIT-compiled function
        A function which computes the trajectory (position and angle) of 
        all swimmers over time.
    """
    
    @jax.jit(static_argnames=['num_steps'])
    def wrapper_simulation(initial_positions, initial_angles, num_steps, prng_key):
        """
        The function which computes the trajectory (positions and orientations) of 
        all swimmers over time.
    
        Parameters
        -----------
        initial_positions : Array, shape (N, 2)
            Initial (x, y) coordinates of all swimmers
        initial_angles : Array, shape (N, )
            Initial orientation, as measured from x-axis, of all swimmers
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
                swimmers, and current random number generator key.
        
            Returns
            --------
            next_carry : tuple
                The value of carry to be used in the next iteration of scan_step.
            outputs : tuple
                The data from the current timestep to be saved, which is (x, y) 
                positions and orientation of all swimmers.
            """
            pos, current_angles, key = carry
            
            # The fluid flow now inherently includes the repulsive steric forces
            u_fluid, omega_fluid, _ = solve_flow_fn(pos, current_angles, sigmas)
            
            # 1. Update positions (Active Propulsion + Background Flow)
            px = jnp.cos(current_angles)
            py = jnp.sin(current_angles)
            p = jnp.stack([px, py], axis=-1)
            
            next_pos = pos + (v0 * p + u_fluid) * dt
            next_pos = next_pos % jnp.array([Lx, Ly])
            
            # 2. Update angles
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=current_angles.shape)
            
            next_angles = current_angles + jnp.sqrt(2.0 * Dr * dt) * noise + omega_fluid * dt
            next_angles = next_angles % (2.0 * jnp.pi)
            
            next_carry = (next_pos, next_angles, key)
            outputs = (pos, current_angles)
            return next_carry, outputs

        init_state = (initial_positions, initial_angles, prng_key)
        _, trajectory = jax.lax.scan(scan_step, init_state, None, length=num_steps)
        
        return trajectory

    return wrapper_simulation