import h5py
import numpy as np
import freud
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def quadratic_refine(bin_centers, y_values, idx):
    """
    Sub-bin refinement of a discrete extremum at idx via a parabola fit
    through (idx-1, idx, idx+1). Returns (r_refined, y_refined).
    """
    # Quadratic fitting doesn't work if point of interest is at the boundary.
    if idx <= 0 or idx >= len(y_values) - 1:
        return bin_centers[idx], y_values[idx]

    y0, y1, y2 = y_values[idx - 1], y_values[idx], y_values[idx + 1]
    
    # If the second derivative is 0, the points are collinear.
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return bin_centers[idx], y_values[idx]

    dr = bin_centers[idx + 1] - bin_centers[idx]
    delta = 0.5 * (y0 - y2) / denom
    delta = np.clip(delta, -0.5, 0.5)  # guard against a degenerate/noisy fit

    r_refined = bin_centers[idx] + delta * dr
    y_refined = y1 - 0.25 * (y0 - y2) * delta
    
    return r_refined, y_refined


def extract_structural_metrics(filename, r_max=5.0, n_bins=400, num_blocks=10):
    """
    Extracts structural metrics from an active matter HDF5 trajectory.
    """
    # 1. Open file and extract attributes/positions
    with h5py.File(filename, 'r') as f:
        positions = f['positions'][:]
        Lx, Ly = f.attrs['Lx'], f.attrs['Ly']
        phi = f.attrs["vol_fraction"]
        p_val = f.attrs["dipole_strength"]

        total_frames = positions.shape[0]
        N = positions.shape[1]

        # 2. Setup temporal blocking for steady-state
        frames_to_analyze = np.arange(0, total_frames)
        frame_blocks = np.array_split(frames_to_analyze, num_blocks)

        block_gr_list = []
        box = freud.box.Box(Lx=Lx, Ly=Ly, is2D=True)
        frame_3d_cache = np.zeros((N, 3), dtype=positions.dtype)

        # 3. Compute block-averaged g(r)
        for block_frames in frame_blocks:
            rdf = freud.density.RDF(bins=n_bins, r_max=r_max)
            for frame_idx in block_frames:
                frame_3d_cache[:, :2] = positions[frame_idx]
                rdf.compute(system=(box, frame_3d_cache), reset=False)
            block_gr_list.append(rdf.rdf)

        bin_centers = rdf.bin_centers

    # Calculate final steady-state g(r) mean
    block_gr_array = np.array(block_gr_list)
    final_gr_mean = np.mean(block_gr_array, axis=0)
    final_gr_std = np.std(block_gr_array, axis=0)

    # 4. Extract Peak Metrics (discrete detection, then quadratic refinement)
    peak_idx = np.argmax(final_gr_mean)
    r_peak, peak_gr_val = quadratic_refine(bin_centers, final_gr_mean, peak_idx)

    # 5. Extract Cluster Radius / First Minimum (discrete detection, then refinement)
    inverted_gr = -final_gr_mean[peak_idx:]
    min_prominence = 0.05 * (final_gr_mean[peak_idx] - 1.0)
    minima_indices, _ = find_peaks(inverted_gr, prominence=min_prominence)

    # 6. Calculate Coordination Number & Fraction Safely
    area = Lx * Ly
    rho = N / area

    if len(minima_indices) > 0:
        first_min_idx = peak_idx + minima_indices[0]
        cluster_radius, g_at_min = quadratic_refine(bin_centers, final_gr_mean, first_min_idx)

        # Integrate up to the refined minimum, not the nearest bin: keep every
        # discrete point up to first_min_idx-1 untouched, then swap the final
        # point for the sub-bin-refined (r, g) pair so the upper integration
        # limit matches cluster_radius exactly.
        r_cluster = bin_centers[:first_min_idx + 1].copy()
        g_cluster = final_gr_mean[:first_min_idx + 1].copy()
        r_cluster[-1] = cluster_radius
        g_cluster[-1] = g_at_min

        integrand = g_cluster * 2 * np.pi * r_cluster * rho
        coordination_number = simpson(y=integrand, x=r_cluster)
        fraction_cn = coordination_number / (N - 1)
    else:
        first_min_idx = 0
        cluster_radius = 0.0
        coordination_number = 0.0
        fraction_cn = 0.0

    # 7. Return compiled results
    return {
        "areal_fraction": phi,
        "dipole_strength": p_val,
        "gr_steady_state": final_gr_mean,
        "gr_std": final_gr_std,
        "peak_gr_value": peak_gr_val,
        "peak_radius": r_peak,
        "cluster_radius": cluster_radius,
        "coordination_number": coordination_number,
        "fraction_coordination_number": fraction_cn,
        "bin_centers": bin_centers,
    }

def extract_orientational_correlation(filename, r_max=5.0, n_bins=400, num_blocks=10):
    """
    Extracts the distance-dependent orientational correlation function, <cos(theta_ij)>(r),
    from an active matter HDF5 trajectory, utilizing temporal block averaging.
    
    Optimized version using freud.density.CorrelationFunction to bypass Python overhead.
    """
    # 1. Open file and extract data
    with h5py.File(filename, 'r') as f:
        positions = f['positions'][:]
        angles = f['angles'][:]
        Lx, Ly = f.attrs['Lx'], f.attrs['Ly']
        phi = f.attrs["vol_fraction"]
        p_val = f.attrs["dipole_strength"]
        dt = f.attrs["dt"]

        total_frames = positions.shape[0]
        N = positions.shape[1]

    # 2. Setup geometries and temporal blocking
    frames_to_analyze = np.arange(0, total_frames)
    frame_blocks = np.array_split(frames_to_analyze, num_blocks)

    box = freud.box.Box(Lx=Lx, Ly=Ly, is2D=True)
    frame_3d_cache = np.zeros((N, 3), dtype=np.float32)
    
    # Initialize Freud's built-in C++ Correlation Function
    cf = freud.density.CorrelationFunction(bins=n_bins, r_max=r_max)
    
    block_Cr_list = []

    # 3. Compute block-averaged C(r)
    for block_frames in frame_blocks:
        is_first_frame = True
        
        for frame_idx in block_frames:
            pos = positions[frame_idx]
            angs = angles[frame_idx]
            
            # Freud requires 3D coordinates
            frame_3d_cache[:, :2] = pos
            
            # Map angles to complex numbers on the unit circle (Euler's formula)
            complex_orientations = np.exp(1j * angs)
            
            # Freud computes < v_i* v_j > entirely in C++, bypassing toNeighborList
            # reset=is_first_frame clears the C++ histogram at the start of a block, 
            # then accumulates the data for the rest of the block's frames.
            cf.compute(
                system=(box, frame_3d_cache), 
                values=complex_orientations, 
                reset=is_first_frame
            )
            is_first_frame = False
            
        # Get real correlation and the actual counts for this block
        block_Cr = np.real(cf.correlation).copy()
        counts = cf.bin_counts
        
        # STATISTICAL FILTER: Mask bins that don't have enough pair counts in this block
        min_counts = 0.01 * np.max(counts)
        block_Cr[counts < min_counts] = np.nan
        block_Cr = np.nan_to_num(block_Cr, nan=0.0)
        block_Cr_list.append(block_Cr)

    # 4. Final ensemble statistics across blocks
    block_Cr_array = np.array(block_Cr_list)
    final_Cr_mean = np.mean(block_Cr_array, axis=0)
    final_Cr_std = np.std(block_Cr_array, axis=0)
    
    return {
        "dt": dt,
        "total_frames": total_frames,
        "areal_fraction": phi,
        "dipole_strength": p_val,
        "Cr_steady_state": final_Cr_mean,
        "Cr_std": final_Cr_std,
        "bin_centers": cf.bin_centers  # Extract centers directly from Freud
    }
   
def extract_C_r_extrema(bin_centers, C_r_mean, n_extrema=3, prominence_frac=0.1):
    """
    Extract the first few significant extrema of C(r) in order, returning
    a flat, strictly numerical numpy array padded with NaNs to guarantee 
    a fixed shape of (3 * n_extrema,) for rectangular HDF5 compatibility.
    """
    nonzero = np.where(np.abs(C_r_mean) > 1e-9)[0]
    contact_idx = max(0, nonzero[0]-5) if len(nonzero) else 0
    region = C_r_mean[contact_idx:]

    prominence = prominence_frac * np.max(np.abs(region))

    maxima, _ = find_peaks(region, prominence=prominence)
    minima, _ = find_peaks(-region, prominence=prominence)

    candidates = [(i, 1.0) for i in maxima]      # 1.0 represents 'max'
    candidates += [(i, -1.0) for i in minima]    # -1.0 represents 'min'
    candidates.sort(key=lambda c: c[0])

    # Pre-populate a flat array of fixed size with NaNs
    # Each extremum needs 3 slots: [r, C, kind]
    flat_extrema = np.full(3 * n_extrema, np.nan, dtype=np.float64)

    # Fill in the extracted extrema up to n_extrema
    for idx, (i, kind_val) in enumerate(candidates[:n_extrema]):
        true_idx = contact_idx + i
        if idx == 0:
            r_refined = bin_centers[true_idx]
            C_refined = C_r_mean[true_idx]
        else:
            r_refined, C_refined = quadratic_refine(bin_centers, C_r_mean, true_idx)
        
        # Calculate the starting slice index for this specific peak
        start_slot = idx * 3
        flat_extrema[start_slot : start_slot + 3] = [r_refined, C_refined, kind_val]
        
    return flat_extrema
    
def compute_pair_angular_autocorrelation(filename, r_cutoff):
    """
    Vectorized computation of R(tau) using matrix slicing.
    """
    # 1. Load data
    with h5py.File(filename, 'r') as f:
        positions = f['positions'][:]
        angles = f['angles'][:]
        Lx, Ly = f.attrs['Lx'], f.attrs['Ly']

    T, N = angles.shape
    max_tau_frames = T // 10
    start_frame = max_tau_frames // 100
    step_t0 = max_tau_frames // 100
    
    box = freud.box.Box(Lx=Lx, Ly=Ly, is2D=True)
    
    # 3D position buffer for freud
    pos_3d_buffer = np.zeros((N, 3), dtype=np.float32)
    
    # Accumulators for the correlation sums and pair counts
    sum_cosines = np.zeros(max_tau_frames + 1, dtype=np.float64)
    total_pairs = np.zeros(max_tau_frames + 1, dtype=np.float64)

    # 2. Loop only over reference frames (t0)
    for t0 in range(start_frame, T - max_tau_frames, step_t0):
        
        pos_t0 = positions[t0]
        angs_t0 = angles[t0]
        
        # Build AABB Tree and query neighbors AT time t0
        pos_3d_buffer[:, :2] = pos_t0
        aq = freud.locality.AABBQuery(box, pos_3d_buffer)
        nl = aq.query(pos_3d_buffer, {'r_max': r_cutoff, 'exclude_ii': True}).toNeighborList()
        
        i = nl.query_point_indices
        j = nl.point_indices
        
        if len(i) == 0:
            continue
            
        # Relative angle between pair (i, j) at reference time t0
        theta_ij_t0 = angs_t0[i] - angs_t0[j]  # Shape: (n_pairs,)
        
        # 3. VECTORIZED SLICING: Grab ALL lag times (0 to max_tau) simultaneously
        # Slice shape: (max_tau_frames + 1, n_pairs)
        theta_i_future = angles[t0 : t0 + max_tau_frames + 1, i]
        theta_j_future = angles[t0 : t0 + max_tau_frames + 1, j]
        
        # Relative angle between the same pairs across all future frames
        theta_ij_future = theta_i_future - theta_j_future
        
        # Calculate the angular drift over time tau for all pairs at once
        # Using [:, np.newaxis] lets us broadcast theta_ij_t0 across the time axis
        delta_theta = theta_ij_future - theta_ij_t0[np.newaxis, :]
        
        # Sum across the pairs axis (axis=1), leaving an array of length max_tau_frames + 1
        sum_cosines += np.sum(np.cos(delta_theta), axis=1)
        total_pairs += len(i)

    # 4. Final normalization
    safe_counts = np.where(total_pairs > 0, total_pairs, 1.0)
    R_tau = np.where(total_pairs > 0, sum_cosines / safe_counts, 0.0)
    
    tau_array = np.arange(max_tau_frames + 1)
    
    return tau_array, R_tau
    
def exponential_decay(t, tau):
    return np.exp(-t / tau)

def extract_diffusive_timescale(dt, R_tau):
    frames = np.arange(len(R_tau))  # your current x-axis (0 to 2000)
    t_array = frames * dt           # convert x-axis to physical time units
    
    # 1. Define a reasonable initial guess (e.g., tau = 10 * dt)
    p0 = [100 * dt]
    
    # 2. Enforce strict physical bounds: tau cannot be negative or 0
    # Lower bound: a fraction of a timestep | Upper bound: infinite
    bounds = (dt, np.inf)
    
    # Perform the curve fit
    try:
        popt, pcov = curve_fit(
            exponential_decay, 
            t_array, 
            R_tau, 
            p0=p0, 
            bounds=bounds,
            maxfev=2000
        )
        tau_physical = popt[0]
        
        # Check if pcov is infinite (happens in some degenerate fits)
        if np.isinf(pcov).any():
            tau_error = np.nan
        else:
            tau_error = np.sqrt(pcov[0, 0])
            
    except Exception as e:
        # If the fit fails to converge, print the reason to the PBS log file
        print(f"  -> Fit failed to converge: {e}")
        
        # Return NaNs instead of crashing the job
        tau_physical = np.nan
        tau_error = np.nan
        
    return tau_physical, tau_error