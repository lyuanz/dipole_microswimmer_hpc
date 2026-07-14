import h5py
import numpy as np
import freud
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def quadratic_refine(bin_centers, y_values, idx):
    """
    Sub-bin refinement of a discrete extremum at idx via a parabola fit
    through (idx-1, idx, idx+1). Returns (r_refined, y_refined) - both the
    refined location AND the refined height/depth at that location.
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
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 trajectory file.
    r_max : float
        Maximum distance to compute the correlation up to.
    n_bins : int
        Number of radial bins.
    num_blocks : int
        Number of temporal blocks to split the steady-state trajectory into for error estimation.
        
    Returns
    -------
    results : dict
        A dictionary containing the block-averaged C(r), its standard deviation, and the bin centers.
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
    
    # Define bin edges and centers
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    block_Cr_list = []

    # 3. Compute block-averaged C(r)
    for block_frames in frame_blocks:
        # Accumulators for this specific block
        block_sums = np.zeros(n_bins, dtype=np.float64)
        block_counts = np.zeros(n_bins, dtype=np.float64)
        
        for frame_idx in block_frames:
            pos = positions[frame_idx]
            angs = angles[frame_idx]
            
            # Freud requires 3D arrays
            frame_3d_cache[:, :2] = pos
            
            # Query all neighbor pairs within r_max using AABB tree
            aq = freud.locality.AABBQuery(box, frame_3d_cache)
            nl = aq.query(frame_3d_cache, {'r_max': r_max, 'exclude_ii': True}).toNeighborList()
            
            distances = nl.distances
            
            # nl.query_point_indices is 'i', nl.point_indices is 'j'
            delta_theta = angs[nl.query_point_indices] - angs[nl.point_indices]
            cos_theta = np.cos(delta_theta)
            
            # Use numpy's rapid histogram to bin the counts and the cos(theta) weights
            counts, _ = np.histogram(distances, bins=bin_edges)
            sums, _ = np.histogram(distances, bins=bin_edges, weights=cos_theta)
            
            block_counts += counts
            block_sums += sums
            
        # Calculate the mean C(r) for this block, guarding against empty bins
        safe_counts = np.where(block_counts > 0, block_counts, 1.0)
        block_Cr = np.where(block_counts > 0, block_sums / safe_counts, 0.0)
        block_Cr_list.append(block_Cr)

    # 4. Final ensemble statistics across blocks
    block_Cr_array = np.array(block_Cr_list)
    final_Cr_mean = np.mean(block_Cr_array, axis=0)
    final_Cr_std = np.std(block_Cr_array, axis=0)
    
    return {
        "time_step": dt,
        "total_frames": total_frames,
        "areal_fraction": phi,
        "dipole_strength": p_val,
        "Cr_steady_state": final_Cr_mean,
        "Cr_std": final_Cr_std,
        "bin_centers": bin_centers
    }
    
def extract_C_r_extrema(bin_centers, C_r_mean, n_extrema=3, prominence_frac=0.1):
    """
    Extract the first few significant extrema of C(r) in order, returning
    a flat, strictly numerical numpy array padded with NaNs to guarantee 
    a fixed shape of (3 * n_extrema,) for rectangular HDF5 compatibility.
    """
    nonzero = np.where(np.abs(C_r_mean) > 1e-9)[0]
    contact_idx = nonzero[0] if len(nonzero) else 0
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
        r_refined, C_refined = quadratic_refine(bin_centers, C_r_mean, true_idx)
        
        # Calculate the starting slice index for this specific peak
        start_slot = idx * 3
        flat_extrema[start_slot : start_slot + 3] = [r_refined, C_refined, kind_val]
        
    return flat_extrema
    
def compute_pair_angular_autocorrelation(filename, r_cutoff, start_frame, max_tau_frames):
    """
    Vectorized computation of R(tau) using matrix slicing.
    """
    # 1. Load data
    with h5py.File(filename, 'r') as f:
        positions = f['positions'][:]
        angles = f['angles'][:]
        Lx, Ly = f.attrs['Lx'], f.attrs['Ly']

    T, N = angles.shape
    box = freud.box.Box(Lx=Lx, Ly=Ly, is2D=True)
    
    # 3D position buffer for freud
    pos_3d_buffer = np.zeros((N, 3), dtype=np.float32)
    
    # Accumulators for the correlation sums and pair counts
    sum_cosines = np.zeros(max_tau_frames + 1, dtype=np.float64)
    total_pairs = np.zeros(max_tau_frames + 1, dtype=np.float64)

    # 2. Loop only over reference frames (t0)
    for t0 in range(start_frame, T - max_tau_frames):
        
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
    
    # Perform the curve fit
    # p0=[100.0] is an initial guess for tau in physical time units
    try:
        popt, pcov = curve_fit(exponential_decay, t_array, R_tau, p0=[100.0])
        tau_physical = popt[0]
        # Check if pcov is infinite (happens in some degenerate fits)
        if np.isinf(pcov).any():
            tau_error = np.nan
        else:
            tau_error = np.sqrt(pcov[0, 0])
            
    except RuntimeError:
        # If the fit fails to converge, return NaNs instead of crashing the job
        tau_physical = np.nan
        tau_error = np.nan
        
    return tau_physical, tau_error