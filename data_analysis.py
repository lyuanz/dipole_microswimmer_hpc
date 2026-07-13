import h5py
import numpy as np
import freud
from scipy.integrate import simpson
from scipy.signal import find_peaks


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