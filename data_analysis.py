import h5py
import numpy as np
import freud
from scipy.integrate import simpson
from scipy.signal import find_peaks

def extract_structural_metrics(filename, r_max=8.0, n_bins=250, num_blocks=10):
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
        
        # Pre-allocate a single 3D array in memory (Z-axis remains 0.0 permanently)
        frame_3d_cache = np.zeros((N, 3), dtype=positions.dtype)
    
        # 3. Compute block-averaged g(r)
        for block_frames in frame_blocks:
            rdf = freud.density.RDF(bins=n_bins, r_max=r_max)
    
            for frame_idx in block_frames:
                # Instantly overwrite only the X and Y columns
                frame_3d_cache[:, :2] = positions[frame_idx]
                
                rdf.compute(system=(box, frame_3d_cache), reset=False)
                
            block_gr_list.append(rdf.rdf)
            
        bin_centers = rdf.bin_centers
    
    # Calculate final steady-state g(r) mean
    block_gr_array = np.array(block_gr_list)
    final_gr_mean = np.mean(block_gr_array, axis=0)
    final_gr_std  = np.std(block_gr_array, axis=0)
    
    # 4. Extract Peak Metrics
    peak_idx = np.argmax(final_gr_mean)
    peak_gr_val = final_gr_mean[peak_idx]
    r_peak = bin_centers[peak_idx]
    
    # 5. Extract Cluster Radius (First Minimum)
    inverted_gr = -final_gr_mean[peak_idx:]
    # Scale prominence to avoid registering numerical wiggling as minimas
    min_prominence = 0.05 * (peak_gr_val - 1.0)
    minima_indices, _ = find_peaks(inverted_gr, prominence=min_prominence)

    # 6. Calculate Coordination Number & Fraction Safely
    area = Lx * Ly
    rho = N / area

    if len(minima_indices) > 0:
        first_min_idx = peak_idx + minima_indices[0]
        cluster_radius = bin_centers[first_min_idx]
        
        r_cluster = bin_centers[:first_min_idx + 1]
        g_cluster = final_gr_mean[:first_min_idx + 1]
        
        integrand = g_cluster * 2 * np.pi * r_cluster * rho
        coordination_number = simpson(y=integrand, x=r_cluster)
        fraction_cn = coordination_number / (N - 1)
    else:
        # Fallback if no clear minimum (no cluster) exists
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
        "bin_centers": bin_centers
    }