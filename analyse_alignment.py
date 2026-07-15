import os
import glob
import h5py
import numpy as np
import time
import concurrent.futures
import freud
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from data_analysis import *

# ==========================================
# MASTER WRAPPER FUNCTION (FOR MULTIPROCESSING)
# ==========================================
def process_alignment_file(filename):
    # 1. C(r) Analysis
    metrics_dict = extract_orientational_correlation(filename)
    dt = metrics_dict["dt"]
    total_frames = metrics_dict["total_frames"]
    phi = metrics_dict["areal_fraction"]
    p_val = metrics_dict["dipole_strength"]
    Cr_mean = metrics_dict["Cr_steady_state"]
    Cr_std = metrics_dict["Cr_std"]
    bin_centers = metrics_dict["bin_centers"]
    
    # 2. Extrema Extraction (Directly returns a flat array of length 9)
    ext_data = extract_C_r_extrema(bin_centers, Cr_mean)
    
    # 3. Diffusive Timescale (tau)
    tau_phys, tau_err = np.nan, np.nan
    tau_array = np.array([])
    R_tau_array = np.array([])
    
    # Check if the first position element (index 0) isn't NaN to verify a peak was found
    if not np.isnan(ext_data[0]):
        r_cutoff = 1.1*ext_data[0]
        tau_array, R_tau_array = compute_pair_angular_autocorrelation(filename, r_cutoff)
        tau_phys, tau_err = extract_diffusive_timescale(dt, R_tau_array)
        
    # Build the final array row effortlessly
    scalar_metrics = [phi, p_val] + list(ext_data) + [tau_phys, tau_err]
    
    return {
        "phi": phi, 
        "p_val": p_val,
        "Cr_steady_state": Cr_mean, 
        "Cr_std": Cr_std, 
        "bin_centers": bin_centers,
        "tau_array": tau_array,
        "R_tau": R_tau_array,
        "scalar_metrics": scalar_metrics
    }

# ==========================================
# HPC EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    current_dir = os.getcwd()
    search_pattern = "*.h5"
    output_filename = "compiled_alignment_order_results.h5"
    
    all_h5_files = sorted(glob.glob(search_pattern))
    h5_files = [f for f in all_h5_files if os.path.basename(f) != output_filename]
    
    print("\n" + "-"*60)
    print(f"ALIGNMENT BATCH ANALYSIS | Target: {current_dir}")
    print("-" * 60)
    
    if not h5_files:
        print(f"Error: No files matching '{search_pattern}' found.")
        import sys
        sys.exit(1)
        
    total_files = len(h5_files)
    print(f"Found {total_files} trajectory files. Commencing multiprocessing...")
    
    # Storage Arrays (Added R_tau_matrix)
    results_data, Cr_mean_matrix, Cr_std_matrix, R_tau_matrix, file_names = [], [], [], [], []
    bin_centers_ref = None  
    tau_array_ref = None  # Reference x-axis for the time autocorrelation
    
    start_time = time.time()
    print("\n" + "-"*60)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        future_to_file = {executor.submit(process_alignment_file, f): f for f in h5_files}
        completed_count = 0
        
        for future in concurrent.futures.as_completed(future_to_file):
            filename = os.path.basename(future_to_file[future])
            completed_count += 1
            
            try:
                metrics = future.result()
                
                results_data.append(metrics["scalar_metrics"])
                Cr_mean_matrix.append(metrics["Cr_steady_state"])
                Cr_std_matrix.append(metrics["Cr_std"])
                
                R_tau_matrix.append(metrics["R_tau"]) 
                file_names.append(filename)
                
                if bin_centers_ref is None:
                    bin_centers_ref = metrics["bin_centers"]
                    
                # Store the shared time-lag axis when we hit a successful run
                if tau_array_ref is None and len(metrics["tau_array"]) > 0:
                    tau_array_ref = metrics["tau_array"]
                    
                print(f"[{completed_count:03d}/{total_files:03d}] Processed: {filename:35s} | phi={metrics['phi']:.2f} | p={metrics['p_val']:.3f}")
                
            except Exception as e:
                print(f"[{completed_count:03d}/{total_files:03d}] FAILED: {filename} | Error: {e}")
            
    print("-" * 60)
    print("All files processed! Saving consolidated data...")
    
    # If no files processed successfully or none generated a valid time series:
    if tau_array_ref is None:
        tau_array_ref = np.array([])
        
    # Standardize empty R_tau slots to match the expected length with NaNs 
    # (prevents ragged array conversions if some files didn't find peaks)
    expected_tau_len = len(tau_array_ref)
    sanitized_R_tau = []
    for r_t in R_tau_matrix:
        if len(r_t) == expected_tau_len:
            sanitized_R_tau.append(r_t)
        else:
            sanitized_R_tau.append(np.full(expected_tau_len, np.nan))

    # Format to structured numpy arrays
    results_array = np.array(results_data, dtype=np.float64)
    Cr_mean_array = np.array(Cr_mean_matrix, dtype=np.float64)
    Cr_std_array = np.array(Cr_std_matrix, dtype=np.float64)
    R_tau_array = np.array(sanitized_R_tau, dtype=np.float64)
    filenames_array = np.array(file_names, dtype=object) 
    
    # Sort alphabetically by filename
    sort_idx = np.argsort(filenames_array)
    results_array = results_array[sort_idx]
    Cr_mean_array = Cr_mean_array[sort_idx]
    Cr_std_array = Cr_std_array[sort_idx]
    R_tau_array = R_tau_array[sort_idx]
    filenames_array = filenames_array[sort_idx]
    
    str_dtype = h5py.string_dtype(encoding='utf-8')
    
    with h5py.File(output_filename, "w") as f:
        # Shape is (N, 13): phi, p_val, [r, C, kind]*3, tau_phys, tau_error
        dset_table = f.create_dataset("alignment_data", data=results_array, compression="gzip", compression_opts=4)
        dset_table.attrs["columns"] = (
            "phi, dipole_strength, "
            "extrema_1_r, extrema_1_C, extrema_1_kind (1=max,-1=min), "
            "extrema_2_r, extrema_2_C, extrema_2_kind, "
            "extrema_3_r, extrema_3_C, extrema_3_kind, "
            "tau_physical, tau_error"
        )
        
        # Matrix/Vector saves
        f.create_dataset("Cr_steady_state_curves", data=Cr_mean_array, compression="gzip", compression_opts=4)
        f.create_dataset("Cr_std_curves", data=Cr_std_array, compression="gzip", compression_opts=4)
        f.create_dataset("bin_centers", data=bin_centers_ref, compression="gzip", compression_opts=4)
        
        # New datasets for plotting R(tau)
        f.create_dataset("R_tau_curves", data=R_tau_array, compression="gzip", compression_opts=4)
        f.create_dataset("tau_array", data=tau_array_ref, compression="gzip", compression_opts=4)
        
        f.create_dataset("filenames", data=filenames_array, dtype=str_dtype, compression="gzip", compression_opts=4)

    elapsed = time.time() - start_time
    print(f"Successfully saved: {output_filename} in {elapsed:.2f} seconds.")
    print("-" * 60 + "\n")
