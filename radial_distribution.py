import os
import glob
import h5py
import numpy as np
import freud
import time

from data_analysis import extract_gr_quantities

if __name__ == "__main__":
    # 1. Setup & File Discovery
    # (The PBS script cd's into the scratch directory before running this)
    current_dir = os.getcwd()
    search_pattern = "swimmer_traj_*.h5"
    h5_files = sorted(glob.glob(search_pattern))
    
    print("\n" + "-"*60)
    print(f"BATCH ANALYSIS | Target: {current_dir}")
    print("-" * 60)
    
    if not h5_files:
        print(f"Error: No files matching '{search_pattern}' found in current directory.")
        import sys
        sys.exit(1)
        
    total_files = len(h5_files)
    print(f"Found {total_files} trajectory files. Commencing processing...")
    
    # 2. Data Storage Initialization
    results_data = []
    file_names = []
    start_time = time.time()
    
    # 3. Main Analysis Loop
    print("\n" + "-"*60)
    for idx, filepath in enumerate(h5_files):
        filename = os.path.basename(filepath)
        try:
            # Extract the 5 parameters
            phi, p_val, peak_gr, decay_length, relaxation_time = extract_gr_quantities(filepath)
            
            # Store them
            results_data.append([phi, p_val, peak_gr, decay_length, relaxation_time])
            file_names.append(filename)
            
            # Print standard progress update
            print(f"[{idx + 1:03d}/{total_files:03d}] Processed: {filename:35s} | phi={phi:.2f} | p={p_val:.3f}")
            
        except Exception as e:
            print(f"[{idx + 1:03d}/{total_files:03d}] FAILED: {filename} | Error: {e}")
            
    # 4. Unique Compressed HDF5 Export
    print("-" * 60)
    print("All files processed! Saving consolidated data...")
    
    output_filename = "compiled_radial_distribution_results.h5"
    
    # Convert lists to proper numpy arrays
    results_array = np.array(results_data, dtype=np.float64)
    filenames_array = np.array(file_names, dtype='S') # Store as ASCII bytes for HDF5 compatibility
    
    with h5py.File(output_filename, "w") as f:
        # Save the numerical array
        dset = f.create_dataset("phase_data", data=results_array, compression="gzip", compression_opts=4)
        
        # Attach column labels as attributes so the format is self-documenting
        dset.attrs["columns"] = "phi, dipole_strength, peak_g_r, decay_length, relaxation_time"
        dset.attrs["total_processed"] = len(results_data)
        
        # Save the filename strings so you can map rows back to specific runs
        f.create_dataset("filenames", data=filenames_array, compression="gzip", compression_opts=4)

    elapsed = time.time() - start_time
    print(f"Successfully saved: {output_filename} in {elapsed:.2f} seconds.")
    print("-" * 60 + "\n")