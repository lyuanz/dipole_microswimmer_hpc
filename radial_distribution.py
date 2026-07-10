import os
import glob
import h5py
import numpy as np
import time
import concurrent.futures

from data_analysis import extract_structural_metrics

if __name__ == "__main__":
    # 1. Setup & File Discovery
    current_dir = os.getcwd()
    search_pattern = "*.h5"
    
    output_filename = "compiled_radial_distribution_results.h5"
    all_h5_files = sorted(glob.glob(search_pattern))
    h5_files = [f for f in all_h5_files if os.path.basename(f) != output_filename]
    
    print("\n" + "-"*60)
    print(f"BATCH ANALYSIS | Target: {current_dir}")
    print("-" * 60)
    
    if not h5_files:
        print(f"Error: No files matching '{search_pattern}' found in current directory.")
        import sys
        sys.exit(1)
        
    total_files = len(h5_files)
    print(f"Found {total_files} trajectory files. Commencing multiprocessing...")
    
    # 2. Data Storage Initialization
    results_data = []       
    gr_mean_matrix = []     
    gr_std_matrix = []      
    file_names = []
    bin_centers_ref = None  
    
    start_time = time.time()
    
    # 3. Main Analysis Loop (MULTIPROCESSED)
    print("\n" + "-"*60)
    
    # max_workers=8 maps exactly to your PBS select=1:ncpus=32
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        # Submit all files to the pool
        future_to_file = {executor.submit(extract_structural_metrics, f): f for f in h5_files}
        
        completed_count = 0
        # as_completed yields files as soon as they finish, out of order
        for future in concurrent.futures.as_completed(future_to_file):
            filename = os.path.basename(future_to_file[future])
            completed_count += 1
            
            try:
                # Retrieve the dictionary returned by your function
                metrics = future.result()
                
                phi = metrics["areal_fraction"]
                p_val = metrics["dipole_strength"]
                
                results_data.append([
                    phi, p_val, 
                    metrics["peak_gr_value"], metrics["peak_radius"], 
                    metrics["cluster_radius"], metrics["coordination_number"], 
                    metrics["fraction_coordination_number"]
                ])
                gr_mean_matrix.append(metrics["gr_steady_state"])
                gr_std_matrix.append(metrics["gr_std"])
                file_names.append(filename)
                
                if bin_centers_ref is None:
                    bin_centers_ref = metrics["bin_centers"]
                    
                print(f"[{completed_count:03d}/{total_files:03d}] Processed: {filename:35s} | phi={phi:.2f} | p={p_val:.3f}")
                
            except Exception as e:
                print(f"[{completed_count:03d}/{total_files:03d}] FAILED: {filename} | Error: {e}")
            
    # 4. Unique Compressed HDF5 Export
    print("-" * 60)
    print("All files processed! Saving consolidated data...")
    
    # 1. Convert lists to raw numpy arrays (currently out of order)
    results_array = np.array(results_data, dtype=np.float64)
    gr_mean_array = np.array(gr_mean_matrix, dtype=np.float64)
    gr_std_array = np.array(gr_std_matrix, dtype=np.float64)
    filenames_array = np.array(file_names, dtype=object) 
    
    # 2. Get the index map that sorts the filenames alphabetically
    sort_idx = np.argsort(filenames_array)
    
    # 3. Apply that exact same sorting map to all arrays simultaneously
    results_array = results_array[sort_idx]
    gr_mean_array = gr_mean_array[sort_idx]
    gr_std_array = gr_std_array[sort_idx]
    filenames_array = filenames_array[sort_idx]
    
    # Define the variable-length string data type (UTF-8 is standard and safe)
    str_dtype = h5py.string_dtype(encoding='utf-8')
    
    with h5py.File(output_filename, "w") as f:
        dset_table = f.create_dataset("phase_data", data=results_array, compression="gzip", compression_opts=4)
        dset_table.attrs["columns"] = "phi, dipole_strength, peak_g_r, peak_radius, cluster_radius, coordination_number, fraction_coordination_number"
        dset_table.attrs["total_processed"] = len(results_data)
        
        f.create_dataset("gr_steady_state_curves", data=gr_mean_array, compression="gzip", compression_opts=4)
        f.create_dataset("gr_std_curves", data=gr_std_array, compression="gzip", compression_opts=4)
        f.create_dataset("bin_centers", data=bin_centers_ref, compression="gzip", compression_opts=4)
        f.create_dataset("filenames", data=filenames_array, dtype=str_dtype, compression="gzip", compression_opts=4)

    elapsed = time.time() - start_time
    print(f"Successfully saved: {output_filename} in {elapsed:.2f} seconds.")
    print("-" * 60 + "\n")