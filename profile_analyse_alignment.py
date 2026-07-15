import time
import glob
import numpy as np

from data_analysis import *

filename_list = sorted(glob.glob("swimmer_*.h5"))

for i in [0, 50, 99]:
    filename = filename_list[i]
    print('-'*60)
    print(f"Profiling file {filename}")
    t0 = time.time()
    metrics_dict = extract_orientational_correlation(filename)
    print(f"extract_orientational_correlation: {time.time()-t0:.1f}s")
    
    t1 = time.time()
    ext_data = extract_C_r_extrema(metrics_dict["bin_centers"], metrics_dict["Cr_steady_state"])
    print(f"extract_C_r_extrema: {time.time()-t1:.1f}s")
    
    t2 = time.time()
    if not np.isnan(ext_data[0]):
        r_cutoff = 1.1*ext_data[0]
        tau_array, R_tau_array = compute_pair_angular_autocorrelation(
            filename, r_cutoff
        )
        print(f"compute_pair_angular_autocorrelation: {time.time()-t2:.1f}s")
        
        t3 = time.time()
        dt = metrics_dict["dt"]
        tau_phys, tau_err = extract_diffusive_timescale(dt, R_tau_array)
        print(f"extract_diffusive_timescale (FIT):    {time.time()-t3:.3f}s")
        
    print('-'*60 + '\n')