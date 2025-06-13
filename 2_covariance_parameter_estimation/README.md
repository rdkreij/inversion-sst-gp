### How to obtain the covariance parameter estimates in the `./outputs`

---

1. Run `./2_1_optimize_osse_per_process.py` to estimate the covariance parameters of each process for the fully observed noiseless experiment at 2014-02-19 18:00:00. This creates `./outputs/num_model_estimated_t.csv`.  
2. To estimate parameters for all the observing system simulation experiments, run the parameter estimations on an HPC system by submitting jobs located in `./sbatch_scripts/` using SLURM to parallelize the process. Specifically, running `./2_2_run_multi_optimize_osse.sh` executes `./sbatch_scripts/optimize_osse.py` for all experiments. All parameter estimates are stored as JSON files in `./intermediate`.  
3. The same sbatch approach is used for the satellite application. Running `./2_2_run_multi_optimize_osse.sh` executes `./sbatch_scripts/optimize_satellite.py`. All parameter estimates are stored as JSON files in `./intermediate`.  
4. The individual parameter estimates in `./intermediate` are combined per application per model using `./2_4_merge_outputs.py`, creating corresponding CSV files in `./outputs`.  
